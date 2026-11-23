"""Model backends.

Every system talks to a model through :class:`Backend`. Two backends ship:

* :class:`OllamaBackend` calls an Ollama server on this machine. It refuses any
  host that is not a loopback address, so a run cannot reach a hosted service
  by accident.
* :class:`FakeBackend` is deterministic and needs no model. Tests and the fast
  reproduce mode use it.

A hosted provider can be added by writing one more class with a ``complete``
method. None is included.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import urlparse

from agon.prompts import BEAR_ROLE, BULL_ROLE, CLOSED_BOOK, PEERS_FOOTER, PEERS_HEADER


@dataclass(frozen=True)
class Request:
    """One model call. Two equal requests are the same call and run once."""

    model: str
    prompt: str
    max_tokens: int
    temperature: float
    seed: int
    system: str = ""

    def key(self) -> str:
        """Stable identifier used for the call cache and the raw log."""
        blob = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class Completion:
    """What a backend returns for a request."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool
    seconds: float = 0.0


class Backend(Protocol):
    """The whole provider interface."""

    def complete(self, request: Request) -> Completion: ...


# Decoding settings that are the same for every call in a run.
TOP_P = 0.95
TOP_K = 40
REPEAT_PENALTY = 1.0
CONTEXT_TOKENS = 4096

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class OllamaBackend:
    """Local Ollama server over HTTP. Loopback only."""

    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 600.0) -> None:
        name = urlparse(host).hostname
        if name not in _LOOPBACK:
            raise ValueError(f"OllamaBackend only talks to this machine, not {name!r}")
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._loaded: str | None = None

    def _post(self, path: str, body: Mapping[str, object]) -> dict[str, object]:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out: dict[str, object] = json.load(resp)
        return out

    def complete(self, request: Request) -> Completion:
        # One model in memory at a time: two 7B models do not fit in 10 GB of VRAM
        # next to a desktop, and a model that spills into system memory is slow.
        if self._loaded not in (None, request.model):
            self.unload(str(self._loaded))
        self._loaded = request.model
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        out = self._post(
            "/api/chat",
            {
                "model": request.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": "15m",
                "options": {
                    "temperature": request.temperature,
                    "seed": request.seed,
                    "num_predict": request.max_tokens,
                    "num_ctx": CONTEXT_TOKENS,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "repeat_penalty": REPEAT_PENALTY,
                },
            },
        )
        message = out.get("message")
        text = str(message.get("content", "")) if isinstance(message, dict) else ""
        return Completion(
            text=text,
            prompt_tokens=_as_int(out.get("prompt_eval_count")),
            completion_tokens=_as_int(out.get("eval_count")),
            truncated=out.get("done_reason") == "length",
            seconds=_as_int(out.get("total_duration")) / 1e9,
        )

    def model_info(self) -> dict[str, dict[str, str]]:
        """Digest, size and quantisation of every model the server holds."""
        req = urllib.request.Request(self.host + "/api/tags")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            tags = json.load(resp)
        info: dict[str, dict[str, str]] = {}
        for m in tags.get("models", []):
            details = m.get("details", {})
            info[str(m.get("name"))] = {
                "digest": str(m.get("digest", ""))[:12],
                "parameters": str(details.get("parameter_size", "")),
                "quantisation": str(details.get("quantization_level", "")),
                "family": str(details.get("family", "")),
            }
        return info

    def unload(self, model: str) -> None:
        """Ask the server to free a model's memory."""
        self._post("/api/generate", {"model": model, "keep_alive": 0})


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


_CLAIM_RE = re.compile(r"^Claim: (.+)$", re.MULTILINE)
_COMPANY_RE = re.compile(r"^Company: (.+)$", re.MULTILINE)
_PEER_RE = re.compile(r"ANSWER:\s*(TRUE|FALSE)")


def _unit(*parts: object) -> float:
    """Uniform number in [0, 1) that depends only on the arguments."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def fake_key(company: str, claim: str) -> str:
    """How :class:`FakeBackend` identifies a question."""
    return f"{company.strip()} | {claim.strip()}"


class FakeBackend:
    """A model with known behaviour.

    It reads the company and the claim from the prompt and looks the truth up in
    ``truth``, which is keyed by :func:`fake_key`. It
    answers correctly with probability ``skill[model]``, decided by a hash of
    the model, the seed and the claim, so the same request always gets the same
    answer. With ``herding`` above zero it copies the majority of any other
    agents' answers quoted in the prompt with that probability, which gives the
    debate something to do. One token is one whitespace-separated word, and the
    reply is cut at ``max_tokens``.
    """

    def __init__(
        self,
        truth: Mapping[str, bool],
        skill: Mapping[str, float],
        herding: float = 0.0,
        filler_words: int = 12,
    ) -> None:
        self.truth = dict(truth)
        self.skill = dict(skill)
        self.herding = herding
        self.filler_words = filler_words
        self.calls = 0

    def complete(self, request: Request) -> Completion:
        self.calls += 1
        found = _CLAIM_RE.search(request.prompt)
        company = _COMPANY_RE.search(request.prompt)
        claim = fake_key(company.group(1) if company else "", found.group(1) if found else "")
        truth = self.truth.get(claim, False)
        p_correct = self.skill.get(request.model, 0.5)
        if CLOSED_BOOK in request.prompt:
            p_correct = 0.5
        correct = _unit("skill", request.model, request.seed, claim) < p_correct
        answer = truth if correct else not truth

        if BULL_ROLE in request.prompt:
            words = [f"{request.model}-bull"] * self.filler_words + ["the", "claim", "holds"]
        elif BEAR_ROLE in request.prompt:
            words = [f"{request.model}-bear"] * self.filler_words + ["the", "claim", "fails"]
        else:
            quoted = request.prompt.partition(PEERS_HEADER)[2].partition(PEERS_FOOTER)[0]
            peers = _PEER_RE.findall(quoted)
            follow = _unit("herd", request.model, request.seed, claim) < self.herding
            if peers and follow:
                answer = peers.count("TRUE") * 2 > len(peers)
            confidence = 60 + int(40 * _unit("conf", request.model, request.seed, claim))
            words = ["step"] * self.filler_words
            words += ["ANSWER:", "TRUE" if answer else "FALSE", "CONFIDENCE:", str(confidence)]
        truncated = len(words) > request.max_tokens
        words = words[: request.max_tokens]
        prompt_tokens = len(request.prompt.split()) + len(request.system.split())
        return Completion(" ".join(words), prompt_tokens, len(words), truncated)


class RunawayBackend:
    """A broken backend that ignores ``max_tokens``. Used to test enforcement."""

    def complete(self, request: Request) -> Completion:
        n = request.max_tokens + 50
        return Completion(" ".join(["x"] * n), 1, n, False)
