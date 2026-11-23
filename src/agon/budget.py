"""Token budgets that are enforced, not assumed.

Each system gets one :class:`Ledger` per question. The ledger counts generated
(completion) tokens. A system must reserve tokens before a call and settle the
real count after it. Asking for more than is left raises, and so does a backend
that returns more tokens than the call was allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetError(RuntimeError):
    """A system or a backend broke the token budget."""


@dataclass
class Ledger:
    """Completion-token account for one system on one question."""

    budget: int
    spent: int = 0
    prompt_tokens: int = 0
    calls: int = 0
    _reserved: int = field(default=0, repr=False)

    @property
    def remaining(self) -> int:
        return self.budget - self.spent - self._reserved

    def reserve(self, max_tokens: int) -> None:
        """Set aside ``max_tokens`` for a call that is about to be made."""
        if max_tokens <= 0:
            raise BudgetError("a call must be allowed at least one token")
        if max_tokens > self.remaining:
            raise BudgetError(
                f"call wants {max_tokens} tokens but only {self.remaining} of {self.budget} remain"
            )
        self._reserved += max_tokens

    def settle(self, max_tokens: int, completion_tokens: int, prompt_tokens: int) -> None:
        """Replace a reservation with what the call really used."""
        if completion_tokens > max_tokens:
            raise BudgetError(
                f"backend returned {completion_tokens} tokens for a call capped at {max_tokens}"
            )
        if max_tokens > self._reserved:
            raise BudgetError("settling a call that was never reserved")
        self._reserved -= max_tokens
        self.spent += completion_tokens
        self.prompt_tokens += prompt_tokens
        self.calls += 1
