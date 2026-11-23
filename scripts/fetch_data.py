"""Download the SEC XBRL frames the question set is built from.

Source: https://data.sec.gov/api/xbrl/frames/ (public, no key, no account).
Files land in data/frames/, which git ignores. About fifty small requests,
spaced out to stay far below the SEC's limit of ten per second.

The frames for a period keep growing while companies are still filing for it,
so a download on a later date can hold more companies than the one the
committed results used. results/questions.csv records the filings that were
used.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from agon.filings import frame_names

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json"
# A plain client name. No personal details are sent.
USER_AGENT = "agon-research-script/0.1"


def fetch(concept: str, period: str, out: Path, pause: float) -> str:
    target = out / f"{concept}_{period}.json"
    if target.exists():
        return "kept"
    request = urllib.request.Request(
        URL.format(concept=concept, period=period), headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            payload = {"data": []}
        else:
            raise
    finally:
        time.sleep(pause)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return f"{len(payload.get('data', []))} facts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "frames")
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for concept, period in frame_names():
        print(f"{concept} {period}: {fetch(concept, period, args.out, args.pause)}")


if __name__ == "__main__":
    main()
