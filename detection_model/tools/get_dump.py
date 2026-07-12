"""Download dated public Poker44 benchmark chunks.

Example:
    python -m detection_model.tools.get_dump --start-date 2026-07-06 --end-date 2026-07-12
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

API_URL = "https://api.poker44.net/api/v1/benchmark/chunks"


def _dates(start: str, end: str):
    current = datetime.strptime(start, "%Y-%m-%d").date()
    stop = datetime.strptime(end, "%Y-%m-%d").date()
    if stop < current:
        raise ValueError("end date is before start date")
    while current <= stop:
        yield current.isoformat()
        current += timedelta(days=1)


def _fetch(session: requests.Session, source_date: str, timeout: int):
    response = session.get(API_URL, params={"sourceDate": source_date}, timeout=timeout)
    response.raise_for_status()
    dumps = (((response.json() or {}).get("data") or {}).get("chunks") or [])
    items = []
    for dump in dumps:
        if not isinstance(dump, dict):
            continue
        chunks, labels = dump.get("chunks") or [], dump.get("groundTruth") or []
        if len(chunks) != len(labels):
            raise ValueError(f"API chunk/label mismatch for {source_date}")
        for hands, label in zip(chunks, labels):
            items.append({"hands": hands, "is_bot": bool(label), "source_date": source_date})
    return items


def main() -> None:
    today = date.today().isoformat()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-date", default="2026-07-06")
    ap.add_argument("--end-date", default=today)
    ap.add_argument("--out-dir", default="detection_model/data_v3")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    combined = []
    for source_date in _dates(args.start_date, args.end_date):
        path = out_dir / f"benchmark_chunks_{source_date}.json"
        if path.exists() and not args.overwrite:
            items = json.loads(path.read_text(encoding="utf-8"))
            status = "cached"
        else:
            items = _fetch(session, source_date, args.timeout)
            path.write_text(json.dumps(items, separators=(",", ":")), encoding="utf-8")
            status = "downloaded"
        bots = sum(bool(item.get("is_bot")) for item in items)
        print(f"{source_date}: {status} {len(items)} chunks (bot={bots}, human={len(items)-bots})")
        combined.extend(items)

    combined_path = out_dir / f"benchmark_chunks_{args.start_date}_to_{args.end_date}.json"
    combined_path.write_text(json.dumps(combined, separators=(",", ":")), encoding="utf-8")
    print(f"Combined {len(combined)} chunks -> {combined_path}")


if __name__ == "__main__":
    main()

