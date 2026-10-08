#!/usr/bin/env python3
"""Gộp bonus/data/raw/*.jsonl thành dữ liệu sở thích cho gia sư Python gợi mở.

    python bonus/build_data.py

Sinh ra:
    data/prompts.jsonl    200 câu hỏi (id, topic, kind, prompt)
    data/pairs.parquet    200 cặp đầy đủ (id, topic, kind, prompt, chosen, rejected)
    data/train.parquet    cặp huấn luyện, định dạng hội thoại của TRL
    data/eval.parquet     cặp held-out (không trùng câu hỏi với train)
    data/stats.json       thống kê độ dài và phân bố
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from lab22 import data as D  # noqa: E402

N_EVAL = 30
SEED = 42
KEYS = ("id", "topic", "kind", "prompt", "chosen", "rejected")


def load_raw() -> list[dict]:
    rows = []
    for path in sorted((HERE / "data" / "raw").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                assert all(row.get(k) for k in KEYS), f"{path.name}: thiếu trường ở {row.get('id')}"
                rows.append({k: row[k] for k in KEYS})
    assert len({r["id"] for r in rows}) == len(rows), "trùng id"
    assert len({D.normalize_prompt(r["prompt"]) for r in rows}) == len(rows), "trùng câu hỏi"
    return rows


def main() -> int:
    import pandas as pd

    rows = load_raw()
    out = HERE / "data"
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "topic", "kind", "prompt")}, ensure_ascii=False) + "\n")
    pd.DataFrame(rows).to_parquet(out / "pairs.parquet", index=False)

    pairs = [{**D.to_conversational(r), "id": r["id"], "topic": r["topic"], "kind": r["kind"]} for r in rows]
    train, held = D.split_by_prompt(pairs, n_train=len(pairs) - N_EVAL, n_eval=N_EVAL, seed=SEED)
    D.assert_disjoint(train, held)
    pd.DataFrame(train).to_parquet(out / "train.parquet", index=False)
    pd.DataFrame(held).to_parquet(out / "eval.parquet", index=False)

    stats = {
        "n_pairs": len(rows),
        "n_train": len(train),
        "n_eval": len(held),
        "kind": {k: sum(r["kind"] == k for r in rows) for k in ("hint", "check")},
        "topics": len({r["topic"] for r in rows}),
        "length_chars": D.length_stats(pairs),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
