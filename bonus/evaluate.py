#!/usr/bin/env python3
"""So sánh SFT và SFT+DPO-gia-sư trên 30 câu hỏi held-out bằng các chỉ số hành vi.

    python bonus/evaluate.py              # → bonus/eval_results.json, bonus/demo/5-samples.md

Không có giáo viên thật chấm, nên ở đây đo ba hành vi quan sát được, khớp với
ba điều mà dữ liệu sở thích muốn dạy:
  - gives_full_code : câu trả lời có khối code từ 60 ký tự trở lên (đưa thẳng lời giải;
                      mô hình 1.7B hay viết cả khối code trên một dòng nên không đếm theo dòng)
  - asks_back       : câu trả lời có ít nhất một câu hỏi ngược lại cho học sinh
  - english_jargon  : số từ tiếng Anh thừa (ngoài code) trên mỗi câu trả lời
Các chỉ số này là heuristic; đọc kèm 5 mẫu trong demo/5-samples.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

JARGON = re.compile(
    r"\b(loop|iterate|check|condition|list|string|function|return|output|input|code|fix|bug|debug|run|"
    r"here's|solution|variable|value|index|array|step|copy|paste|convert|cast|file|method|class|object|"
    r"basic|simple|just|use|using|call|pass|error|issue|trace|skill|review|update|store|add|remove)\b",
    re.IGNORECASE,
)
FENCE = re.compile(r"```.*?```", re.DOTALL)


def behaviours(text: str) -> dict:
    blocks = FENCE.findall(text)
    prose = re.sub(r"`[^`]*`", " ", FENCE.sub(" ", text))
    return {
        "gives_full_code": any(len(re.sub(r"^```\w*|```$", "", b).strip()) >= 60 for b in blocks),
        "asks_back": "?" in prose,
        "english_jargon": len(JARGON.findall(prose)),
        "chars": len(text),
    }


def aggregate(texts: list[str]) -> dict:
    stats = [behaviours(t) for t in texts]
    n = len(stats)
    return {
        "n": n,
        "gives_full_code_frac": sum(s["gives_full_code"] for s in stats) / n,
        "asks_back_frac": sum(s["asks_back"] for s in stats) / n,
        "english_jargon_per_answer": sum(s["english_jargon"] for s in stats) / n,
        "mean_chars": sum(s["chars"] for s in stats) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(REPO / "adapters" / "dpo-bonus"))
    parser.add_argument("--tag", default="", help="hậu tố tên file kết quả, ví dụ -v1-dpo")
    args = parser.parse_args()

    import unsloth  # noqa: F401
    import pandas as pd

    from lab22 import config as C
    from lab22 import modeling as MD

    adapter = Path(args.adapter)
    held = pd.read_parquet(HERE / "data" / "eval.parquet").to_dict("records")
    prompts = [r["prompt"][0]["content"] for r in held]

    model, tokenizer = MD.load_model(C.SFT_MERGED)
    sft = MD.generate(model, tokenizer, prompts, batch_size=4)
    del model
    MD.cleanup()
    model, tokenizer = MD.load_model(adapter)
    dpo = MD.generate(model, tokenizer, prompts, batch_size=4)
    del model
    MD.cleanup()

    rows = [
        {"id": r["id"], "topic": r["topic"], "kind": r["kind"], "prompt": p, "sft": s, "dpo": d}
        for r, p, s, d in zip(held, prompts, sft, dpo)
    ]
    hints = [r for r in rows if r["kind"] == "hint"]
    checks = [r for r in rows if r["kind"] == "check"]
    held_hints = [r for r in held if r["kind"] == "hint"]
    summary = {
        "n_heldout": len(rows),
        "hint": {
            "sft": aggregate([r["sft"] for r in hints]),
            "dpo": aggregate([r["dpo"] for r in hints]),
        },
        "check": {
            "sft": aggregate([r["sft"] for r in checks]),
            "dpo": aggregate([r["dpo"] for r in checks]),
        } if checks else None,
        # Mốc tham chiếu: chính các câu chosen/rejected do người soạn viết, đo bằng cùng heuristic.
        "reference_labels": {
            "chosen": aggregate([r["chosen"][0]["content"] for r in held_hints]),
            "rejected": aggregate([r["rejected"][0]["content"] for r in held_hints]),
        },
    }
    (HERE / f"eval_results{args.tag}.json").write_text(
        json.dumps({"summary": summary, "records": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    picks = hints[:4] + checks[:1]
    lines = ["# 5 mẫu trước/sau DPO (câu hỏi held-out, giải mã tham lam)\n"]
    for i, r in enumerate(picks, 1):
        lines += [
            f"## Mẫu {i} — `{r['id']}` · {r['topic']} · {r['kind']}\n",
            f"**Học sinh hỏi:**\n\n{r['prompt']}\n",
            f"**Trước (SFT):**\n\n{r['sft']}\n",
            f"**Sau (SFT + DPO gia sư):**\n\n{r['dpo']}\n",
            "---\n",
        ]
    (HERE / "demo" / f"5-samples{args.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved bonus/eval_results{args.tag}.json and bonus/demo/5-samples{args.tag}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
