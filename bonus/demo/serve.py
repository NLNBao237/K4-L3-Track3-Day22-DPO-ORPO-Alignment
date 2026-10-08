#!/usr/bin/env python3
"""Trò chuyện với gia sư Python gợi mở ngay trong cửa sổ dòng lệnh.

    python bonus/demo/serve.py                      # GPU: models/sft-merged + adapters/dpo-bonus
    python bonus/demo/serve.py --gguf path/to.gguf  # CPU: file GGUF qua llama-cpp-python
    python bonus/demo/serve.py --baseline           # chỉ mô hình SFT, để so sánh

Gõ câu hỏi rồi Enter; dòng trống hoặc "thoát" để dừng.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def hf_backend(baseline: bool):
    import unsloth  # noqa: F401

    from lab22 import config as C
    from lab22 import modeling as MD

    model, tokenizer = MD.load_model(C.SFT_MERGED if baseline else REPO / "adapters" / "dpo-bonus")
    return lambda question: MD.generate(model, tokenizer, [question], max_new_tokens=384)[0]


def gguf_backend(path: str):
    from llama_cpp import Llama

    llm = Llama(model_path=path, n_ctx=2048, n_gpu_layers=-1, verbose=False)

    def answer(question: str) -> str:
        # Qwen3 lai: "/no_think" tắt chế độ suy nghĩ khi chạy qua chat template của GGUF.
        messages = [{"role": "user", "content": question + " /no_think"}]
        out = llm.create_chat_completion(messages, max_tokens=384, temperature=0.0)
        return out["choices"][0]["message"]["content"].replace("<think>\n\n</think>", "").strip()

    return answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default=None, help="đường dẫn file GGUF (chạy CPU)")
    parser.add_argument("--baseline", action="store_true", help="dùng mô hình SFT chưa qua DPO")
    args = parser.parse_args()

    answer = gguf_backend(args.gguf) if args.gguf else hf_backend(args.baseline)
    print("Gia sư Python (bản thử nghiệm v0). Gõ câu hỏi, dòng trống để thoát.\n")
    while True:
        try:
            question = input("Em hỏi > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"thoát", "thoat", "exit", "quit"}:
            break
        print(f"\nGia sư > {answer(question)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
