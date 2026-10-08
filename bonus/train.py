#!/usr/bin/env python3
"""Huấn luyện adapter DPO "gia sư Python gợi mở" trên mô hình SFT của lab.

    python bonus/build_data.py
    python bonus/train.py                 # → adapters/dpo-bonus/

Cùng công thức với NB3 (LoRA mới trên models/sft-merged, tính sẵn log-xác suất
tham chiếu). Khác NB3 ở hai chỗ:
  - dữ liệu chỉ có 170 cặp, nên chạy nhiều epoch hơn với tốc độ học cao hơn;
  - loss mặc định là RPO (sigmoid + sft). Lượt đầu chỉ dùng DPO (--loss sigmoid
    --epochs 3 --lr 2e-5) xếp hạng đúng 30/30 cặp held-out nhưng câu trả lời
    sinh ra gần như không đổi; thêm NLL trên câu chosen để kéo cả cách viết.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss", default="sigmoid,sft", help="TRL loss_type; sigmoid = DPO, sigmoid,sft = RPO")
    parser.add_argument("--output-dir", default=str(REPO / "adapters" / "dpo-bonus"))
    args = parser.parse_args()

    import unsloth  # noqa: F401  (patch trl/transformers first)
    from datasets import Dataset
    from trl import DPOTrainer

    from lab22 import config as C
    from lab22 import modeling as MD

    output = Path(args.output_dir)
    cols = ["prompt", "chosen", "rejected"]
    kwargs = dict(C.CHAT_TEMPLATE_KWARGS)
    train_ds = Dataset.from_parquet(str(HERE / "data" / "train.parquet")).select_columns(cols)
    eval_ds = Dataset.from_parquet(str(HERE / "data" / "eval.parquet")).select_columns(cols)
    train_ds = train_ds.map(lambda r: {"chat_template_kwargs": kwargs})
    eval_ds = eval_ds.map(lambda r: {"chat_template_kwargs": kwargs})
    print(f"train={len(train_ds)} eval={len(eval_ds)} epochs={args.epochs} lr={args.lr} beta={args.beta}")

    model, tokenizer = MD.load_model(C.SFT_MERGED)
    model = MD.add_lora(model)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=MD.dpo_config(
            output.parent / f"{output.name}-checkpoints",
            loss_type=args.loss.split(","),
            beta=args.beta,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            gradient_accumulation_steps=4,
            eval_steps=20,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    result = trainer.train()
    ev = trainer.evaluate()

    output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    train_hist = MD.reward_history(trainer.state.log_history)
    eval_hist = MD.reward_history(trainer.state.log_history, prefix="eval_")
    MD.plot_rewards(train_hist, eval_hist, f"Gia sư Python · {args.loss} · β={args.beta} lr={args.lr}", HERE / "reward-curves.png")
    end = train_hist.iloc[-1] if not train_hist.empty else {}
    metrics = {
        "base_model": C.BASE_MODEL,
        "reference": "models/sft-merged (precomputed)",
        "pref_dataset": "bonus/data (gia sư Python gợi mở, 170 train / 30 held-out)",
        "loss_type": args.loss.split(","),
        "beta": args.beta,
        "lr": args.lr,
        "epochs": args.epochs,
        "final_train_loss": float(result.training_loss),
        "end_chosen_reward": float(end["rewards/chosen"]) if len(end) else None,
        "end_rejected_reward": float(end["rewards/rejected"]) if len(end) else None,
        "end_reward_gap": float(end["rewards/margins"]) if len(end) else None,
        "eval_chosen_reward": ev.get("eval_rewards/chosen"),
        "eval_rejected_reward": ev.get("eval_rewards/rejected"),
        "eval_reward_gap": ev.get("eval_rewards/margins"),
        "eval_reward_accuracy": ev.get("eval_rewards/accuracies"),
        "diagnosis": MD.diagnose(eval_hist if len(eval_hist) >= 2 else train_hist)[0],
    }
    (output / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
