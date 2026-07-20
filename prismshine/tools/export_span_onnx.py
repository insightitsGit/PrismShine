#!/usr/bin/env python3
"""Export LettuceDetect-class token classifier to ONNX for PrismShine Tier-3.

Usage:
  pip install \"prismshine[spans]\" torch transformers optimum onnx
  python -m prismshine.tools.export_span_onnx --out models/lettucedetect

Then pin:
  set PRISMSHINE_SPAN_ONNX=models/lettucedetect/model.onnx
  set PRISMSHINE_SPAN_TOKENIZER=models/lettucedetect/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


def export(model_id: str, out_dir: Path, opset: int = 17, max_length: int = 512) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "export requires torch + transformers. "
            "pip install torch transformers onnx"
        ) from exc

    print(f"loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval()

    # Save tokenizer.json for tokenizers.Tokenizer.from_file
    tokenizer.save_pretrained(out_dir)
    tok_json = out_dir / "tokenizer.json"
    if not tok_json.is_file():
        # slow tokenizers may only write vocab — try convert
        raise SystemExit(f"tokenizer.json missing under {out_dir}; use a fast tokenizer model")

    onnx_path = out_dir / "model.onnx"
    # Dummy batch: token-classification over context+question+answer style sequence
    dummy = tokenizer(
        "context passage here",
        "question? answer tokens here",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_names = ["input_ids", "attention_mask"]
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }
    if "token_type_ids" in dummy:
        input_names.append("token_type_ids")
        inputs = (*inputs, dummy["token_type_ids"])
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "sequence"}

    print(f"exporting ONNX -> {onnx_path}")
    # dynamo=False: ModernBERT Split(num_outputs) breaks older onnxruntime graphs
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    meta = {
        "model_id": model_id,
        "opset": opset,
        "max_length": max_length,
        "onnx": str(onnx_path.name),
        "tokenizer": "tokenizer.json",
        "note": "Pin via PRISMSHINE_SPAN_ONNX / PRISMSHINE_SPAN_TOKENIZER",
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", meta)
    return onnx_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=Path("models/lettucedetect"))
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()
    export(args.model, args.out, opset=args.opset, max_length=args.max_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
