#!/usr/bin/env python3
"""Turn stories into token ids (train.bin / val.bin) with a tokenizer you choose.

It does NOT train or change any tokenizer, so it can never overwrite your existing tokenizer.json.
Output folder: train.bin, val.bin, meta.json (vocab size etc.) and a copy of the tokenizer.

Examples
  python encode_data.py --tokenizer tokenizer_new_8k/tokenizer.json --out-dir data_new_8k
  python encode_data.py --tokenizer tokenizer.json --out-dir data_old_8k   # should reproduce the old
                                                                           # 44,290,410 / 983,707 tokens
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tok_data import iter_texts

EOS = "<|endoftext|>"


def encode_to_bin(tok, texts, out_path, eos_id, batch=5000):
    stories = total = 0
    buf = []
    with open(out_path, "wb") as f:
        def flush():
            nonlocal stories, total
            if not buf:
                return
            arrays = [np.array(e.ids + [eos_id], dtype=np.uint16) for e in tok.encode_batch(buf)]
            np.concatenate(arrays).tofile(f)
            total += sum(len(a) for a in arrays)
            stories += len(buf)
            buf.clear()

        for text in texts:
            buf.append(text)
            if len(buf) >= batch:
                flush()
        flush()
    return stories, total


def main():
    ap = argparse.ArgumentParser(description="Encode stories into train.bin / val.bin")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--source", choices=["huggingface", "local"], default="huggingface")
    ap.add_argument("--data-path", type=Path, help="local training data (file or folder)")
    ap.add_argument("--val-data-path", type=Path, help="local validation data (file or folder)")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--n-val", type=int, default=5_000)
    args = ap.parse_args()

    if args.source == "local" and (args.data_path is None or args.val_data_path is None):
        ap.error("--data-path and --val-data-path are required when --source local")

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id(EOS)
    vocab = tok.get_vocab_size()
    if eos_id is None:
        raise SystemExit(f"{EOS} is missing from this tokenizer")
    if vocab > 65535:
        raise SystemExit("vocabulary is too large for the uint16 .bin format")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tr = iter_texts(args.source, "train", args.data_path, args.text_column, args.n_train)
    va = iter_texts(args.source, "validation", args.val_data_path, args.text_column, args.n_val)
    n_tr, tok_tr = encode_to_bin(tok, tr, args.out_dir / "train.bin", eos_id)
    n_va, tok_va = encode_to_bin(tok, va, args.out_dir / "val.bin", eos_id)

    for name in ("train.bin", "val.bin"):  # safety check: every id must be inside the vocabulary
        arr = np.memmap(args.out_dir / name, dtype=np.uint16, mode="r")
        if int(arr.max()) >= vocab:
            raise SystemExit(f"{name} contains an id outside the vocabulary")

    dest = args.out_dir / "tokenizer.json"
    if Path(args.tokenizer).resolve() != dest.resolve():
        shutil.copy(args.tokenizer, dest)
    meta = {"vocab_size": vocab, "eos_id": eos_id, "dtype": "uint16",
            "tokenizer_file": "tokenizer.json", "source": args.source,
            "train_stories": n_tr, "val_stories": n_va,
            "train_tokens": tok_tr, "val_tokens": tok_va}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"train.bin: {n_tr:,} stories, {tok_tr:,} tokens")
    print(f"val.bin:   {n_va:,} stories, {tok_va:,} tokens")
    print(f"vocab size {vocab}, end-of-text id {eos_id}. Saved to {args.out_dir.resolve()}")
    print(f"\nNext: python train.py --data_dir {args.out_dir} --out ckpt_NAME.pt --log loss_NAME.csv")


if __name__ == "__main__":
    main()
