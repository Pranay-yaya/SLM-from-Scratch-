"""Shared text sources for the tokenizer scripts.

  huggingface : roneneldan/TinyStories, streamed (the first N rows, same order as split[:N])
  local       : a .txt file (one story per line), a .jsonl file (one JSON object per line),
                a .csv file, or a folder containing such files
"""
import csv
import json
from pathlib import Path


def _local_texts(path, text_column, limit):
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".jsonl", ".csv"})
    count = 0
    for file in files:
        suffix = file.suffix.lower()
        with file.open("r", encoding="utf-8", errors="replace", newline="") as f:
            if suffix == ".txt":
                values = (line.rstrip("\r\n") for line in f)
            elif suffix == ".jsonl":
                values = (json.loads(line).get(text_column, "") for line in f if line.strip())
            elif suffix == ".csv":
                values = (row.get(text_column, "") for row in csv.DictReader(f))
            else:
                continue
            for value in values:
                if isinstance(value, str) and value.strip():
                    yield value
                    count += 1
                    if limit and count >= limit:
                        return


def _hf_texts(split, limit):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    count = 0
    for row in ds:
        text = row.get("text", "")
        if isinstance(text, str):
            yield text
            count += 1
            if limit and count >= limit:
                break


def iter_texts(source="huggingface", split="train", data_path=None, text_column="text", limit=0):
    """Yield raw story texts (no stripping, so results match the original pipeline)."""
    if source == "huggingface":
        return _hf_texts(split, limit)
    if source == "local":
        if data_path is None:
            raise ValueError("a data path is required when the source is local")
        return _local_texts(data_path, text_column, limit)
    raise ValueError(f"unknown source: {source}")
