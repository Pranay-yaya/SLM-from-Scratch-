#!/usr/bin/env python3
"""Train the project tokenizer: a byte-level BPE for TinyStories-style English.

Combines the useful parts of three earlier versions:
  * byte-level BPE: any text can be encoded, there is never an unknown token
  * optional LIGHT punctuation normalisation (curly quotes, dashes, ellipsis, NFC)
  * optional single-digit splitting
  * special tokens with fixed ids; end-of-text is id 0 (train.py and generate.py rely on that)
  * Hugging Face or local data, and a census of unusual characters in the training text
  * Hugging Face style config files saved next to tokenizer.json

Examples
  python train_tokenizer.py --out-dir tokenizer_new_8k
  python train_tokenizer.py --vocab-size 16000 --out-dir tokenizer_16k
  python train_tokenizer.py --normalize none --no-digits --out-dir tokenizer_plain
  python train_tokenizer.py --source local --data-path stories.jsonl --out-dir tok_local
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, Regex, models, normalizers, pre_tokenizers, decoders, trainers

from tok_data import iter_texts

# Order matters: the trainer gives these ids 0..4. End-of-text MUST stay id 0.
SPECIAL = ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|im_start|>", "<|im_end|>"]

# TinyStories contains UTF-8 punctuation that was wrongly decoded as Windows-1252, so an apostrophe
# shows up as the three characters a-circumflex, euro sign, trademark sign. These sequences never occur
# in genuine English text, so repairing them is safe. Order matters: specific rules first.
MOJIBAKE_MARK = "\u00e2\u20ac"
MOJIBAKE = [
    ("\u00e2\u20ac\u2122", "'"),    # right single quote / apostrophe
    ("\u00e2\u20ac\u02dc", "'"),    # left single quote
    ("\u00e2\u20ac\u0153", '"'),    # left double quote
    ("\u00e2\u20ac\u009d", '"'),    # right double quote
    ("\u00e2\u20ac\u201c", "-"),    # en dash
    ("\u00e2\u20ac\u201d", "-"),    # em dash
    ("\u00e2\u20ac\u00a6", "..."),  # ellipsis
    (MOJIBAKE_MARK, '"'),            # leftover: the right double quote often loses its last byte
]


def build_normalizer(mode):
    """'light' repairs the corrupted quote characters above and turns curly quotes, dashes and
    the ellipsis into plain ones. It is deliberately NOT NFKC, which rewrites many characters."""
    if mode == "none":
        return None
    return normalizers.Sequence([
        normalizers.NFC(),
        *[normalizers.Replace(bad, good) for bad, good in MOJIBAKE],
        normalizers.Replace(Regex("[\u2018\u2019\u201b\u2032]"), "'"),
        normalizers.Replace(Regex("[\u201c\u201d\u201f\u2033]"), '"'),
        normalizers.Replace(Regex("[\u2013\u2014\u2212]"), "-"),
        normalizers.Replace("\u2026", "..."),
        normalizers.Replace("\u00a0", " "),
    ])


def build_pretokenizer(split_digits):
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)  # GPT-2 style regex
    if split_digits:
        return pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), byte_level])
    return byte_level


class Census:
    """Wraps the text stream and counts stories and non-ASCII characters while training reads it."""

    def __init__(self):
        self.stories = 0
        self.chars = 0
        self.mojibake = 0
        self.non_ascii = Counter()

    def __call__(self, texts):
        for text in texts:
            self.stories += 1
            self.chars += len(text)
            if not text.isascii():
                self.mojibake += text.count(MOJIBAKE_MARK)
                for ch in text:
                    if ord(ch) > 127:
                        self.non_ascii[ch] += 1
            yield text


def main():
    ap = argparse.ArgumentParser(description="Train the project tokenizer")
    ap.add_argument("--source", choices=["huggingface", "local"], default="huggingface")
    ap.add_argument("--data-path", type=Path, help="file or folder, for --source local")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--max-stories", type=int, default=200_000,
                    help="default matches the 200,000 stories the model trains on")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--normalize", choices=["none", "light"], default="light")
    ap.add_argument("--digits", action=argparse.BooleanOptionalAction, default=True,
                    help="split numbers into single digits (use --no-digits to turn off)")
    ap.add_argument("--context-length", type=int, default=256, help="written to the config only")
    ap.add_argument("--out-dir", type=Path, default=Path("tokenizer_new_8k"))
    args = ap.parse_args()

    if args.vocab_size <= 256 + len(SPECIAL):
        ap.error("--vocab-size must be larger than 256 bytes plus the special tokens")
    if args.source == "local" and args.data_path is None:
        ap.error("--data-path is required when --source local")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE())  # no unk token: byte-level BPE never needs one
    norm = build_normalizer(args.normalize)
    if norm is not None:
        tok.normalizer = norm
    tok.pre_tokenizer = build_pretokenizer(args.digits)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    census = Census()
    stream = iter_texts(args.source, "train", args.data_path, args.text_column, args.max_stories)
    print(f"Training: vocab={args.vocab_size}, normalize={args.normalize}, digits={args.digits}, "
          f"stories<={args.max_stories:,}")
    tok.train_from_iterator(census(stream), trainer=trainer)

    ids = {t: tok.token_to_id(t) for t in SPECIAL}
    if ids != {t: i for i, t in enumerate(SPECIAL)}:
        raise SystemExit(f"Special token ids are wrong: {ids}. End-of-text must be id 0.")

    tok.save(str(args.out_dir / "tokenizer.json"))
    tok.model.save(str(args.out_dir))  # vocab.json and merges.txt, for tools that expect them

    hf_tokens = {"bos_token": "<|bos|>", "eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}
    (args.out_dir / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": args.context_length, **hf_tokens},
        indent=2), encoding="utf-8")
    (args.out_dir / "special_tokens_map.json").write_text(json.dumps(hf_tokens, indent=2), encoding="utf-8")

    actual = tok.get_vocab_size()
    info = {
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual,
        "model": "ByteLevel BPE",
        "normalize": args.normalize,
        "split_digits": args.digits,
        "min_frequency": args.min_frequency,
        "source": args.source,
        "stories_used": census.stories,
        "characters_seen": census.chars,
        "corrupted_quote_sequences_seen": census.mojibake,
        "special_token_ids": ids,
        "top_non_ascii_characters": [[c, n] for c, n in census.non_ascii.most_common(30)],
        "note": "Actual vocabulary is smaller than requested only if the corpus cannot support that many merges.",
    }
    (args.out_dir / "training_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")

    print(f"\nSaved to {args.out_dir.resolve()}")
    print(f"Vocabulary size: {actual} (requested {args.vocab_size}); special ids: {ids}")
    print(f"Stories read: {census.stories:,}; characters: {census.chars:,}")
    print(f"Embedding table at n_embd=384: {actual * 384 / 1e6:.2f}M parameters "
          f"(tied with the output layer)")
    if census.non_ascii:
        top = ", ".join(f"{c!r}x{n:,}" for c, n in census.non_ascii.most_common(12))
        print(f"Non-ASCII characters in the training text: {sum(census.non_ascii.values()):,} total. Top: {top}")
        if census.mojibake:
            print(f"Corrupted quote sequences found in the text: {census.mojibake:,}.")
            if args.normalize == "light":
                print("  The light normalizer repairs them (apostrophes, quotes, dashes, ellipsis).")
            else:
                print("  WARNING: --normalize none keeps them, so the tokenizer will learn them as tokens.")
    else:
        print("The training text is pure ASCII, so the normalizer has nothing to change here.")
    if actual < args.vocab_size:
        print("Warning: the corpus could not support the requested vocabulary size.")
    print("\nNext: python evaluate_tokenizer.py --tokenizer", args.out_dir / "tokenizer.json")


if __name__ == "__main__":
    main()
