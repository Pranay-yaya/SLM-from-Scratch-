#!/usr/bin/env python3
"""Tokenizer checks and a fair comparison.

1. special tokens (end-of-text must be id 0)
2. lossless round trip on tricky English text (checked against the normalised text, so a
   normalizer that rewrites quotes is not reported as a failure)
3. compression on REAL stories: characters/bytes per token, tokens per story,
   how many stories fit in the model context, how much of the vocabulary is actually used
4. a multilingual round trip (information only: the data is English)

Examples
  python evaluate_tokenizer.py --tokenizer tokenizer_new_8k/tokenizer.json
  python evaluate_tokenizer.py --tokenizer tokenizer_new_8k/tokenizer.json --compare tokenizer.json
  python evaluate_tokenizer.py --tokenizer tok.json --skip-data     # no internet needed
"""
import argparse
import statistics
from pathlib import Path

from tokenizers import Tokenizer

from tok_data import iter_texts

ROUND_TRIP = {
    "plain": "Once upon a time, a little girl found a tiny bird in her garden.",
    "dialogue": 'Mummy said, "Of course, Tim!" Tim smiled.',
    "contractions": "Don't worry, she's fine. We'll see what they've done.",
    "curly quotes": "She said, \u201cI can\u2019t go.\u201d",
    "numbers": "Tom was 5 years old and had 12 red balls in 2024.",
    "newlines": "First line.\n\nSecond paragraph.\n  Indented line.",
    "emoji": "The dog was happy \U0001F436 and the sun was bright \u2600\ufe0f",
    "rare characters": "caf\u00e9 na\u00efve \u2014 a long dash \u2026 and an ellipsis",
    "corrupted quotes": "He didn\u00e2\u20ac\u2122t go. She said \u00e2\u20ac\u0153hi\u00e2\u20ac\u009d.",
}
MOJIBAKE_MARK = "\u00e2\u20ac"

MULTILINGUAL = {
    "English": "Once upon a time, a little girl found a tiny bird in her garden.",
    "Hindi": "एक छोटी बच्ची को अपने बगीचे में एक सुंदर चिड़िया मिली।",
    "Punjabi": "ਇੱਕ ਛੋਟੀ ਕੁੜੀ ਨੂੰ ਆਪਣੇ ਬਾਗ ਵਿੱਚ ਇੱਕ ਪੰਛੀ ਮਿਲਿਆ।",
    "Bengali": "একটি ছোট মেয়ে তার বাগানে একটি পাখি খুঁজে পেল।",
    "Tamil": "ஒரு சிறுமி தனது தோட்டத்தில் ஒரு சிறிய பறவையைக் கண்டாள்.",
    "Telugu": "ఒక చిన్న అమ్మాయి తన తోటలో ఒక చిన్న పక్షిని చూసింది.",
    "Urdu": "ایک چھوٹی لڑکی کو اپنے باغ میں ایک پرندہ ملا۔",
}


def expected_text(tok, text):
    return tok.normalizer.normalize_str(text) if tok.normalizer is not None else text


def check_round_trip(tok, tests, title, verbose_chars=False):
    print(f"\n{title}")
    failures = 0
    for name, text in tests.items():
        ids = tok.encode(text).ids
        decoded = tok.decode(ids)
        ok = decoded == expected_text(tok, text)
        failures += not ok
        extra = f" tokens/char={len(ids) / max(1, len(text)):.2f}" if verbose_chars else ""
        print(f"  {name:16s} tokens={len(ids):4d}{extra}  round_trip={'OK' if ok else 'FAIL'}")
        if not ok:
            print("    expected:", repr(expected_text(tok, text)))
            print("    decoded :", repr(decoded))
    return failures


def corpus_stats(tok, texts, context_length, batch=1000):
    vocab = tok.get_vocab_size()
    counts = [0] * vocab
    lengths = []
    n_chars = n_bytes = n_words = 0
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        for text, enc in zip(chunk, tok.encode_batch(chunk)):
            lengths.append(len(enc.ids))
            n_chars += len(text)
            n_bytes += len(text.encode("utf-8"))
            n_words += len(text.split())
            for tid in enc.ids:
                counts[tid] += 1
    n_tokens = sum(lengths)
    lengths_sorted = sorted(lengths)
    raw_bad = sum(t.count(MOJIBAKE_MARK) for t in texts)
    if tok.normalizer is not None:
        left_bad = sum(tok.normalizer.normalize_str(t).count(MOJIBAKE_MARK) for t in texts)
    else:
        left_bad = raw_bad
    return {
        "vocabulary size": vocab,
        "stories measured": len(texts),
        "characters per token": n_chars / n_tokens,
        "bytes per token": n_bytes / n_tokens,
        "tokens per word": n_tokens / max(1, n_words),
        "tokens per story (mean)": statistics.mean(lengths),
        "tokens per story (median)": statistics.median(lengths),
        "tokens per story (95th pct)": lengths_sorted[int(0.95 * (len(lengths_sorted) - 1))],
        f"stories within {context_length} tokens (%)": 100 * sum(l <= context_length for l in lengths) / len(lengths),
        "vocabulary ids used (%)": 100 * sum(c > 0 for c in counts) / vocab,
        "ids used fewer than 5 times": sum(0 < c < 5 for c in counts),
        "ids never used": sum(c == 0 for c in counts),
        "corrupted quote sequences in text": raw_bad,
        "  ...still there after tokenizer": left_bad,
    }


def print_table(names, all_stats):
    keys = list(all_stats[0].keys())
    width = max(len(k) for k in keys) + 2
    print("\n" + " " * width + "".join(f"{n:>22s}" for n in names))
    for k in keys:
        row = ""
        for s in all_stats:
            v = s[k]
            row += f"{v:>22,}" if isinstance(v, int) else f"{v:>22,.3f}"
        print(f"{k:<{width}s}{row}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate and compare tokenizers")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--compare", help="a second tokenizer.json to compare against")
    ap.add_argument("--source", choices=["huggingface", "local"], default="huggingface")
    ap.add_argument("--data-path", type=Path)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--val-stories", type=int, default=2000)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--skip-data", action="store_true", help="only run the round-trip checks")
    args = ap.parse_args()

    paths = [args.tokenizer] + ([args.compare] if args.compare else [])
    toks = [Tokenizer.from_file(p) for p in paths]
    failures = 0

    for p, tok in zip(paths, toks):
        print("=" * 70)
        print("Tokenizer:", p)
        eos = tok.token_to_id("<|endoftext|>")
        print(f"Vocabulary size: {tok.get_vocab_size()}   end-of-text id: {eos}"
              f"   {'OK' if eos == 0 else 'WARNING: train.py and generate.py expect id 0'}")
        failures += check_round_trip(tok, ROUND_TRIP, "Round trip on tricky English text")
        failures += check_round_trip(tok, MULTILINGUAL,
                                     "Multilingual round trip (information only: training data is English, "
                                     "so tokens/char is expected to be high)", verbose_chars=True)

    if not args.skip_data:
        print("=" * 70)
        print(f"Loading up to {args.val_stories} '{args.split}' stories...")
        texts = list(iter_texts(args.source, args.split, args.data_path, args.text_column, args.val_stories))
        stats = [corpus_stats(t, texts, args.context_length) for t in toks]
        print_table([Path(p).parent.name or p for p in paths], stats)
        print("\nHow to read this: more bytes per token = better compression, but a bigger vocabulary "
              "makes the embedding table larger.\nCompression alone does not decide the winner: compare "
              "trained models with bits per byte.")
        if len(stats) == 2:
            a, b = stats
            change = 100 * (b["bytes per token"] / a["bytes per token"] - 1)
            print(f"\nSecond tokenizer vs first: {change:+.1f}% bytes per token.")

    if failures:
        raise SystemExit(f"\n{failures} round-trip check(s) failed.")
    print("\nAll round-trip checks passed.")


if __name__ == "__main__":
    main()
