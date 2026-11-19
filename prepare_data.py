"""
Step 1: download TinyStories, train a BPE tokenizer, and save the data as
token-id arrays (train.bin / val.bin) that the training loop reads fast.

Run:  python prepare_data.py --n_train 200000
"""
import argparse
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=200_000, help="number of stories to use")
parser.add_argument("--n_val", type=int, default=5_000)
parser.add_argument("--vocab_size", type=int, default=8192)
args = parser.parse_args()

EOS = "<|endoftext|>"

print("Loading TinyStories...")
train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{args.n_train}]")
val_ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{args.n_val}]")


# ---- 1. Train a byte-level BPE tokenizer on the training text -------------
def text_iterator(ds, batch=1000):
    for i in range(0, len(ds), batch):
        yield ds[i : i + batch]["text"]


tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=[EOS],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
print("Training BPE tokenizer...")
tok.train_from_iterator(text_iterator(train_ds), trainer)
tok.save("tokenizer.json")
eos_id = tok.token_to_id(EOS)
print(f"Vocab size: {tok.get_vocab_size()}  |  EOS id: {eos_id}")


# ---- 2. Encode the datasets into one long stream of token ids -------------
def encode_split(ds, out_path, batch=5000):
    chunks = []
    for i in range(0, len(ds), batch):
        texts = ds[i : i + batch]["text"]
        for enc in tok.encode_batch(texts):
            chunks.append(np.array(enc.ids + [eos_id], dtype=np.uint16))  # story + EOS
    arr = np.concatenate(chunks)
    arr.tofile(out_path)
    print(f"{out_path}: {len(arr):,} tokens")


encode_split(train_ds, "train.bin")
encode_split(val_ds, "val.bin")
print("Done. Next: python train.py")
