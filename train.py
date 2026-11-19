"""
Step 2: train the model.   Run:  python train.py
Saves the best checkpoint to ckpt.pt and the loss history to loss_log.csv.
"""
import os
import json
import math
import time
import argparse
import numpy as np
import torch

from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--n_layer", type=int, default=6)
parser.add_argument("--n_head", type=int, default=6)
parser.add_argument("--n_embd", type=int, default=384)
parser.add_argument("--block_size", type=int, default=256)
parser.add_argument("--vocab_size", type=int, default=None,
                    help="default: read from data_dir/meta.json, otherwise 8192")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_iters", type=int, default=5000)
parser.add_argument("--lr", type=float, default=6e-4)
parser.add_argument("--min_lr", type=float, default=6e-5)
parser.add_argument("--warmup", type=int, default=200)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--eval_interval", type=int, default=250)
parser.add_argument("--eval_iters", type=int, default=50)
parser.add_argument("--out", type=str, default="ckpt.pt")
parser.add_argument("--data_dir", type=str, default=".", help="folder with train.bin, val.bin (and meta.json)")
parser.add_argument("--log", type=str, default="loss_log.csv")
args = parser.parse_args()

# vocabulary size: explicit flag > meta.json written by encode_data.py > 8192 (the old default)
meta_path = os.path.join(args.data_dir, "meta.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
if args.vocab_size is None:
    args.vocab_size = meta["vocab_size"] if meta else 8192
elif meta and meta["vocab_size"] != args.vocab_size:
    raise SystemExit(f"--vocab_size {args.vocab_size} does not match the data in {meta_path} ({meta['vocab_size']}).")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
torch.manual_seed(1337)

# ---- data -----------------------------------------------------------------
train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + args.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + args.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)   # y is x shifted by one: predict the next token


# ---- model ----------------------------------------------------------------
cfg = GPTConfig(
    vocab_size=args.vocab_size, block_size=args.block_size,
    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
)
model = GPT(cfg).to(device)
print(f"Parameters: {model.num_params() / 1e6:.2f}M  |  device: {device}  |  vocab: {args.vocab_size}")

# no weight decay on biases / LayerNorm / embeddings (1-D params)
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)   # fp16 needs loss scaling (T4 has no bf16)


def get_lr(it):
    """Linear warmup, then cosine decay down to min_lr."""
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---- training loop --------------------------------------------------------
best_val = float("inf")
log = open(args.log, "w")
log.write("iter,train_loss,val_loss,val_perplexity,lr\n")
t0 = time.time()

for it in range(args.max_iters + 1):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if it % args.eval_interval == 0:
        losses = estimate_loss()
        ppl = math.exp(losses["val"])
        print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
              f"ppl {ppl:.2f} | lr {lr:.2e} | {time.time() - t0:.0f}s")
        log.write(f"{it},{losses['train']:.4f},{losses['val']:.4f},{ppl:.2f},{lr:.6f}\n")
        log.flush()
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "iter": it, "val_loss": best_val}, args.out)

    if it == args.max_iters:
        break

    x, y = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()

log.close()
print(f"Finished. Best val loss {best_val:.4f} (perplexity {math.exp(best_val):.2f}). Saved to {args.out}")
