"""
Generate text from a trained checkpoint, stopping at the end-of-story token.
Run:  python generate.py --prompt "Once upon a time" --temperature 0.8
      python generate.py --ckpt ckpt_6layer_newtok.pt --tokenizer tokenizer_new_8k/tokenizer.json
"""
import argparse
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="ckpt.pt")
parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
parser.add_argument("--prompt", type=str, default="Once upon a time")
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--num_samples", type=int, default=3)
parser.add_argument("--seed", type=int, default=None, help="set it to compare models on equal terms")
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_file(args.tokenizer)
eos_id = tok.token_to_id("<|endoftext|>")

ckpt = torch.load(args.ckpt, map_location=device)
if ckpt["config"]["vocab_size"] != tok.get_vocab_size():
    raise SystemExit(f"This checkpoint was trained with a vocabulary of {ckpt['config']['vocab_size']}, "
                     f"but {args.tokenizer} has {tok.get_vocab_size()}. Use the tokenizer it was trained with.")
model = GPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

ids = tok.encode(args.prompt).ids
for n in range(args.num_samples):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
    gen = out[0].tolist()[len(ids):]
    if eos_id in gen:
        gen = gen[:gen.index(eos_id)]   # stop at end-of-story
    print(f"--- sample {n + 1} ---")
    print(tok.decode(ids + gen))
    print()
