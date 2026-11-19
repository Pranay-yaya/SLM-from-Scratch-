# Small Language Model (SLM) from Scratch

A GPT-style small language model written from scratch in PyTorch and trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, as a Deep Learning course project
at Lovely Professional University.

The project covers the whole pipeline: a byte-level BPE tokenizer, a decoder-only transformer, a training loop,
evaluation, text generation, model-size ablations, and a tokenizer data-quality study.
Everything trains on a free Google Colab T4 GPU in 20 to 40 minutes per model.

> **Important:** these are *base* language models. They continue text in the style of children's stories.
> They are not chat models and do not answer questions.

## Results at a glance

| Model | Layers | Embedding | Parameters | Best val loss | Val perplexity | Training time (T4) |
|---|---|---|---|---|---|---|
| Small | 3 | 384 | 8.56M | [TO FILL] | [TO FILL] | [TO FILL] |
| Main | 6 | 384 | 13.87M | 1.885 | 6.59 | about 20.6 min |
| Large | 8 | 512 | 29.51M | 1.782 | 5.94 | about 37.3 min |

- All runs use the same data (about 44M training tokens), schedule and settings, 5,000 steps, one seed.
- The 29.51M model has about 2.1x the parameters of the main model and gave about 10% lower perplexity for about 1.8x the training time.
- The large model changes depth and width together, so it shows the effect of overall model size, not depth alone.

A sample from the main model (temperature 0.8, top-k 40):

> Once upon a time, there was a very pretty bunny named Benny. Benny loved to eat vegetables, especially carrots and bananas. One day, Benny met a wise owl who asked him if he was a good bunny.

The models write fluent, story-like text, but repeat themselves and sometimes lose track of characters,
which is expected at this size.

<!-- After adding your plot to results/, uncomment the next line:
![Validation loss of all runs](results/ablation.png)
-->

## Repository structure

```
.
├── model.py               # GPT-style transformer (attention, MLP, blocks, sampling)
├── train.py               # training loop, LR schedule, evaluation, checkpointing
├── generate.py            # generate text from a checkpoint
├── prepare_data.py        # baseline pipeline: trains the original tokenizer and writes train.bin / val.bin
├── tok_data.py            # data source helper (Hugging Face TinyStories or local files)
├── train_tokenizer.py     # improved tokenizer (repairs corrupted quote characters)
├── evaluate_tokenizer.py  # tokenizer checks and side-by-side comparison
├── encode_data.py         # encode stories to train.bin / val.bin with any tokenizer
├── tokenizer.json         # baseline tokenizer used by the reported checkpoints (add this file)
├── results/               # loss logs (loss_*.csv) and plots (add your files)
├── requirements.txt
└── .gitignore             # checkpoints and .bin files are not committed
```

## Setup

Python 3.9 or newer. On Google Colab, PyTorch is already installed, so you only need:

```bash
pip install datasets tokenizers
```

Locally: `pip install -r requirements.txt`. A GPU is strongly recommended. Training was run on a free Colab Tesla T4 (PyTorch 2.11.0).

## Quick start

### Option A: baseline pipeline (reproduces the main results)

```bash
python prepare_data.py --n_train 200000                       # tokenizer.json, train.bin, val.bin
python train.py --n_layer 6 --out ckpt_6layer.pt              # about 20 minutes on a T4
python generate.py --ckpt ckpt_6layer.pt --prompt "Once upon a time" --seed 42
```

Model sizes compared in the ablation:

```bash
python train.py --n_layer 3 --out ckpt_3layer.pt
python train.py --n_layer 6 --out ckpt_6layer.pt
python train.py --n_layer 8 --n_embd 512 --n_head 8 --out ckpt_30M.pt
```

`prepare_data.py` rewrites `tokenizer.json`, `train.bin` and `val.bin`. Do not run it again once you have trained models,
because a checkpoint only works with the tokenizer it was trained with.

### Option B: improved tokenizer

```bash
python train_tokenizer.py --max-stories 100000 --out-dir tokenizer_new_8k
python evaluate_tokenizer.py --tokenizer tokenizer_new_8k/tokenizer.json --compare tokenizer.json
python encode_data.py --tokenizer tokenizer_new_8k/tokenizer.json --out-dir data_new_8k
python train.py --data_dir data_new_8k --out ckpt_6layer_newtok.pt --log loss_6layer_newtok.csv
python generate.py --ckpt ckpt_6layer_newtok.pt --tokenizer tokenizer_new_8k/tokenizer.json --seed 42
```

`encode_data.py` never overwrites an existing tokenizer. `train.py` reads the vocabulary size from `data_dir/meta.json`.

## Model

Decoder-only transformer, implemented in `model.py`.

| Setting | Main model |
|---|---|
| Layers / heads / embedding | 6 / 6 / 384 (head size 64) |
| MLP hidden size | 1,536 (4x) |
| Context length | 256 tokens |
| Vocabulary | 8,192 (byte-level BPE) |
| Dropout | 0.1 |
| Positions | learned embeddings |
| Normalisation | pre-LayerNorm |
| Activation | GELU |
| Output layer | tied to the token embedding |
| Parameters | 13,870,848 |

Data flow: token + position embeddings, then N blocks of `x = x + Attention(LN(x))` and `x = x + MLP(LN(x))`,
then a final LayerNorm and a linear layer over the vocabulary. Attention is
`softmax(QKᵀ / √d_k) V` with a causal mask.

Parameters grow linearly with depth, but the embedding table is a fixed cost, so the 3-layer model (8,557,824) is
not half of the 6-layer model (13,870,848). The 8-layer, 512-dimension model has 29,508,608.

## Training setup

| Setting | Value |
|---|---|
| Data | first 200,000 TinyStories training stories (44,290,410 tokens), first 5,000 validation stories (983,707 tokens) |
| Optimiser | AdamW, betas 0.9 / 0.95, weight decay 0.1 (none on LayerNorm) |
| Learning rate | 6e-4, 200 warmup steps, cosine decay to 6e-5 |
| Batch | 64 sequences x 256 tokens (16,384 tokens per step) |
| Steps | 5,000 (about 82M tokens, about 1.85 passes over the data) |
| Gradient clipping | 1.0 |
| Precision | fp16 mixed precision with loss scaling |
| Evaluation | every 250 steps, 50 random batches per split; the best checkpoint is saved |

Training loss starts at about 9.05, which matches ln(8192) = 9.01 (random guessing), and falls to about 1.80
(train) and 1.89 (validation) for the main model.

## Tokenizer study: corrupted quote characters

A character census of the training data found **33,014 corrupted quote sequences in 100,000 stories**
(about one for every three stories). An apostrophe such as in *don't* is stored as three odd characters
(`â€™`) because UTF-8 punctuation was wrongly decoded as Windows-1252.

- The baseline model **learned the corruption**: sampled at temperature 1.0 (no top-k cut), it wrote 17 corrupted marks
  in 100 stories. At normal settings (temperature 0.8, top-k 40) it wrote 0 in 30 stories, so the problem is easy to miss.
- The improved tokenizer (`train_tokenizer.py`) adds a light normaliser that repairs these sequences and turns curly quotes,
  dashes and the ellipsis into plain characters. It deliberately does not use NFKC, which is lossy.
- Compression is the same as the baseline, so the benefit is cleaner data, not shorter sequences:

| Measure (2,000 validation stories) | Improved tokenizer | Baseline tokenizer |
|---|---|---|
| Bytes per token | 4.194 | 4.185 |
| Tokens per story (mean) | 190.0 | 190.4 |
| Stories within 256 tokens | 92.95% | 92.95% |
| Corrupted sequences in the text | 495 | 495 |
| Corrupted sequences remaining after tokenization | 0 | 495 |

[TO FILL: result of the 6-layer model retrained with the improved tokenizer, if that run was completed.]

**Comparing across tokenizers.** Perplexity is computed per token, so it cannot be compared between models with
different tokenizers. Use bits per byte on the same text (preferably stories that are pure ASCII, so both models see
exactly the same bytes).

## Model weights and data

Checkpoints (`*.pt`) and token files (`*.bin`) are not stored in this repository, because GitHub rejects files over 100 MB
(the 29.51M checkpoint is about 120 MB). A checkpoint works only with the tokenizer it was trained with.

- Weights: [TO FILL: link to a GitHub Release, Hugging Face Hub repo or Google Drive folder]
- Data: downloaded automatically from Hugging Face by `prepare_data.py` or `train_tokenizer.py`.

## Limitations

- One dataset, one architecture family and one run per configuration, with no repeated seeds.
- No evaluation beyond validation loss, perplexity, bits per byte and inspection of generated samples.
- The models are base models: they cannot follow instructions or answer questions.
- Only the most common corrupted-quote patterns are repaired. Rarer Latin-1 corruptions remain.

## Possible next steps

- Instruction-tune a model on question and answer data to build a small chat assistant.
- Train longer or on more data (the 8-layer model was still improving at the last step).
- Try a 512-token context (about 7% of stories exceed 256 tokens) and a smaller vocabulary (about 28% of vocabulary ids were unused in a 2,000-story sample).
- Replace learned positions with RoPE and LayerNorm with RMSNorm, and add a KV cache.

## Team

| Member | Contribution |
|---|---|
| [TO FILL: name] | [TO FILL] |
| [TO FILL: name] | [TO FILL] |
| [TO FILL: name] | [TO FILL] |

## AI assistance

An AI assistant (Claude, by Anthropic) was used for planning, drafting and debugging code, explaining concepts, and drafting
documentation. The team ran all experiments. [TO FILL: edit this statement so it matches exactly what the team did,
and follow your course's AI-use disclosure rules.]

## Acknowledgements and references

- Eldan, R., and Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Sennrich, R., Haddow, B., and Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
- Loshchilov, I., and Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
- Karpathy, A. [nanoGPT](https://github.com/karpathy/nanoGPT), a public reference implementation of a small GPT.

Built with [PyTorch](https://pytorch.org/) and the Hugging Face `datasets` and `tokenizers` libraries.

## License

[TO FILL: choose a license, for example MIT, and add a LICENSE file. The TinyStories dataset has its own license; check its page on Hugging Face.]
