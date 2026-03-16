# LLM Data Pipeline Assignment

A complete Language Model (LM) data pipeline built with HuggingFace Datasets, Transformers, and PyTorch. This assignment demonstrates both in-memory and streaming approaches to preparing text data for LLM pre-training, along with multi-process sharding and performance benchmarking.

---

## Folder Structure

```
LLM_Data_Pipeline/
│
├── LLM_Data_Pipeline_Assignment.ipynb   # Main assignment notebook
└── README.md                            # This file
```

---

## Assignment Overview

| Part | Topic | Description |
|------|-------|-------------|
| 1 | Non-Streaming Pipeline | Load full dataset into memory, tokenize, group into 128-token blocks, create DataLoader |
| 2 | Token Frequency Analysis ✨ | Count token frequencies, top-20 tokens, vocabulary coverage at 80/90/99% |
| 3 | Streaming + Rolling Buffer | Memory-efficient streaming with on-the-fly tokenization and rolling buffer chunking |
| 4 | Multi-Process Sharding | Round-robin data sharding across 4 simulated workers |
| 5 | Throughput Benchmarking ✨ | Compare tokens/sec between streaming vs non-streaming pipelines |

> ✨ = New features added beyond the reference labs

---

## Dataset & Model

| Property | Value |
|----------|-------|
| Dataset | WikiText-2-raw-v1 (HuggingFace) |
| Tokenizer | GPT-2 (`gpt2`) |
| Vocab Size | 50,257 tokens |
| Block Size | 128 tokens |
| Batch Size | 8 (Parts 1, 3, 5), 4 (Part 4) |

---

## Requirements

- Python >= 3.8
- PyTorch
- HuggingFace `datasets`
- HuggingFace `transformers`

Install all dependencies with:

```bash
pip install torch datasets transformers jupyter ipykernel
```

---

## How to Run

### Step 1 — Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
```

### Step 2 — Install dependencies
```bash
pip install torch datasets transformers jupyter ipykernel
```

### Step 3 — Open in VS Code
1. Open VS Code
2. Click **File** → **Open Folder** → select `LLM_Data_Pipeline`
3. Open `LLM_Data_Pipeline_Assignment.ipynb`
4. Select kernel → **venv (Python 3.x)**
5. Click **Run All** or run cells one by one with `Shift + Enter`

---

## What Each Part Does

### Part 1 — Non-Streaming Pipeline
Loads the entire WikiText-2 dataset into RAM, tokenizes it using the GPT-2 tokenizer, groups all tokens into fixed-length blocks of 128, and wraps them in a PyTorch DataLoader for training.

### Part 2 — Token Frequency & Vocabulary Analysis ✨
Counts every token across the full dataset and identifies the top-20 most frequent tokens. Also measures vocabulary coverage — how many unique tokens are needed to account for 80%, 90%, and 99% of all token occurrences. This demonstrates **Zipf's Law** in natural language.

**Key Finding:**
```
80% of all tokens → only 3,404 unique tokens (6.8% of vocab)
90% of all tokens → only 8,118 unique tokens (16.2% of vocab)
99% of all tokens → only 25,567 unique tokens (50.9% of vocab)
```

### Part 3 — Streaming + Rolling Buffer
Instead of loading the full dataset into memory, data is streamed one example at a time. A rolling buffer accumulates tokens across document boundaries and yields complete fixed-length blocks — exactly how web-scale LLM pre-training pipelines work.

**How the Rolling Buffer works:**
```
Document 1: [1, 2, 3, 4, 5]
Document 2: [6, 7, 8, 9, 10]
Buffer:     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Block 1:    [1, 2, 3, 4, 5]
Block 2:    [6, 7, 8, 9, 10]
```

### Part 4 — Multi-Process Sharding
Simulates distributing data across 4 parallel workers using round-robin sharding. Each worker receives a unique, non-overlapping subset of the dataset — critical for avoiding duplicate gradient updates in distributed training.

**Sharding pattern:**
```
Worker 0 → examples 0, 4, 8,  12 ...
Worker 1 → examples 1, 5, 9,  13 ...
Worker 2 → examples 2, 6, 10, 14 ...
Worker 3 → examples 3, 7, 11, 15 ...
```

### Part 5 — Throughput Benchmarking ✨
Runs 50 batches through both the non-streaming and streaming pipelines and measures tokens/second, mean batch latency, and total processing time.

**Typical Results:**
```
Non-Streaming :  ~185,000 tokens/sec  ✅ faster at batch time
Streaming     :   ~62,000 tokens/sec  ✅ lower memory usage
```

**Trade-off Summary:**

| | Non-Streaming | Streaming |
|--|--|--|
| Speed | ✅ Faster | ❌ Slower |
| Memory Usage | ❌ High | ✅ Low |
| Random Access | ✅ Yes | ❌ No |
| Shuffle | ✅ Yes | ❌ No |
| Best For | Small/Medium datasets | Web-scale corpora |

---

## Key Concepts

### Tokenization
Converting raw text into integer token IDs that a neural network can process. GPT-2 uses Byte Pair Encoding (BPE) with a vocabulary of 50,257 tokens.

### Fixed-Length Blocking
Neural networks require fixed-size inputs. All text is concatenated and sliced into equal blocks of 128 tokens so every training sample is the same shape.

### Zipf's Law
In any natural language corpus, a small number of tokens appear extremely frequently while most tokens are rare. Only ~7% of GPT-2's vocabulary covers 80% of WikiText-2.

### Streaming
Processing data one example at a time without loading the full dataset into RAM. Essential for training on terabyte-scale corpora like Common Crawl.

### Sharding
Splitting data across multiple workers so each processes a unique subset. Used in distributed training across multiple GPUs or machines.

---

## The Big Picture

```
Raw Text (WikiText-2)
        ↓
  GPT-2 Tokenizer
  (words → token IDs)
        ↓
  Rolling Buffer
  (fixed 128-token blocks)
        ↓
  Sharding
  (split across 4 workers)
        ↓
  DataLoader
  (batches of 8)
        ↓
  Ready for LLM Training 🚀
```

This mirrors the real-world data pipelines used to train large language models like GPT-4, LLaMA, and Gemini — just at a much larger scale.

---

## References

- [HuggingFace Datasets Docs](https://huggingface.co/docs/datasets)
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [PyTorch DataLoader Docs](https://pytorch.org/docs/stable/data.html)
- [GPT-2 Model Card](https://huggingface.co/gpt2)
- [WikiText Dataset](https://huggingface.co/datasets/wikitext)