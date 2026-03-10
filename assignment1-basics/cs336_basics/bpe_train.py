# bpe_train.py
"""
BPE 训练的两处加速（相对原始逐段实现）：

1) 多进程预分词
   原版: 整份语料一次 count_pretokens(corpus, special_tokens)，大文件时 regex 是瓶颈。
   现版: 用 find_chunk_boundaries 按特殊 token 把文件切成多块，每块在独立进程里
         count_pretokens，最后把各进程返回的 Counter 合并。单块或 Pool 创建失败时回退到串行。

2) 合并阶段的 pair 计数缓存 + 增量更新
   原版: 每轮 merge 都 count_pairs(seqs) 全量扫所有序列找最大 pair，每轮 O(总符号数)。
   现版: 只在开始时算一次 pair_counts；每轮只对“受本次合并影响的 pair”做增减
         (update_pair_counts_incremental)，避免每轮全量重算。
"""
from __future__ import annotations

import os
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO, Dict, List, Tuple
import re
import regex  # third-party "regex" package (needed for \p{L}, \p{N})
from pathlib import Path

# GPT-2 style pre-tokenizer pattern (from handout)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# -----------------------------
# Chunking for parallel pre-tokenization (from pretokenization_example.py)
# -----------------------------
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Boundaries align to the start of split_special_token so we never split mid-document.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // desired_num_chunks)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

# -----------------------------
# 0) Helpers: special-token split + regex pre-tokenization
# -----------------------------
def build_special_split_pattern(special_tokens: List[str]) -> str:
    """
    Return a regex pattern that matches any of the special tokens literally.
    Use re.escape on each token, then join with '|'.
    """
    if not special_tokens:
        return "" 
    escaped = [re.escape(tok) for tok in special_tokens]
    return "|".join(escaped)
    


def split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    Split `text` on special tokens and return the non-empty segments.
    Segments do NOT include the special tokens themselves.
    """
    if not special_tokens:
        return [text] if text else []

    pattern = build_special_split_pattern(special_tokens)
    parts = re.split(pattern,text)
    return [p for p in parts if p]


def pretokenize_segment(segment: str) -> List[str]:
    """
    Apply the GPT-2 regex pre-tokenizer PAT to `segment` and return pre-tokens.
    Use regex.finditer, NOT re.finditer.
    """
    return[m.group(0) for m in regex.finditer(PAT, segment)] 
        


def count_pretokens(text: str, special_tokens: List[str]) -> Counter[str]:
    """
    Split text on special tokens, pretokenize each segment, and count pre-tokens.
    """
    counts: Counter[str] = Counter()
    segments = split_on_special_tokens(text, special_tokens)
    for seg in segments:
        pretoks = pretokenize_segment(seg)
        for pretok in pretoks:
            counts[pretok] += 1

    return counts

# -----------------------------
# 1) Byte-level representation
# -----------------------------
def pretoken_to_byte_symbols(pretoken: str) -> List[bytes]:
    """
    Convert one pre-token (str) into an initial sequence of byte symbols:
    - pretoken.encode("utf-8") -> bytes
    - return [bytes([b0]), bytes([b1]), ...]
    """
    b = pretoken.encode("utf-8")
    return [bytes([x]) for x in b]


def build_initial_sequences(pretoken_counts: Counter[str]) -> List[Tuple[List[bytes], int]]:
    """
    Build training sequences from pretoken counts.

    Return a list of (symbol_sequence, weight) pairs, where:
    - symbol_sequence is List[bytes] (each bytes is length 1 initially)
    - weight is how many times that pretoken occurred
    """
    seqs: List[Tuple[List[bytes], int]] = []
    for pretoken, count in pretoken_counts.items():
        symbols = pretoken_to_byte_symbols(pretoken)
        seqs.append((symbols, count))
    return seqs

# -----------------------------
# 2) Pair counting + selecting best pair
# -----------------------------
Pair = Tuple[bytes, bytes]


def count_pairs(seqs: List[Tuple[List[bytes], int]]) -> Counter[Pair]:
    """
    Count frequencies of adjacent byte-symbol pairs across all sequences,
    weighted by each sequence's weight.
    Return Counter[(a,b)] -> frequency
    """
    counters:Counter[Pair] = Counter()
    for symbols, weight in seqs:
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            counters[pair] += weight
    return counters


def choose_best_pair(pair_counts: Counter[Pair]) -> Pair | None:
    """
    Choose the pair with highest frequency.
    Tie-break: prefer lexicographically greater pair (Python tuple comparison).
    Return None if no pairs exist (i.e., everything length <= 1).
    """
    if not pair_counts:
        return None
    best_pair, _best_freq = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
    return best_pair
        


# -----------------------------
# 3) Applying a merge to sequences
# -----------------------------
def merge_pair_in_sequence(seq: List[bytes], pair: Pair, merged_symbol: bytes) -> List[bytes]:
    """
    Replace all non-overlapping occurrences of `pair` in `seq` with `merged_symbol`.
    Example: [X, A, B, Y] with pair=(A,B) => [X, AB, Y]
    Must scan left-to-right; when you merge at i, skip i+1.
    """
    a, b = pair
    out: List[bytes] = []
    i = 0
    while i < len(seq):
        # 看看从 i 开始能不能匹配到 (a,b)
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(merged_symbol)
            i += 2              # 合并了两个元素，跳过 i+1
        else:
            out.append(seq[i])
            i += 1
    return out
    


def apply_merge(
    seqs: List[Tuple[List[bytes], int]],
    pair: Pair,
    merged_symbol: bytes
) -> List[Tuple[List[bytes], int]]:
    """
    Apply merge to every sequence (keeping weights the same).
    Return updated sequences.
    """
    out:List[Tuple[List[bytes], int]] = []
    for seq, weight in seqs:
        out.append((merge_pair_in_sequence(seq, pair, merged_symbol), weight))
    return out


def update_pair_counts_incremental(
    seqs: List[Tuple[List[bytes], int]],
    pair: Pair,
    merged_symbol: bytes,
    pair_counts: Counter[Pair],
) -> None:
    """
    Incrementally update pair_counts after merging (a, b) -> merged_symbol.
    Only pairs that overlap with the merged pair change; we subtract the old
    pairs and add the new ones instead of recounting all sequences.
    Modifies pair_counts in place.
    """
    a, b = pair
    for symbols, weight in seqs:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                if i > 0:
                    left = symbols[i - 1]
                    pair_counts[(left, a)] -= weight
                    if pair_counts[(left, a)] <= 0:
                        del pair_counts[(left, a)]
                    pair_counts[(left, merged_symbol)] += weight
                pair_counts[(a, b)] -= weight
                if pair_counts[(a, b)] <= 0:
                    del pair_counts[(a, b)]
                if i + 2 < len(symbols):
                    right = symbols[i + 2]
                    pair_counts[(b, right)] -= weight
                    if pair_counts[(b, right)] <= 0:
                        del pair_counts[(b, right)]
                    pair_counts[(merged_symbol, right)] += weight
                i += 2
            else:
                i += 1


# -----------------------------
# 4) Vocab construction
# -----------------------------
def init_byte_vocab() -> Dict[int, bytes]:
    """
    Initialize vocab with 256 single-byte tokens:
    vocab[i] = bytes([i]) for i in 0..255
    """
    out: Dict[int, bytes] = {} # Dict init with curly braces
    for i in range(256):
        out[i] = bytes([i]) # Edit dict using square braces
    return out


def add_special_tokens_to_vocab(vocab: Dict[int, bytes], special_tokens: List[str]) -> None:
    """
    Add special tokens to vocab as UTF-8 bytes.
    They do not affect training; they just occupy vocab slots.
    """
    next_id = max(vocab.keys(), default=-1) + 1
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1



# -----------------------------
# 5) Main entry: train_bpe (required signature)
# -----------------------------
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Pair]]:
    """
    Train a byte-level BPE tokenizer.

    Must return:
      vocab: dict[int, bytes]
      merges: list[tuple[bytes, bytes]] in creation order
    """
    # A) Load corpus and count pre-tokens
    # --- 优化 1: 多进程预分词 ---
    # 原版: 整份语料一次 count_pretokens(corpus, special_tokens)，大文件时 regex 是瓶颈。
    # 现版: 按特殊 token 切块，每块在独立进程里做预分词，最后合并 Counter。
    num_workers = max(1, min(8, os.cpu_count() or 4))
    pretoken_counts: Counter[str]
    if special_tokens:
        split_token = special_tokens[0].encode("utf-8")
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_workers, split_token)
        if len(boundaries) < 2:
            with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                corpus = f.read()
            pretoken_counts = count_pretokens(corpus, special_tokens)
        else:
            chunks = []
            with open(input_path, "rb") as f:
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
            if len(chunks) == 1:
                pretoken_counts = count_pretokens(chunks[0], special_tokens)
            else:
                try:
                    with Pool(min(num_workers, len(chunks))) as pool:
                        chunk_counts = pool.starmap(
                            count_pretokens, [(c, special_tokens) for c in chunks]
                        )
                    pretoken_counts = sum(chunk_counts, Counter())
                except (PermissionError, OSError):
                    pretoken_counts = sum(
                        (count_pretokens(c, special_tokens) for c in chunks), Counter()
                    )
    else:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            corpus = f.read()
        pretoken_counts = count_pretokens(corpus, special_tokens)

    # C) Build initial byte-symbol sequences (with weights)
    seqs = build_initial_sequences(pretoken_counts)

    # D) Initialize vocab (256 bytes) + add special tokens
    vocab = init_byte_vocab()
    add_special_tokens_to_vocab(vocab, special_tokens)

    # E) Iteratively merge; use cached pair counts and incremental updates
    # --- 优化 2: 合并阶段的 pair 计数缓存 + 增量更新 ---
    # 原版: 每轮都 count_pairs(seqs) 全量扫所有序列，O(总符号数) 每轮。
    # 现版: 只在开始时算一次 pair_counts，每轮只更新“受本次合并影响的 pair”。
    merges: List[Pair] = []
    next_id = max(vocab.keys(), default=-1) + 1
    pair_counts = count_pairs(seqs)
    while len(vocab) < vocab_size:
        pair = choose_best_pair(pair_counts)
        if pair is None:
            break
        a, b = pair
        merged_symbol = a + b
        merges.append(pair)
        vocab[next_id] = merged_symbol
        next_id += 1
        update_pair_counts_incremental(seqs, pair, merged_symbol, pair_counts)
        seqs = apply_merge(seqs, pair, merged_symbol)

    return vocab, merges