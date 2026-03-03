# bpe_train.py
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple
import re
import regex  # third-party "regex" package (needed for \p{L}, \p{N})
from pathlib import Path
# GPT-2 style pre-tokenizer pattern (from handout)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
    # A) Load corpus
    # - read bytes from input_path
    # - decode to str for regex pretokenization (common approach)
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus = f.read()   # corpus 是 str

    # B) Count pre-tokens with special-token boundaries respected
    pretoken_counts = count_pretokens(corpus, special_tokens)

    # C) Build initial byte-symbol sequences (with weights)
    seqs = build_initial_sequences(pretoken_counts)

    # D) Initialize vocab (256 bytes) + add special tokens
    vocab = init_byte_vocab()
    add_special_tokens_to_vocab(vocab, special_tokens)
    # E) Iteratively:
    #   - count pair frequencies
    #   - pick best pair (freq, tie-break)
    #   - create merged symbol bytes = a + b
    #   - add to vocab if space
    #   - record merge
    #   - apply merge to sequences
    # until vocab reaches vocab_size or no more pairs
    merges:List[Pair] = []
    next_id = max(vocab.keys(), default=-1) + 1
    while len(vocab) < vocab_size:
        pair_counts = count_pairs(seqs)
        pair = choose_best_pair(pair_counts)
        if pair is None:
            break
        a,b = pair
        merged_symbol = a + b
        merges.append(pair) # document merges
        vocab[next_id] = merged_symbol
        next_id += 1

        seqs = apply_merge(seqs, pair, merged_symbol) # update the sequence using merged_symbols

    # F) Return vocab, merges
    return vocab, merges