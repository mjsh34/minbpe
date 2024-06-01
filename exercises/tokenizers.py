from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Tuple, Optional
import sys

from tqdm import tqdm
import regex


class TokenizerBase(ABC):
    def __init__(self):
        self.lg = logging.getLogger()

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    @abstractmethod
    def train(self, text: str, vocab_size: int, verbose=False):
        pass
    
    @classmethod
    def _tokenize(cls, text: str) -> List[int]:
        bb = text.encode('utf-8')
        return list(bb)

    @classmethod
    def _count_pairs(cls, ids: List[int], pair_counts: Optional[Dict[Tuple[int, int], int]]=None) \
            -> Dict[Tuple[int, int], int]:
        if pair_counts is None:
            pair_counts = {}
        for i in range(len(ids) - 1):
            p = (ids[i], ids[i+1])
            pair_counts[p] = pair_counts.get(p, 0) + 1
        return pair_counts

    @classmethod
    def _merge(cls, ids: List[int], pair: Tuple[int, int], pair_id: int):
        assert pair_id not in ids, f"Pair id '{pair_id}' for pair {pair} must not be in ids"
        i = 0
        ids_merged = []
        while i <= len(ids) - 1:
            if i <= len(ids) - 2 and (ids[i] == pair[0] and ids[i+1] == pair[1]):
                ids_merged.append(pair_id)
                i += 2
            else:
                ids_merged.append(ids[i])
                i += 1
        return ids_merged


class BPETokenizerBase(TokenizerBase):
    def __init__(self):
        super().__init__()
        self._merges = []

    def decode(self, ids: List[int], errors='strict') -> str:
        vocab = self._build_vocab()
        self.lg.info("Start decode token ids (#=%d), vocab size=%d.", len(ids), len(vocab))
        return b''.join(map(vocab.__getitem__, ids)).decode('utf-8', errors=errors)

    def _build_vocab(self) -> Dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        self.lg.info("Building vocab from %d merges.", len(self._merges))
        for imerge, (pair, token_id) in enumerate(self._merges):
            assert token_id > 255
            assert imerge == 0 or self._merges[imerge-1][1] < token_id, \
                    "Token ids assigned to merges must be strictly monotonically increasing."
            vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]
        return vocab


class BasicTokenizer(BPETokenizerBase):
    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("Vocab size must be >= 256")
        if len(self._merges) > 0:
            self.lg.info("Tokenizer already trained with %d merges; will clear previous train data.",
                         len(self._merges))
            self._merges.clear()
        n_merges = vocab_size - 256
        token_ids = self._tokenize(text)
        for merge_i in (pbar := tqdm(range(n_merges))):
            if len(token_ids) <= 1:
                self.lg.warning(f"At merge %d/%d we have %d tokens. Stopping early.",
                                merge_i+1, n_merges, len(token_ids))
            pair_counts = self._count_pairs(token_ids)
            most_frequent_pair = max(pair_counts, key=pair_counts.__getitem__)
            new_token_id = 256 + merge_i
            token_ids = self._merge(ids=token_ids, pair=most_frequent_pair, pair_id=new_token_id)
            self._merges.append((most_frequent_pair, new_token_id))

            self.lg.debug("merge %d/%d, #tokens=%d, pair=%s -> id: %d",
                          merge_i+1, n_merges, len(token_ids), most_frequent_pair, new_token_id)
            pbar.set_description(f"New token: {new_token_id}")

    def encode(self, text: str) -> List[int]:
        self.lg.info("Will attempt to encode text: '%s' with model currently trained with %d merges.",
                     text, len(self._merges))
        token_ids = self._tokenize(text)
        for imerge, (pair, token_id) in enumerate(self._merges):
            assert imerge == 0 or self._merges[imerge-1][1] < token_id, \
                    "Token ids assigned to merges must be strictly monotonically increasing."
            token_ids = self._merge(ids=token_ids, pair=pair, pair_id=token_id)
            if len(token_ids) < 2:
                break
        return token_ids


class RegexTokenizer(BPETokenizerBase):
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    _SPLIT_PAT = regex.compile(GPT4_SPLIT_PATTERN)

    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose=False):
        if vocab_size < 256:
            raise ValueError("Vocab size must be >= 256")
        if len(self._merges) > 0:
            self.lg.info("Tokenizer already trained with %d merges; will clear previous train data.",
                         len(self._merges))
            self._merges.clear()
        n_merges = vocab_size - 256
        text_segments = regex.findall(self._SPLIT_PAT, text) 
        self.lg.info("Identified %d text segments.", len(text_segments))
        segs_token_ids = [self._tokenize(seg) for seg in text_segments]

        for merge_i in (merge_pbar := tqdm(range(n_merges))):
            pair_counts = {}
            for token_ids in segs_token_ids:
                pair_counts = self._count_pairs(token_ids, pair_counts=pair_counts)
            most_frequent_pair = max(pair_counts, key=pair_counts.__getitem__)
            new_token_id = 256 + merge_i
            for iseg, token_ids in enumerate(segs_token_ids):
                token_ids_merged = self._merge(ids=token_ids, pair=most_frequent_pair, pair_id=new_token_id)
                segs_token_ids[iseg] = token_ids_merged
            self._merges.append((most_frequent_pair, new_token_id))

            self.lg.debug("merge %d/%d, #tokens=%d, pair=%s -> id: %d",
                          merge_i+1, n_merges, len(token_ids), most_frequent_pair, new_token_id)

    def encode(self, text: str) -> List[int]:
        self.lg.info("Will attempt to encode text: '%s' with model currently trained with %d merges.",
                     text, len(self._merges))
        text_segments = regex.findall(self._SPLIT_PAT, text) 
        self.lg.info("Identified %d text segments.", len(text_segments))
        segs_token_ids = [self._tokenize(seg) for seg in text_segments]
        for merge_i, (pair, token_id) in enumerate(self._merges):
            assert merge_i == 0 or self._merges[merge_i-1][1] < token_id, \
                    "Token ids assigned to merges must be strictly monotonically increasing."
            for iseg, token_ids in enumerate(segs_token_ids):
                if len(token_ids) < 2:
                    continue
                token_ids_merged = self._merge(ids=token_ids, pair=pair, pair_id=token_id)
                segs_token_ids[iseg] = token_ids_merged

        token_ids_all = []
        for token_ids in segs_token_ids:
            token_ids_all.extend(token_ids)
        return token_ids_all


if __name__ == '__main__':
    logging.getLogger().handlers.clear()
    logging.basicConfig()
    logging.root.setLevel(level=logging.DEBUG)

    # Test tokenizers
    test_strings = [
        "", # empty string
        "?", # single character
        "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    ]
    tokenizer_factories = [BasicTokenizer, RegexTokenizer]
    for tokenizer_factory in tokenizer_factories:
        print(f"{30*'='} Tokenizer: {tokenizer_factory.__name__} {30*'='}")
        print("Tokenizer:", tokenizer_factory)
        for text in test_strings:
            tokenizer = tokenizer_factory()
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            assert text == decoded, f"{text=}\n{decoded=}\n{ids=}"

            text = "aaabdaaabac"
            tokenizer.train(text, 256 + 3)
            ids = tokenizer.encode(text)
            assert ids == [258, 100, 258, 97, 99], "wikipedia - ids"
            assert tokenizer.decode(tokenizer.encode(text)) == text, "wikipedia - recon"
        print("\n\n")
