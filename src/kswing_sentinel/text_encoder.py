from __future__ import annotations

import hashlib


class KoreanTextEncoder:
    def __init__(self, encoder_version: str = "ko_bert_placeholder_v1", tokenizer_version: str = "spm_placeholder_v1") -> None:
        self.encoder_version = encoder_version
        self.tokenizer_version = tokenizer_version

    def encode(self, text: str, dim: int) -> list[float]:
        seed = text.encode("utf-8")
        out: list[float] = []
        for i in range(dim):
            h = hashlib.sha256(seed + i.to_bytes(2, "little")).digest()
            out.append((int.from_bytes(h[:4], "little") / 2**32) * 2 - 1)
        return out

    def batch_encode(self, texts: list[str], dim: int) -> list[list[float]]:
        return [self.encode(t, dim) for t in texts]

    def metadata(self) -> dict:
        return {
            "encoder_version": self.encoder_version,
            "tokenizer_version": self.tokenizer_version,
        }
