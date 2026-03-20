from __future__ import annotations

import hashlib


def _hash_to_vec(text: str, dim: int) -> list[float]:
    out = []
    seed = text.encode("utf-8")
    for i in range(dim):
        h = hashlib.sha256(seed + i.to_bytes(2, "little")).digest()
        v = int.from_bytes(h[:4], "little") / 2**32
        out.append(2.0 * v - 1.0)
    return out


class VectorizationPipeline:
    def build(self, summary: str, social: str = "", macro: str = "") -> dict:
        return {
            "z_event": _hash_to_vec(summary, 64),
            "z_social": _hash_to_vec(social or summary, 32),
            "z_macro": _hash_to_vec(macro or "KOSPI", 16),
        }
