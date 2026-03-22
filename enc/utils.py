from __future__ import annotations

import re
from collections.abc import Sequence

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_document(document: str) -> list[str]:
    normalized = " ".join(str(document).split())
    if not normalized:
        return []
    chunks = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(normalized) if chunk.strip()]
    return chunks or [normalized]


def normalize_documents(documents: Sequence[str] | Sequence[Sequence[str]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for document in documents:
        if isinstance(document, str):
            normalized.append(split_document(document))
            continue
        sentences = [" ".join(str(sentence).split()) for sentence in document if str(sentence).strip()]
        normalized.append(sentences)
    return normalized


def collate_documents(
    documents: Sequence[str] | Sequence[Sequence[str]],
    tokenizer,
    *,
    max_sentences: int = 32,
    max_tokens: int = 64,
) -> dict[str, torch.Tensor]:
    """Convert raw documents into padded tensors for the hierarchical encoder."""
    if torch is None:
        raise RuntimeError("torch is required for collate_documents")
    normalized_docs = normalize_documents(documents)
    if not normalized_docs:
        raise ValueError("documents must not be empty")
    all_input_ids: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_sentence_masks: list[torch.Tensor] = []

    for document in normalized_docs:
        doc = list(document[:max_sentences])
        num_sentences = len(doc)

        if num_sentences > 0:
            encoded = tokenizer(
                doc,
                padding="max_length",
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            input_ids = torch.zeros((0, max_tokens), dtype=torch.long)
            attention_mask = torch.zeros((0, max_tokens), dtype=torch.long)

        if num_sentences < max_sentences:
            pad_sentences = max_sentences - num_sentences
            input_ids = torch.cat(
                [input_ids, torch.zeros((pad_sentences, max_tokens), dtype=torch.long)],
                dim=0,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros((pad_sentences, max_tokens), dtype=torch.long)],
                dim=0,
            )

        sentence_mask = torch.zeros(max_sentences, dtype=torch.long)
        sentence_mask[:num_sentences] = 1
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_sentence_masks.append(sentence_mask)

    return {
        "input_ids": torch.stack(all_input_ids, dim=0),
        "attention_mask": torch.stack(all_attention_masks, dim=0),
        "sentence_mask": torch.stack(all_sentence_masks, dim=0),
    }
