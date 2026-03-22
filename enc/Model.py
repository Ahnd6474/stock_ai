from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None

from .utils import collate_documents

DEFAULT_MODEL_ID = "klue/roberta-base"


class SentenceLevelTransformer(nn.Module if nn is not None else object):
    """Hierarchical document encoder on top of a sentence-level RoBERTa backbone."""

    def __init__(
        self,
        *,
        roberta_name: str = DEFAULT_MODEL_ID,
        num_classes: int | None = None,
        sent_transformer_layers: int = 2,
        sent_transformer_heads: int = 8,
        sent_dropout: float = 0.1,
        max_sentences: int = 128,
        freeze_roberta: bool = False,
        pooling: str = "cls",
    ) -> None:
        if nn is None or torch is None:
            raise RuntimeError("torch is required for SentenceLevelTransformer")
        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError("transformers is required for SentenceLevelTransformer")
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.tokenizer = AutoTokenizer.from_pretrained(roberta_name)
        self.roberta = AutoModel.from_pretrained(roberta_name)
        self.hidden_size = int(self.roberta.config.hidden_size)
        self.pooling = pooling
        self.max_sentences = max_sentences

        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.sentence_pos_embedding = nn.Embedding(max_sentences, self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=sent_transformer_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=sent_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sentence_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=sent_transformer_layers,
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(sent_dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes) if num_classes is not None else None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_sentences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, num_sentences, num_tokens]
            attention_mask: [batch, num_sentences, num_tokens]
        Returns:
            Sentence embeddings with shape [batch, num_sentences, hidden_size].
        """
        batch_size, num_sentences, num_tokens = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_sentences, num_tokens)
        flat_attention_mask = attention_mask.view(batch_size * num_sentences, num_tokens)

        outputs = self.roberta(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        last_hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            sent_emb = last_hidden[:, 0, :]
        else:
            mask = flat_attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            sent_emb = summed / denom

        return sent_emb.view(batch_size, num_sentences, self.hidden_size)

    def encode_documents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, num_sentences, num_tokens]
            attention_mask: [batch, num_sentences, num_tokens]
            sentence_mask: [batch, num_sentences]
        Returns:
            Document embeddings with shape [batch, hidden_size].
        """
        _, num_sentences, _ = input_ids.shape
        sent_emb = self.encode_sentences(input_ids, attention_mask)
        positions = torch.arange(num_sentences, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(input_ids.size(0), num_sentences)
        sent_emb = sent_emb + self.sentence_pos_embedding(positions)
        sent_emb = self.dropout(self.norm(sent_emb))

        padding_mask = sentence_mask == 0
        sent_out = self.sentence_transformer(sent_emb, src_key_padding_mask=padding_mask)
        mask = sentence_mask.unsqueeze(-1).float()
        return (sent_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_mask: torch.Tensor,
        *,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        doc_emb = self.encode_documents(input_ids, attention_mask, sentence_mask)
        if return_embeddings or self.classifier is None:
            return doc_emb
        return self.classifier(self.dropout(doc_emb))

    def forward_texts(
        self,
        documents: Sequence[str] | Sequence[Sequence[str]],
        *,
        max_sentences: int | None = None,
        max_tokens: int = 64,
        return_embeddings: bool = True,
    ) -> torch.Tensor:
        batch = collate_documents(
            documents,
            self.tokenizer,
            max_sentences=max_sentences or self.max_sentences,
            max_tokens=max_tokens,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        return self.forward(**batch, return_embeddings=return_embeddings)
