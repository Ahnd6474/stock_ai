import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class SentenceLevelTransformer(nn.Module):
    def __init__(
        self,
        roberta_name="roberta-base",
        num_classes=2,
        sent_transformer_layers=2,
        sent_transformer_heads=8,
        sent_dropout=0.1,
        max_sentences=128,
        freeze_roberta=False,
        pooling="cls",   # "cls" or "mean"
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(roberta_name)
        self.roberta = AutoModel.from_pretrained(roberta_name)
        self.hidden_size = self.roberta.config.hidden_size
        self.pooling = pooling
        self.max_sentences = max_sentences

        if freeze_roberta:
            for p in self.roberta.parameters():
                p.requires_grad = False

        # 문장 위치 임베딩
        self.sentence_pos_embedding = nn.Embedding(max_sentences, self.hidden_size)

        # 문장 수준 Transformer
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
            num_layers=sent_transformer_layers
        )

        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(sent_dropout)

        # 문서-level head
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def encode_sentences(self, input_ids, attention_mask):
        """
        input_ids:      [B, S, T]
        attention_mask: [B, S, T]
        B=batch, S=num_sentences, T=token_length
        """
        B, S, T = input_ids.shape

        # 문장별로 RoBERTa 적용 위해 flatten
        flat_input_ids = input_ids.view(B * S, T)
        flat_attention_mask = attention_mask.view(B * S, T)

        outputs = self.roberta(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask
        )
        last_hidden = outputs.last_hidden_state   # [B*S, T, H]

        if self.pooling == "cls":
            # RoBERTa는 실질적으로 첫 토큰(<s>) 벡터 사용
            sent_emb = last_hidden[:, 0, :]       # [B*S, H]
        elif self.pooling == "mean":
            mask = flat_attention_mask.unsqueeze(-1)  # [B*S, T, 1]
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            sent_emb = summed / denom
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

        sent_emb = sent_emb.view(B, S, self.hidden_size)  # [B, S, H]
        return sent_emb

    def forward(self, input_ids, attention_mask, sentence_mask):
        """
        input_ids:      [B, S, T]
        attention_mask: [B, S, T]
        sentence_mask:  [B, S]   (1=real sentence, 0=padding sentence)
        """
        B, S, T = input_ids.shape

        sent_emb = self.encode_sentences(input_ids, attention_mask)  # [B, S, H]

        # 문장 위치 정보 추가
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        sent_emb = sent_emb + self.sentence_pos_embedding(positions)

        sent_emb = self.norm(sent_emb)
        sent_emb = self.dropout(sent_emb)

        # TransformerEncoder의 src_key_padding_mask는 True가 padding
        padding_mask = (sentence_mask == 0)  # [B, S]

        sent_out = self.sentence_transformer(
            sent_emb,
            src_key_padding_mask=padding_mask
        )  # [B, S, H]

        # 문서 표현: 유효 문장 평균
        mask = sentence_mask.unsqueeze(-1)  # [B, S, 1]
        doc_emb = (sent_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits = self.classifier(self.dropout(doc_emb))
        return logits

