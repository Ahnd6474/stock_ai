from __future__ import annotations

import torch

from enc.Model import SentenceLevelTransformer
from enc.utils import collate_documents


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceLevelTransformer(
        roberta_name="klue/roberta-base",
        num_classes=3,
        sent_transformer_layers=2,
        sent_transformer_heads=8,
        pooling="cls",
        freeze_roberta=True,
        max_sentences=8,
    ).to(device)
    model.eval()

    documents = [
        "메모리 가격이 반등하면서 삼성전자 이익 추정치가 상향됐다. 기관 수급도 동시에 유입됐다.",
        [
            "외국인 순매수가 둔화됐다.",
            "단기 과열 구간이라 추격 매수는 부담이다.",
        ],
    ]

    batch = collate_documents(documents, model.tokenizer, max_sentences=8, max_tokens=32)
    batch = {key: value.to(device) for key, value in batch.items()}

    with torch.no_grad():
        embeddings = model(**batch, return_embeddings=True)
        logits = model(**batch)

    print("document_embeddings:", tuple(embeddings.shape))
    print("logits:", tuple(logits.shape))
    print(logits)


if __name__ == "__main__":
    main()
