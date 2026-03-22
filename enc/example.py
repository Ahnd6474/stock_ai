
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceLevelTransformer(
    roberta_name="roberta-base",
    num_classes=3,
    sent_transformer_layers=2,
    sent_transformer_heads=8,
    pooling="cls",
).to(device)

documents = [
    ["이 영화는 정말 재미있었다.", "배우 연기도 훌륭했다.", "다만 결말은 조금 아쉬웠다."],
    ["배송이 빨랐다.", "제품 품질도 괜찮았다."]
]

tokenizer = model.tokenizer
batch = collate_documents(documents, tokenizer, max_sentences=8, max_tokens=32)

batch = {k: v.to(device) for k, v in batch.items()}
logits = model(**batch)

print(logits.shape)  # [B, num_classes]
print(logits)