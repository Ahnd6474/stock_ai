def collate_documents(documents, tokenizer, max_sentences=32, max_tokens=64):
    """
    documents: List[List[str]]
        예: [
            ["첫 문장입니다.", "둘째 문장입니다."],
            ["다른 문서의 첫 문장.", "둘째.", "셋째."]
        ]
    """
    batch_size = len(documents)

    all_input_ids = []
    all_attention_masks = []
    all_sentence_masks = []

    for doc in documents:
        doc = doc[:max_sentences]
        num_sents = len(doc)

        encoded = tokenizer(
            doc,
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"]          # [S, T]
        attention_mask = encoded["attention_mask"]

        # 문장 수 패딩
        if num_sents < max_sentences:
            pad_s = max_sentences - num_sents
            pad_ids = torch.zeros((pad_s, max_tokens), dtype=torch.long)
            pad_mask = torch.zeros((pad_s, max_tokens), dtype=torch.long)

            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)

        sentence_mask = torch.zeros(max_sentences, dtype=torch.long)
        sentence_mask[:num_sents] = 1

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_sentence_masks.append(sentence_mask)

    batch = {
        "input_ids": torch.stack(all_input_ids, dim=0),          # [B, S, T]
        "attention_mask": torch.stack(all_attention_masks, dim=0),
        "sentence_mask": torch.stack(all_sentence_masks, dim=0), # [B, S]
    }
    return batch