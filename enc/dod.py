from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-base')
unmasker("Hello I'm a <mask> model.")
