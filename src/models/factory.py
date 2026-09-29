from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import (
    ID2LABEL,
    LABEL2ID,
    MODEL_NAME,
)


def create_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )


def create_sequence_classifier():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )