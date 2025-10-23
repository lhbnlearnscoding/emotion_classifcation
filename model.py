import torch.nn as nn
from transformers import AutoModelForSequenceClassification

def build_model(model_name, num_labels, id2label,label2id):
    """Build and return a sequence classification model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model   
