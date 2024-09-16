import torch
import torch_directml
from transformers import AutoModel, AutoModelForSeq2SeqLM

device = torch_directml.device()

# Fungsi memuat model IndoBERT
def load_indobert_model():
    model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
    model.to(device)  # Move the model to DirectML device (or the appropriate device)
    return model


def summarize_text(inputs, model):
    # Generate summary using the model's generation method
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150,  # You can adjust the max length of the summary
            num_beams=2,  # You can adjust beam search for better results
            early_stopping=True
        )

    return summary_ids

