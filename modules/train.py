import torch
import torch_directml
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from model import load_summarization_model, load_tokenizer, create_loss_function, forward_pass
from sklearn.model_selection import train_test_split

# Custom Dataset
class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length=512):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        labels = self.tokenizer(summary, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = labels['input_ids'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Fungsi pelatihan
def train_model(model, dataset, tokenizer, epochs=3, batch_size=4, learning_rate=1e-5):
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss function
    loss_function = create_loss_function()

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            loss = forward_pass(model, input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Fungsi untuk memuat data
def load_data():
    # Contoh data
    texts = [
        "Teks artikel panjang pertama.",
        "Teks artikel panjang kedua.",
    ]
    summaries = [
        "Ringkasan pertama.",
        "Ringkasan kedua.",
    ]
    return texts, summaries

# Main function
def main():
    # Inisialisasi device DirectML
    device = torch_directml.device()

    # Load model and tokenizer
    model = load_summarization_model().to(device)
    tokenizer = load_tokenizer()

    # Load data
    texts, summaries = load_data()

    # Split data into train and validation sets
    train_texts, val_texts, train_summaries, val_summaries = train_test_split(texts, summaries, test_size=0.1)

    # Create datasets
    train_dataset = SummarizationDataset(train_texts, train_summaries, tokenizer)

    # Train the model
    train_model(model, train_dataset, tokenizer)

if __name__ == "__main__":
    main()
