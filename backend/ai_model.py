import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Load and prepare the fashion knowledge base
df = pd.read_csv('fashion_kb.csv')
questions = df['question'].tolist()
answers = df['answer'].tolist()

# Tokenize the text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(questions, truncation=True, padding=True, return_tensors='pt')

# Prepare labels
labels = torch.tensor([1] * len(questions))  # Adjust labels as needed

# Create PyTorch dataset and dataloaders
dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], labels)
batch_size = 16
train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

# Instantiate and fine-tune the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f\"Epoch {epoch + 1}/3, Loss: {total_loss / len(train_dataloader)}\")

# Save the trained model
model.save_pretrained('models/trained_fashion_model')
tokenizer.save_pretrained('models/trained_fashion_model')
