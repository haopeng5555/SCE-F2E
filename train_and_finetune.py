import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import TranslationModel

# Load Dataset
with open('data/SCE_F2E_Dataset/SCE_F2E_Dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = {
            'pos': torch.tensor(item['POS'], dtype=torch.float32),
            'dep': torch.tensor(item['Dependency'], dtype=torch.float32),
            'valence': torch.tensor([item['Valence']], dtype=torch.float32),
            'position': torch.tensor([item['Position']], dtype=torch.float32)
        }
        y = torch.tensor(item['Label'], dtype=torch.float32)  # Assuming 'Label' for supervised task
        return x, y

# Split Data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = TranslationDataset(train_data)
val_dataset = TranslationDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model, Optimizer, Loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TranslationModel(input_dim=10, hidden_dim=128, output_dim=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = (outputs > 0.5).float()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall: {rec:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
