import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import modules
import dataset

train_split, val_split = dataset.train_val_split("/content/drive/My Drive/AD_NC/train", "AD", "NC")

train_loader = DataLoader(dataset.ADNI_train("./ADNI/AD_NC/train", "AD", "NC", train_split), batch_size=32, shuffle=True)
val_loader = DataLoader(dataset.ADNI_test("./ADNI/AD_NC/train", "AD", "NC", val_split), batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.ADNI_test("./ADNI/AD_NC/test", "AD", "NC"), batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epochs = 75
learning_rate = 4e-4

model = modules.ConvNeXt()
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training Start")

# Based off code from previous demo
model.train()
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch: {epoch} Loss: {total_loss / len(train_loader)} Validation accuracy: {correct/total:.4f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test accuracy: {correct/total:.4f}")

torch.save(model.state_dict(), './model_weights/convnext.pth')