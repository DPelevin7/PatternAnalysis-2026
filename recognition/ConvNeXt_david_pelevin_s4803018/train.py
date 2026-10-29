import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import modules
import dataset

# Removes Randomness and allows for Reproducibility
# Based off suggestions from https://docs.pytorch.org/docs/2.15/notes/randomness.html
torch.manual_seed(50)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

train_generator = torch.Generator()
train_generator.manual_seed(50)

train_split, val_split = dataset.train_val_split("/content/drive/My Drive/AD_NC/train", "AD", "NC")

train_loader = DataLoader(dataset.ADNI_train("./ADNI/AD_NC/train", "AD", "NC", train_split), batch_size=32, shuffle=True, generator=train_generator)
val_loader = DataLoader(dataset.ADNI_test("./ADNI/AD_NC/train", "AD", "NC", val_split), batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.ADNI_test("./ADNI/AD_NC/test", "AD", "NC"), batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epochs = 100
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

    model.eval()
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