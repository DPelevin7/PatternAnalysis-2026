import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

# Based off code from previous demo
class ADNIDataset(Dataset):
    def __init__(self, root_directory, ad_directory, nc_directory):
        self.images = []
        self.labels = []

        ad_path = Path(root_directory + "/" + ad_directory)
        nc_path = Path(root_directory + "/" + nc_directory)

        for path in ad_path.glob("*.jpeg"):
            image = Image.open(path).convert("L")

            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(1))

        for path in nc_path.glob("*.jpeg"):
            image = Image.open(path).convert("L")
            
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

## ConvNeXt architecture based on https://www.geeksforgeeks.org/computer-vision/convnext/
class ConvNeXtBlock(nn.Module):
    def __init__(self, dimension, layer_scale=1e-6):
        super().__init__()

        # Depthwise convolution settings found in
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
        self.depthConv2d = nn.Conv2d(dimension, dimension, 7, padding=3, groups=dimension)

        self.l1 = nn.LayerNorm(dimension)
        self.conv1 = nn.Conv2d(dimension, dimension * 4, 1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(dimension * 4, dimension, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dimension))

    def forward(self, input):
        carryThrough = input

        input = self.depthConv2d(input)
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.conv1(input)
        input = self.gelu(input)
        input = self.conv2(input)
        input = (self.gamma * input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return input + carryThrough


class downSample(nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.l1 = nn.LayerNorm(dimension)
        self.conv1 = nn.Conv2d(dimension, dimension * 2, 2, 2)

    def forward(self, input):
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.conv1(input)
        return input


class ConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 96, 4, 4)
        self.l1 = nn.LayerNorm(96)

        self.convnext1 = nn.Sequential(
            ConvNeXtBlock(96),
            ConvNeXtBlock(96),
            ConvNeXtBlock(96)
        )

        self.downSample1 = downSample(96)

        self.convnext2 = nn.Sequential(
            ConvNeXtBlock(192),
            ConvNeXtBlock(192),
            ConvNeXtBlock(192)
        )

        self.downSample2 = downSample(192)

        self.convnext3 = nn.Sequential(
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384)
        )

        self.downSample3 = downSample(384)

        self.convnext4 = nn.Sequential(
            ConvNeXtBlock(768),
            ConvNeXtBlock(768),
            ConvNeXtBlock(768)
        )

        self.averagePool = nn.AdaptiveAvgPool2d(1)

        self.l2 = nn.LayerNorm(768)

        self.linear = nn.Linear(768, 2)

    def forward(self, input):
        input = self.conv1(input)
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.convnext1(input)
        input = self.downSample1(input)
        input = self.convnext2(input)
        input = self.downSample2(input)
        input = self.convnext3(input)
        input = self.downSample3(input)
        input = self.convnext4(input)
        input = self.averagePool(input)
        input = self.l2(input.permute(0, 2, 3, 1))
        input = self.linear(input).permute(0, 3, 1, 2)
        input = torch.flatten(input, 1)
        return input

train_loader = DataLoader(ADNIDataset("./ADNI/AD_NC/train", "AD", "NC"), batch_size=32, shuffle=True)
test_loader = DataLoader(ADNIDataset("./ADNI/AD_NC/test", "AD", "NC"), batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epochs = 200
learning_rate = 4e-3

model = ConvNeXt()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
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

    print(f"Epoch: {epoch} Loss: {total_loss / len(train_loader)}")

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