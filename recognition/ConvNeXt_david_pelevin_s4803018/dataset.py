import torch, torchvision
from torch.utils.data import Dataset
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