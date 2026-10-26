import torch, torchvision
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch.nn.functional as F


# Based off code from previous demo
class ADNIDataset(Dataset):
    def __init__(self, root_directory, ad_directory, nc_directory):
        self.images = []
        self.labels = []
        self.mean = 0
        self.std = 0

        total_sum = 0.0
        total_sum_sq = 0.0
        total_pixels = 0

        ad_path = Path(root_directory + "/" + ad_directory)
        nc_path = Path(root_directory + "/" + nc_directory)

        for path in ad_path.glob("*.jpeg"):
            image = Image.open(path).convert("L")

            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(1))

            total_sum += image_tensor.sum()
            total_sum_sq += (image_tensor**2).sum()
            total_pixels += image_tensor.size(1) * image_tensor.size(2)

        for path in nc_path.glob("*.jpeg"):
            image = Image.open(path).convert("L")
            
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(0))

            total_sum += image_tensor.sum()
            total_sum_sq += (image_tensor**2).sum()
            total_pixels += image_tensor.size(1) * image_tensor.size(2)
        
        self.mean = total_sum / total_pixels
        self.std = torch.sqrt((total_sum_sq / total_pixels) - (self.mean**2))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std


# Based off code from Shakes for previous demo
def normalise_image(image, mean, std, dtype=torch.float32):
    image = image.to(dtype) / 255.0
    _mean = mean.to(dtype=dtype, device=image.device)
    _std = std.to(dtype=dtype, device=image.device)
    return (image - _mean) / _std

# Based off code from Shakes for previous demo
def preprocess_image(image, mean, std, crop_pad=4):
    image = normalise_image(image, mean, std)

    #horizontal flip
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=[2])

    _, height, width = image.shape

    #pad (reflect) then random crop back to the original size
    image = F.pad(image.unsqueeze(0), [crop_pad, crop_pad, crop_pad, crop_pad], mode='reflect')
    image = image.squeeze(0)
    top = int(torch.randint(0, 2 * crop_pad + 1, (1,)))
    left = int(torch.randint(0, 2 * crop_pad + 1, (1,)))
    image = image[:, top:top + height, left:left + width]

    return image

class ADNI_train(ADNIDataset):
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = preprocess_image(image, self.mean, self.std)
        return image, label

class ADNI_test(ADNIDataset):
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = normalise_image(image, self.mean, self.std)
        return image, label