import torch, torchvision
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Calculated using the ADNIDatasetFull class
full_dataset_mean = torch.tensor(29.5524, dtype=torch.float32)
full_dataset_std = torch.tensor(57.5199, dtype=torch.float32)

# Based off code from previous demo
class ADNIDataset(Dataset):
    def __init__(self, root_directory, ad_directory, nc_directory, patients=None):
        self.images = []
        self.labels = []

        ad_path = Path(root_directory + "/" + ad_directory)
        nc_path = Path(root_directory + "/" + nc_directory)

        for path in ad_path.glob("*.jpeg"):
            patient_id = path.name.split(".")[0].split("_")[0]
            if (patients != None) and (patient_id not in patients):
                # Not in this split
                continue

            image = Image.open(path).convert("L")

            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(1))

        for path in nc_path.glob("*.jpeg"):
            patient_id = path.name.split(".")[0].split("_")[0]
            if (patients != None) and (patient_id not in patients):
                # Not in this split
                continue
            
            image = Image.open(path).convert("L")
            
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# Based off code from Shakes for previous demo
def normalise_image(image, dtype=torch.float32):
    image = image.to(dtype)
    _mean = full_dataset_mean.to(dtype=dtype, device=image.device)
    _std = full_dataset_std.to(dtype=dtype, device=image.device)
    return (image - _mean) / _std

# Based off code from Shakes for previous demo
def preprocess_image(image, crop_pad=4):
    image = normalise_image(image)

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
        image = preprocess_image(image)
        return image, label

class ADNI_test(ADNIDataset):
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = normalise_image(image)
        return image, label

class ADNIDatasetFull(Dataset):
    def __init__(self, train_root_directory, test_root_directory, ad_directory, nc_directory):
        self.images = []
        self.labels = []
        self.mean = 0
        self.std = 0

        total_sum = 0.0
        total_sum_sq = 0.0
        total_pixels = 0

        train_ad_path = Path(train_root_directory + "/" + ad_directory)
        train_nc_path = Path(train_root_directory + "/" + nc_directory)
        test_ad_path = Path(test_root_directory + "/" + ad_directory)
        test_nc_path = Path(test_root_directory + "/" + nc_directory)

        paths = [train_ad_path, train_nc_path, test_ad_path, test_nc_path]

        for i in range(4):
            for path in paths[i].glob("*.jpeg"):
                image = Image.open(path).convert("L")

                # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
                image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

                self.images.append(image_tensor)
                self.labels.append(torch.tensor(1 - (i % 2)))

                total_sum += image_tensor.sum()
                total_sum_sq += (image_tensor**2).sum()
                total_pixels += image_tensor.size(1) * image_tensor.size(2)
            print("Path complete")
        
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

def record_unique_patients(directory):
    patients = []

    for path in directory.glob("*.jpeg"):
        file_name = path.name.split(".")[0]
        patient = file_name.split("_")[0]
        if patient not in patients: patients.append(patient)

    return patients

def train_val_split(root_directory, ad_directory, nc_directory):
    ad_patients = record_unique_patients(Path(root_directory + "/" + ad_directory))
    nc_patients = record_unique_patients(Path(root_directory + "/" + nc_directory))

    ad_train, ad_val = train_test_split(ad_patients, test_size=0.4, random_state=50)
    nc_train, nc_val = train_test_split(nc_patients, test_size=0.4, random_state=50)

    train_split = ad_train + nc_train
    val_split = ad_val + nc_val

    return train_split, val_split