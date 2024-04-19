import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np

class TextInvDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and (f.lower().endswith('.jpeg') or f.lower().endswith('.jpg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dir = "./samples/"
model_file = "./trained_model_gender.pth"

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


image_dataset = TextInvDataset(root_dir=dataset_dir, transform=transform)

data_loader = DataLoader(image_dataset, batch_size=64, shuffle=False)

model = models.alexnet(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(model_file))
model.eval()


predictions = []
image_paths = []
with torch.no_grad():
    for images, paths in data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        predictions.extend(predicted.numpy())
        image_paths.extend(paths)

class_0_images = np.array(image_paths)[np.array(predictions) == 0]
class_1_images = np.array(image_paths)[np.array(predictions) == 1]


N = 5  # number of class 0 samples (should be less than total such outputs)
M = 5  # number of class 1 samples (should be less than total such outputs)
sampled_class_0_images = np.random.choice(class_0_images, N, replace=False)
sampled_class_1_images = np.random.choice(class_1_images, M, replace=False)

