from sklearn.model_selection import GridSearchCV
import torch.nn as nn
from multiprocessing import Pool
import torch
import itertools
from tqdm import tqdm
from functools import partial
import numpy as np
from Unet import ResUNet, AttentionUNet, VanillaUNet
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
class CarvanaCustomDataset(Dataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids = [file.stem for file in self.images_dir.glob('*.tif') if not file.name.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
    def __len__(self):
        return len(self.ids)

    def shuffle_data(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        self.ids = [self.ids[i] for i in indices]
    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomAffine(5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        return img

    def preprocess_mask(self, mask_path):
        mask = Image.open(mask_path)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        mask = transform(mask).float()
        return mask
    def __getitem__(self, idx):
        image_name = self.ids[idx]  # Get the image file name without any suffix
        image_filename = f"{image_name}.tif"
        mask_filename = f"{image_name}.tif"  # Modify this to match your naming convention for mask files
        image_path = os.path.join(self.images_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)
        image = self.preprocess_image(image_path)
        mask = self.preprocess_mask(mask_path)

        return {
            'image': image,
            'mask': mask
        }
images_dir = 'C:\\Users\\ADMIN\\Desktop\\tumor\\train'
mask_dir = 'C:\\Users\\ADMIN\\Desktop\\tumor\\train_a'
scale = 0.5
batch_size = 4
num_workers=4
custom_dataset = CarvanaCustomDataset(images_dir, mask_dir, scale)
custom_dataset.shuffle_data()
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
resunet_model = ResUNet(in_channels=3, out_channels=1)
resunet_model.load_state_dict(torch.load('best_resunet_weights.pth'))
resunet_model.eval()
attentionunet_model = AttentionUNet(in_channels=3, out_channels=1)
attentionunet_model.load_state_dict(torch.load('best_attentionunet_weights.pth'))
attentionunet_model.eval()
vanillaunet_model = VanillaUNet(in_channels=3, out_channels=1)
vanillaunet_model.load_state_dict(torch.load('best_vanillaunet_weights.pth'))
vanillaunet_model.eval()
weight_range = [1.0, 0.8, 0.6]
weight_combinations = list(itertools.product(weight_range, repeat=3))
class Ensemble(nn.Module):
    def __init__(self, resunet_model, attentionunet_model, vanillaunet_model, weights):
        super(Ensemble, self).__init__()
        self.resunet_model = resunet_model
        self.attentionunet_model = attentionunet_model
        self.vanillaunet_model = vanillaunet_model
        self.weights = weights
    def forward(self, x):
        output_resunet = self.resunet_model(x)
        output_attentionunet = self.attentionunet_model(x)
        output_vanillaunet = self.vanillaunet_model(x)
        ensemble_output = (output_resunet * self.weights[0] +
                           output_attentionunet * self.weights[1] +
                           output_vanillaunet * self.weights[2])
        return ensemble_output
    
class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=0.26):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, inputs, targets):
            bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-bce_loss)
            focal_loss = (self.alpha * (1 - pt) ** self.gamma) * bce_loss
            return torch.mean(focal_loss)
criterion=FocalLoss(gamma=2,alpha=0.26)
if __name__ == '__main__':
    best_weights = None
    best_loss = float('inf')
    for weights in weight_combinations:
        ensemble_model = Ensemble(resunet_model, attentionunet_model, vanillaunet_model, weights)
        ensemble_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image']
                targets = batch['mask']
                ensemble_output = ensemble_model(inputs)
                loss = criterion(ensemble_output, targets)
                total_loss += loss.item()

        if total_loss < best_loss:
            best_loss = total_loss
            best_weights = weights

    print("Best weights:", best_weights)
    print("Best loss:", best_loss)
    torch.save(ensemble_model.state_dict(), 'best_ensemble_weights.pth')
