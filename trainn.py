import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
from PIL import Image
import torch.nn.init as init
import os
import time
from Unet import ResUNet, AttentionUNet, VanillaUNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

torch.manual_seed(0)
def load_image(filename):
    ext = filename.suffix.lower()
    if ext in ['.npy', '.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == '.png':
        return Image.open(filename)
    else:
        return Image.open(filename)

def unique_mask_values(idx, mask_dir):
    #idx = idx.split('_')[-1]
    mask_file = mask_dir / f"{idx}.tif"
    mask = np.asarray(load_image(mask_file))

    if not mask_file.is_file():
        raise ValueError(f"No mask file found for index: {idx}")

    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class CarvanaCustomDataset(Dataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [file.stem for file in self.images_dir.glob('*.tif') if not file.name.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))

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
        image_name = self.ids[idx]  
        image_filename = f"{image_name}.tif"
        mask_filename = f"{image_name}.tif"  
        image_path = os.path.join(self.images_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)
        image = self.preprocess_image(image_path)
        mask = self.preprocess_mask(mask_path)

        return {
            'image': image,
            'mask': mask
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ResUNet or Attention U-Net")
    parser.add_argument("--model", type=str, choices=["resunet", "attentionunet", "vanillaunet"], required=True, help="Choose the model to train")
    args = parser.parse_args()

    images_dir = 'C:\\Users\\ADMIN\\Desktop\\tumor\\train'
    mask_dir = 'C:\\Users\\ADMIN\\Desktop\\tumor\\train_a'

    scale = 0.5

    custom_dataset = CarvanaCustomDataset(images_dir, mask_dir, scale)
    custom_dataset.shuffle_data()

    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

    batch_size = 4
    learning_rate = 0.0001
    num_epochs = 30
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

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

    class DiceLoss(nn.Module):
        def __init__(self):
            super(DiceLoss, self).__init()

        def forward(self, inputs, targets):
            smooth = 1.0
            intersection = (inputs * targets).sum()
            union = inputs.sum() + targets.sum()
            dice = (2.0 * intersection + smooth) / (union + smooth)
            loss = 1 - dice
            return loss
    class DiceBCELoss(nn.Module):
        def __init__(self, weight_dice=0.08):
            super(DiceBCELoss, self).__init__()
            self.weight_dice = weight_dice

        def forward(self, inputs, targets):
            smooth = 1.0
            intersection = (inputs * targets).sum()
            union = inputs.sum() + targets.sum()
            dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
            bce_loss = nn.BCEWithLogitsLoss()(inputs, targets)
            combined_loss = (self.weight_dice * dice_loss) + ((1 - self.weight_dice) * bce_loss)
        
            return combined_loss
    if args.model == "resunet":
        criterion = FocalLoss(gamma=2, alpha=0.26)
        in_channels = 3
        out_channels = 1
        model = ResUNet(in_channels, out_channels)
    elif args.model == "attentionunet":
        criterion = FocalLoss(gamma=2, alpha=0.26)
        in_channels = 3
        out_channels = 1
        model = AttentionUNet(in_channels, out_channels)
    elif args.model == "vanillaunet":
        criterion = FocalLoss(gamma=2, alpha=0.26)
        in_channels = 3
        out_channels = 1
        model = VanillaUNet(in_channels, out_channels)
    else:
        raise ValueError("Invalid model choice. Use 'resunet', 'attentionunet', or 'vanillaunet'.")
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    best_loss = float('inf')
    best_model_weights = None
    start_time = time.time()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch['image']
            targets = batch['mask']
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image']
                targets = batch['mask']
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - {args.model.capitalize()} - Train Loss: {loss:.3f} - Val Loss: {val_loss:.3f}")
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] took {epoch_time} seconds")
        start_time = time.time()
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = model.state_dict()
    torch.save(best_model_weights, f'best_{args.model}_weights.pth')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label=f'{args.model.capitalize()} Loss')
    plt.title(f'{args.model.capitalize()} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_losses, label=f'{args.model.capitalize()} Validation Loss')
    plt.title(f'{args.model.capitalize()} Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
