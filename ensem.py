import os
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from Unet import ResUNet, AttentionUNet, VanillaUNet
class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.resunet = ResUNet(in_channels=3, out_channels=1)
        self.attentionunet = AttentionUNet(in_channels=3, out_channels=1)  
        self.vanillaunet = VanillaUNet(in_channels=3, out_channels=1)  
        self.res_weight = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.attn_weight = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.vanilla_weight = nn.Parameter(torch.tensor(0.1), requires_grad=False)
    def forward(self, x):
        res_out = self.resunet(x)
        attn_out = self.attentionunet(x)
        vanilla_out = self.vanillaunet(x)
        total_weight = self.res_weight + self.attn_weight + self.vanilla_weight
        res_weight = self.res_weight / total_weight
        attn_weight = self.attn_weight / total_weight
        vanilla_weight = self.vanilla_weight / total_weight
        ensemble_out = res_weight * res_out + attn_weight * attn_out + vanilla_weight * vanilla_out
        return ensemble_out
ensemble_model = Ensemble()
ensemble_model.resunet.load_state_dict(torch.load('best_resunet_weights.pth'))  
ensemble_model.attentionunet.load_state_dict(torch.load('best_attentionunet_weights.pth')) 
ensemble_model.vanillaunet.load_state_dict(torch.load('best_vanillaunet_weights.pth')) 
ensemble_model.resunet.eval()
ensemble_model.attentionunet.eval()
ensemble_model.vanillaunet.eval()
torch.save(ensemble_model.state_dict(), 'ensemble_model_weights.pth')





