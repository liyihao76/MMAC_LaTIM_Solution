import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import timm
import cv2
import torch.nn as nn
from resnet50 import Resnet50Model

class model:
    def __init__(self):
        self.checkpoints = ["bestmodel_f1_resnet50_1686935892.9994488.pth", "bestmodel_kappa_resnet50_1686935892.9994488.pth", "bestmodel_mean_resnet50_1686935892.9994488.pth", "bestmodel_sp_resnet50_1686935892.9994488.pth"]
        self.device = torch.device("cpu")
        self.models = []

    def load(self, dir_path):
        for checkpoint in self.checkpoints:
            backbone = timm.create_model('resnet50', pretrained=False, num_classes=5, in_chans=3)
            bb = nn.Sequential(*list(backbone.children())[:-1])
            model = Resnet50Model(bb)
            checkpoint_path = os.path.join(dir_path, checkpoint)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def predict(self, input_image, patient_info_dict):
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = Image.fromarray(img)
        img = transforms.PILToTensor()(img).unsqueeze(0)
        img = img.to(self.device, torch.float)

        scores = []
        with torch.no_grad():
            for model in self.models:
                score = model(img)
                scores.append(score)

        # Compute mean score over all models and get class with maximum mean score
        mean_score = torch.mean(torch.stack(scores), dim=0)
        _, pred_class = torch.max(mean_score, 1)
        pred_class = pred_class.detach().cpu()

        return int(pred_class)
