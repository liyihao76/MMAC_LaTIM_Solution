import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import timm
import cv2
import torch.nn as nn
from resnet50 import Resnet50Model
from albumentations import Compose, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2

class model:
    def __init__(self, tta_steps=5):
        self.checkpoints = ["bestmodel_f1_resnet50_1686935892.9994488.pth", "bestmodel_kappa_resnet50_1686935892.9994488.pth", "bestmodel_mean_resnet50_1686935892.9994488.pth", "bestmodel_sp_resnet50_1686935892.9994488.pth"]
        self.device = torch.device("cpu")
        self.models = []
        self.tta_steps = tta_steps
        self.transforms = Compose([
                HorizontalFlip(),
                VerticalFlip(),
                ToTensorV2()
            ])

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

        all_predictions = []
        with torch.no_grad():
            for _ in range(self.tta_steps):
                data = self.transforms(image=np.array(img))['image']
                data = data.unsqueeze(0).to(self.device, torch.float)
                predictions = []
                for model in self.models:
                    score = model(data)
                    _, pred_class = torch.max(score, 1)
                    pred_class = pred_class.detach().cpu().numpy()
                    predictions.append(pred_class)
                all_predictions.extend(predictions)

        all_predictions = np.concatenate(all_predictions)
        prediction = np.bincount(all_predictions).argmax()
        
        return int(prediction)

