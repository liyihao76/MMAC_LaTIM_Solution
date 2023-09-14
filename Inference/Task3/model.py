import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Model_ensemble(nn.Module):
    def __init__(self,load_weights,path1,path2):
        super(Model_ensemble, self).__init__()

        self.model1 = timm.create_model('tf_efficientnetv2_l', pretrained=False, num_classes=1, in_chans=3)
        if load_weights == True:
            self.model1.load_state_dict(torch.load(path1, map_location=torch.device("cpu")))
        
        self.model2 = timm.create_model('tf_efficientnetv2_l', pretrained=False, num_classes=1, in_chans=3)     
        if load_weights == True:
            self.model2.load_state_dict(torch.load(path2, map_location=torch.device("cpu")))

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out = (out1+out2)/2
#         print('out1=',out1)
#         print('out2=',out2)
#         print('out=',out)
        return out


class model:
    def __init__(self):
        self.checkpoint = "v2_f3.pth"
        self.checkpoint2 = "v3_f4.pth"
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        #self.model = Model_ensemble(load_weights=False)
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        checkpoint_path2 = os.path.join(dir_path, self.checkpoint2)
        self.model = Model_ensemble(True,checkpoint_path,checkpoint_path2)
        #self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image, patient_info_dict):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: a float value indicating the spherical equivalent value for the input image.
        """
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        transform = A.Compose([
            # A.Resize(input_size, input_size),
            # BensPreprocessing(sigmaX=40),
            # A.Normalize(mean=(0.4128,0.4128,0.4128), std=(0.2331,0.2331,0.2331)),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2(),
        ])
        img = transform(image=img)['image'].unsqueeze(0)
        #image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device, torch.float)

        with torch.no_grad():
            score = self.model(img)

        score = score.detach().cpu()
        return float(score)


