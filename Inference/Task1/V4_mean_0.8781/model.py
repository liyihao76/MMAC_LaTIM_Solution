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
        self.checkpoint = "bestmodel_mean_resnet50_1686324992.1358218.pth"
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
        backbone = timm.create_model('resnet50', pretrained=False, num_classes=5, in_chans=3)
        bb = nn.Sequential(*list(backbone.children())[:-1])
        self.model = Resnet50Model(bb)
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
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
        :return: an int value indicating the class for the input image.
        """
        #image = cv2.resize(input_image, (512, 512))
        #image = image / 255
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = Image.fromarray(img)
        img = transforms.PILToTensor()(img).unsqueeze(0)
        #image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device, torch.float)

        with torch.no_grad():
            score = self.model(img)
        _, pred_class = torch.max(score, 1)
        pred_class = pred_class.detach().cpu()

        return int(pred_class)


