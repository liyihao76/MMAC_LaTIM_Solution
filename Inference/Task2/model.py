import os
import cv2
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


class model:
    def __init__(self):
        self.checkpoint_LC = "bestmodel_LC.pth"
        self.checkpoint_CNV = "bestmodel_CNV.pth"
        self.checkpoint_FS = "bestmodel_FS.pth"
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
        self.model_LC = smp.MAnet(
                            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            classes=2,                      # model output channels (number of classes in your dataset)
                        )
        checkpoint_path_LC = os.path.join(dir_path, self.checkpoint_LC)
        self.model_LC.load_state_dict(torch.load(checkpoint_path_LC, map_location=self.device))
        self.model_LC.to(self.device)
        self.model_LC.eval()

        self.model_CNV = smp.MAnet(
                            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            classes=2,                      # model output channels (number of classes in your dataset)
                        )
        checkpoint_path_CNV = os.path.join(dir_path, self.checkpoint_CNV)
        self.model_CNV.load_state_dict(torch.load(checkpoint_path_CNV, map_location=self.device))
        self.model_CNV.to(self.device)
        self.model_CNV.eval()

        self.model_FS = smp.MAnet(
                            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            classes=2,                      # model output channels (number of classes in your dataset)
                        )
        checkpoint_path_FS = os.path.join(dir_path, self.checkpoint_FS)
        self.model_FS.load_state_dict(torch.load(checkpoint_path_FS, map_location=self.device))
        self.model_FS.to(self.device)
        self.model_FS.eval()

    def predict(self, input_image, lesion_type, patient_info_dict):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param lesion_type: a string indicates the lesion type of the input image: 'Lacquer_Cracks' or 'Choroidal_Neovascularization' or 'Fuchs_Spot'.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: a ndarray indicates the predicted segmentation mask with the shape 800 x 800.
        The pixel value for the lesion area is 255, and the background pixel value is 0.
        """
        #image = cv2.resize(input_image, (512, 512))
        #image = image / 255
        #image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        img0 = np.copy(img)
        img90 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
        img180 = cv2.rotate(img0, cv2.ROTATE_180)
        img270 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

        transform = A.Compose([
            # A.Resize(input_size, input_size),
            # BensPreprocessing(sigmaX=40),
            # A.Normalize(mean=(0.4128,0.4128,0.4128), std=(0.2331,0.2331,0.2331)),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ])

        img90 = transform(image=img90)['image'].unsqueeze(0)
        img180 = transform(image=img180)['image'].unsqueeze(0)
        img270 = transform(image=img270)['image'].unsqueeze(0)
        img = transform(image=img)['image'].unsqueeze(0)
        
        img = img.to(self.device, torch.float)
        img2 = img90.to(self.device, torch.float)
        img3 = img180.to(self.device, torch.float)
        img4 = img270.to(self.device, torch.float)

        #image = image.to(self.device, torch.float)
        pred_mask = None
        with torch.no_grad():
            if lesion_type == 'Lacquer_Cracks':
                pred_mask1 = self.model_LC(img)
                pred_mask2 = self.model_LC(img2)
                pred_mask3 = self.model_LC(img3)
                pred_mask4 = self.model_LC(img4)
            elif lesion_type == 'Choroidal_Neovascularization':
                pred_mask1 = self.model_CNV(img)
                pred_mask2 = self.model_CNV(img2)
                pred_mask3 = self.model_CNV(img3)
                pred_mask4 = self.model_CNV(img4)

            elif lesion_type == 'Fuchs_Spot':
                pred_mask1 = self.model_FS(img)
                pred_mask2 = self.model_FS(img2)
                pred_mask3 = self.model_FS(img3)
                pred_mask4 = self.model_FS(img4)

        pred_mask1 = np.transpose(pred_mask1.numpy().squeeze(), [1, 2, 0])
        pred_mask2 = np.transpose(pred_mask2.numpy().squeeze(), [1, 2, 0])
        pred_mask3 = np.transpose(pred_mask3.numpy().squeeze(), [1, 2, 0])
        pred_mask4 = np.transpose(pred_mask4.numpy().squeeze(), [1, 2, 0])

        pred_mask2 = cv2.rotate(pred_mask2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pred_mask3 = cv2.rotate(pred_mask3, cv2.ROTATE_180)
        pred_mask4 = cv2.rotate(pred_mask4, cv2.ROTATE_90_CLOCKWISE)

        pred_mask = (pred_mask1 + pred_mask2 + pred_mask3 + pred_mask4)
        pred_mask = pred_mask.argmax(2).squeeze()
        pred_mask = pred_mask * 255
        #pred_mask = torch.sigmoid(pred_mask)
        #pred_mask = pred_mask.detach().squeeze().numpy()
        #pred_mask = np.array(pred_mask > 0.5, dtype=np.uint8) * 255
        #pred_mask = cv2.resize(pred_mask, (800, 800), interpolation=cv2.INTER_NEAREST)
        return pred_mask
