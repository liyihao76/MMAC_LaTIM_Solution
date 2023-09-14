import os
import cv2
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math

__all__ = ['U2NET_full', 'U2NET_lite']


def _upsample_like(x, size):
    return nn.Upsample(size=size, mode='bilinear', align_corners=False)(x)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module('rebnconvin', REBNCONV(in_ch, out_ch))
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module(f'rebnconv1', REBNCONV(out_ch, mid_ch))
        self.add_module(f'rebnconv1d', REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', REBNCONV(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', REBNCONV(mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', REBNCONV(mid_ch, mid_ch, dilate=dilate))


class U2NET(nn.Module):
    def __init__(self, cfgs, out_ch):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self._make_layers(cfgs)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            # return [torch.sigmoid(x) for x in maps]
            return [x for x in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', nn.Conv2d(v[2], self.out_ch, 3, padding=1))
        # build fuse layer
        self.add_module('outconv', nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1))


# def U2NET_full(out_ch=1):
#     full = {
#         # cfgs for building RSUs and sides
#         # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
#         'stage1': ['En_1', (7, 1, 32, 64), -1],
#         'stage2': ['En_2', (6, 64, 32, 128), -1],
#         'stage3': ['En_3', (5, 128, 64, 256), -1],
#         'stage4': ['En_4', (4, 256, 128, 512), -1],
#         'stage5': ['En_5', (4, 512, 128, 512, True), -1],
#         'stage6': ['En_6', (4, 512, 128, 512, True), 512],
#         'stage5d': ['De_5', (4, 1024, 128, 512, True), 512],
#         'stage4d': ['De_4', (4, 1024, 128, 256), 256],
#         'stage3d': ['De_3', (5, 512, 64, 128), 128],
#         'stage2d': ['De_2', (6, 256, 32, 64), 64],
#         'stage1d': ['De_1', (7, 128, 16, 64), 64],
#     }
#     return U2NET(cfgs=full, out_ch=out_ch)

def U2NET_full(out_ch=1):
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 32, 64), -1],
        'stage2': ['En_2', (6, 64, 32, 128), -1],
        'stage3': ['En_3', (5, 128, 64, 256), -1],
        'stage4': ['En_4', (4, 256, 128, 512), -1],
        'stage5': ['En_5', (4, 512, 256, 512, True), -1],
        'stage6': ['En_6', (4, 512, 256, 512, True), 512],
        'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
        'stage4d': ['De_4', (4, 1024, 128, 256), 256],
        'stage3d': ['De_3', (5, 512, 64, 128), 128],
        'stage2d': ['De_2', (6, 256, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2NET(cfgs=full, out_ch=out_ch)


# def U2NET_lite(out_ch=1):
#     lite = {
#         # cfgs for building RSUs and sides
#         # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
#         'stage1': ['En_1', (7, 3, 16, 64), -1],
#         'stage2': ['En_2', (6, 64, 16, 64), -1],
#         'stage3': ['En_3', (5, 64, 16, 64), -1],
#         'stage4': ['En_4', (4, 64, 16, 64), -1],
#         'stage5': ['En_5', (4, 64, 16, 64, True), -1],
#         'stage6': ['En_6', (4, 64, 16, 64, True), 64],
#         'stage5d': ['De_5', (4, 128, 16, 64, True), 64],
#         'stage4d': ['De_4', (4, 128, 16, 64), 64],
#         'stage3d': ['De_3', (5, 128, 16, 64), 64],
#         'stage2d': ['De_2', (6, 128, 16, 64), 64],
#         'stage1d': ['De_1', (7, 128, 16, 64), 64],
#     }
#     return U2NET(cfgs=lite, out_ch=out_ch)

def U2NET_lite(out_ch=1):
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 32, 64), -1],
        'stage2': ['En_2', (6, 64, 32, 64), -1],
        'stage3': ['En_3', (5, 64, 32, 64), -1],
        'stage4': ['En_4', (4, 64, 32, 64), -1],
        'stage5': ['En_5', (4, 64, 32, 64, True), -1],
        'stage6': ['En_6', (4, 64, 32, 64, True), 64],
        'stage5d': ['De_5', (4, 128, 32, 64, True), 64],
        'stage4d': ['De_4', (4, 128, 32, 64), 64],
        'stage3d': ['De_3', (5, 128, 32, 64), 64],
        'stage2d': ['De_2', (6, 128, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 32, 64), 64],
    }
    return U2NET(cfgs=lite, out_ch=out_ch)


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
        self.model_LC = U2NET_full(2)
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
                pred_mask1 = self.model_LC(img)[0]
                pred_mask2 = self.model_LC(img2)[0]
                pred_mask3 = self.model_LC(img3)[0]
                pred_mask4 = self.model_LC(img4)[0]
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
