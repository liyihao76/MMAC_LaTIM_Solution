import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from sklearn.model_selection import train_test_split
import monai
from PIL import Image
from monai.losses.dice import DiceLoss
from torchvision.transforms.functional import to_pil_image,affine
from monai.transforms import Rand2DElastic
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


### 设置参数
images_file = '/data_GPU/yihao/MMAC/Task2/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/1. Images/1. Training Set'  # 训练图像路径
gt_file = '/data_GPU/yihao/MMAC/Task2/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/2. Groundtruths/1. Training Set'
image_size = 800 # 输入图像统一尺寸
val_ratio = 0.1  # 训练/验证图像划分比例
batch_size = 5 # 批大小
num_workers = 6 # 数据加载处理器个数

summary_dir = './logs'
torch.backends.cudnn.benchmark = True
print('cuda',torch.cuda.is_available())
print('gpu number',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
summaryWriter = SummaryWriter(summary_dir)

# 训练/验证数据集划分
filelists = os.listdir(gt_file)
print(filelists)
#train_filelists, val_filelists = train_test_split(filelists, test_size = val_ratio,random_state=42)
train_filelists = filelists
val_filelists = filelists
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))


### 从数据文件夹中加载眼底图像，提取相应的金标准，生成训练样本
class MACC_Dataset(Dataset):
    def __init__(self, image_file, gt_path=None, filelists=None,  mode='train'):
        super(MACC_Dataset, self).__init__()
        self.mode = mode
        self.image_path = image_file
        self.gt_path = gt_path
        self.patient_list = filelists
   
    def __getitem__(self, idx):
        patient_name = self.patient_list[idx]
        img_path = os.path.join(self.image_path, patient_name)
        gt_path = os.path.join(self.gt_path, patient_name)
        img = cv2.imread(img_path)   
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.imread(gt_path,0)
        gt_img[gt_img == 255] = 1
        
        #img = img[:,:,np.newaxis]
        #print(img.shape)
        #h,w = img.shape # (800, 1100, 3)     
        #img = img[:,:,np.newaxis]
        #print(img.shape)
             
        if self.mode == "train":
            transform = A.Compose([
                A.Flip(),
                A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=90),  # default = A.ShiftScaleRotate()
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ]),
                ##A.CoarseDropout(max_height=5, min_height=1, max_width=512, min_width=51, mask_fill_value=0),
                A.OneOf([
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
                ]),
                # A.RandomResizedCrop(512, 512, p=0.2),
                A.GridDistortion(p=0.2),
                A.CoarseDropout(max_height=128, min_height=32, max_width=128, min_width=32, max_holes=3, p=0.2,
                                mask_fill_value=0.),

                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ])
        elif self.mode != "train":
            transform = A.Compose([
            # A.Resize(input_size, input_size),
            # BensPreprocessing(sigmaX=40),
            # A.Normalize(mean=(0.4128,0.4128,0.4128), std=(0.2331,0.2331,0.2331)),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2(),
        ])
        
        sample = transform(image=img, mask=gt_img)
        img, gt_img = sample['image'], sample['mask']
        gt_img = gt_img[np.newaxis,:,:]
        
        if self.mode == 'test':
            ### 在测试过程中，加载数据返回眼底图像，数据名称，原始图像的高度和宽度
            return img, patient_name
        
        if self.mode == 'train' or self.mode == 'val':
            ###在训练过程中，加载数据返回眼底图像及其相应的金标准           
            return img, gt_img

    def __len__(self):
        return len(self.patient_list)

train_dataset = MACC_Dataset(image_file = images_file, 
                        gt_path = gt_file,
                        filelists=train_filelists)

val_dataset = MACC_Dataset(image_file = images_file, 
                        gt_path = gt_file,
                        filelists=val_filelists,mode='val')


model = smp.MAnet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)

x=torch.randn(1,3,800,800)
output = model(x)
print(output.shape)

model.cuda()
metric = DiceLoss(to_onehot_y = True, softmax = True, include_background = False)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#scheduler = ExponentialLR(optimizer, gamma=0.99)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=True)

def get_dice(gt, pred, classId=1):
    if np.sum(gt) == 0:
        return np.nan
    else:
        intersection = np.logical_and(gt == classId, pred == classId)
        dice_eff = (2. * intersection.sum()) / (gt.sum() + pred.sum())
        return dice_eff


def get_IoU(gt, pred, classId=1):
    if np.sum(gt) == 0:
        return np.nan
    else:
        intersection = np.logical_and(gt == classId, pred == classId)
        union = np.logical_or(gt == classId, pred == classId)
        iou = np.sum(intersection) / np.sum(union)
        return iou


def get_mean_IoU_dice(gts_list, preds_list):
    assert len(gts_list) == len(preds_list)
    dice_list = []
    iou_list = []
    for gt_array, pred_array in zip(gts_list, preds_list):
        dice = get_dice(gt_array, pred_array, 1)
        iou = get_IoU(gt_array, pred_array, 1)
        dice_list.append(dice)
        iou_list.append(iou)
    mDice = np.nanmean(dice_list)
    mIoU = np.nanmean(iou_list)
    return mDice, mIoU
    

best_dice = 0.0
best_model_path = './weights/bestmodel.pth'
num_epochs = 1000
for epoch in range(num_epochs):
    #print('lr now = ', get_learning_rate(optimizer))
    avg_loss_list = []
    avg_dice_list = []
    
    model.train()
    with torch.enable_grad():
        for batch_idx, data in enumerate(train_loader):
            img = (data[0]).float()
            gt_label = (data[1])
            #print(img.shape)
            #print(gt_label.shape)
            
            img = img.cuda()
            gt_label = gt_label.cuda()
            
            
            logits = model(img)
            #print(logits)
            dice = metric(logits,gt_label)
            #loss = criterion(logits, torch.squeeze(gt_label,dim=1).long()) + dice
            loss = 0.5 * criterion(logits, torch.squeeze(gt_label, dim=1).long()) + dice
            #loss = dice
            #print(loss)
            
            avg_loss_list.append(loss.item())
            avg_dice_list.append(dice.item())
            

            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad = None
            
        avg_loss = np.array(avg_loss_list).mean()
        avg_dice = np.array(avg_dice_list).mean()
        print("[TRAIN] epoch={}/{} avg_loss={:.4f} avg_dice={:.4f}".format(epoch, num_epochs, avg_loss, (1.0-avg_dice)))
        summaryWriter.add_scalars('loss', {"loss": (avg_loss)}, epoch)
        summaryWriter.add_scalars('dice', {"dice": avg_dice}, epoch)
        
    model.eval()
    pred_img_list = []
    gt_label_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            
            img = (data[0]).float()
            gt_label = (data[1])

            img = img.cuda()
            gt_label = gt_label.numpy()


            logits = model(img)
            
            pred_img = logits.detach().cpu().numpy().argmax(1).squeeze()
            gt_label = np.squeeze(gt_label)
            
            pred_img_list.append(pred_img)
            gt_label_list.append(gt_label)
            #print(np.unique(pred_img))
            
            
#             mean_Dice, mean_IoU = get_mean_IoU_dice(gt_label_list, pred_img_list)
#             print(mean_Dice)
#             print(mean_IoU)
#             print(pred_img.shape)
#             print(gt_label.shape)
#             print(abc)

        mean_Dice, mean_IoU = get_mean_IoU_dice(gt_label_list, pred_img_list)
        print("[EVAL] epoch={}/{}  mean_Dice={:.4f} mean_IoU={:.4f} ".format(epoch, num_epochs,mean_Dice,mean_IoU))
        summaryWriter.add_scalars('mean_Dice', {"mean_Dice": mean_Dice}, epoch)
        summaryWriter.add_scalars('mean_IoU', {"mean_IoU": mean_IoU}, epoch)
        
    #scheduler.step()

    filepath = './weights'
    folder = os.path.exists(filepath)
    if not folder:
        # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(filepath)
        
    if mean_Dice >= best_dice:
        print('best model epoch = ',epoch)
        print('best dice  =',mean_Dice)
        best_dice = mean_Dice
        torch.save(model.state_dict(), best_model_path)          

summaryWriter.close()




