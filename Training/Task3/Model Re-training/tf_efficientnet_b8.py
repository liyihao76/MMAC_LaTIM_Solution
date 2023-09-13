#train.py

import os
import numpy as np
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
from PIL import Image, ImageOps
import torch.optim as optim
import random
import timm

from torchmetrics import R2Score
from torchmetrics import MeanAbsoluteError

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

#dataloader param 
batch_size = 5 # 批大小.
num_workers = 6 # 数据加载处理器个数
fold_val = 0

#model param 
model_name = "tf_efficientnet_b8"
pretrained = True 
num_classes = 1
in_chans = 3

lr_rate = 1e-4 
wght_decay = 1e-4

num_epochs = 800
best_kappa = 0.0

#read csv
root = "/data_GPU/yihao/2022/tensorflow/Task3/3. Prediction of Spherical Equivalent"
task3_data_root = os.path.join(root, '1. Images/1. Training Set')
file_name = os.path.join(root,"task3_dataset_5_fold_new_split.csv")
dataframe_data = pd.read_csv(file_name)

#functions : 
def data_split_crossval(label_file,fold= 0 ,aff_info = True):
    #fold = 0
    #label_file = 'task2_data_fold.xls'
    dataframe = pd.read_csv(label_file)
    fold0 = dataframe[dataframe['fold']==0]
    fold1 = dataframe[dataframe['fold']==1]
    fold2 = dataframe[dataframe['fold']==2]
    fold3 = dataframe[dataframe['fold']==3]
    fold4 = dataframe[dataframe['fold']==4]
    
    fold0_patient = fold0['image'].tolist()
    fold0_label = fold0['spherical_equivalent'].tolist()
    fold1_patient = fold1['image'].tolist()
    fold1_label = fold1['spherical_equivalent'].tolist()
    fold2_patient = fold2['image'].tolist()
    fold2_label = fold2['spherical_equivalent'].tolist()
    fold3_patient = fold3['image'].tolist()
    fold3_label = fold3['spherical_equivalent'].tolist()
    fold4_patient = fold4['image'].tolist()
    fold4_label = fold4['spherical_equivalent'].tolist()


    if fold == 0:
        train_patient =  fold2_patient + fold3_patient + fold4_patient
        train_label = fold2_label + fold3_label + fold4_label
        val_patient = fold1_patient
        val_label = fold1_label
        test_patient =  fold0_patient
        test_label = fold0_label

    if fold == 1:
        train_patient = fold0_patient + fold3_patient + fold4_patient
        train_label = fold0_label  + fold3_label + fold4_label
        val_patient = fold2_patient
        val_label = fold2_label
        test_patient =  fold1_patient
        test_label = fold1_label

    if fold == 2:
        train_patient = fold1_patient + fold0_patient + fold4_patient
        train_label = fold1_label + fold0_label + fold4_label
        val_patient = fold3_patient
        val_label = fold3_label
        test_patient =  fold2_patient
        test_label = fold2_label

    if fold == 3:
        train_patient = fold1_patient + fold2_patient + fold0_patient
        train_label = fold1_label + fold2_label + fold0_label
        val_patient = fold4_patient
        val_label = fold4_label
        test_patient =  fold3_patient
        test_label = fold3_label

    if fold == 4:
        train_patient = fold1_patient + fold2_patient + fold3_patient 
        train_label = fold1_label + fold2_label + fold3_label
        val_patient = fold0_patient
        val_label = fold0_label
        test_patient =  fold4_patient
        test_label = fold4_label
        
    if aff_info == True:
        print("Total Nums: {}, train: {}, val: {}, test: {}".format(len(train_patient+ val_patient+ test_patient), len(train_patient), len(val_patient), len(test_patient)))
        print('Trainset = ',train_patient)
        print('Train label = ', train_label)
        print('Valset = ',val_patient)
        print('Val label = ', val_label)
        print('Test patient = ', test_patient)
        print('Test label =',test_label)
    
    return train_patient,train_label,val_patient,val_label,test_patient,test_label
 
# 数据加载
class MMAC_task3_dataset(Dataset):
    def __init__(self,
                dataset_root,
                patient_list ='',
                label_list ='',
                mode='train'):
        self.dataset_root = dataset_root
        self.mode = mode
        self.patient_list = patient_list
        self.label_list = label_list
    
    def __getitem__(self, idx):
        
        patient_name = self.patient_list[idx]
        label = self.label_list[idx]
        
        img_path = os.path.join(self.dataset_root, patient_name) 
        img = cv2.imread(img_path)   
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = ImageOps.grayscale(img)
        #img = img.resize((image_size,image_size))
        # normlize on GPU to save CPU Memory and IO consuming.
        # img = (img / 255.).astype("float32")
        if self.mode == "train":
            transform = A.Compose([
                # A.Resize(input_size, input_size),
                # BensPreprocessing(sigmaX=40),
                # # base
                A.Flip(),
                A.ShiftScaleRotate(),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ]),
                A.CoarseDropout(max_height=5, min_height=1, max_width=512, min_width=51),
                A.OneOf([
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
                ]),
                A.Normalize(mean=(0,0,0), std=(1,1,1)),
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
        img = transform(image=img)['image']
        
        #print(img.shape)

        #img = img.transpose(2, 0, 1) # H, W, C -> C, H, W

        if self.mode == 'test':
            return img, np.float32(label), patient_name

        if self.mode == "train" or self.mode == "val" :           
            return img, np.float32(label)

    def __len__(self):
        return len(self.patient_list)

#LOAD DATA
train_patient,train_label,val_patient,val_label,test_patient,test_label = data_split_crossval(label_file = file_name,fold= fold_val,aff_info = True)

train_patient = train_patient + val_patient
train_label = train_label + val_label
val_patient = test_patient
val_label = test_label

# 训练/验证数据集划分
print('########################################## V2 Dataset ##########################################')
print('Trainset = ',train_patient)
print('Train label = ', train_label)
print('Valset = ',val_patient)
print('Val label = ', val_label)
print('Test patient = ', test_patient)
print('Test label =',test_label)

train_dataset = MMAC_task3_dataset(dataset_root=task3_data_root, 
                            patient_list = train_patient,
                            label_list = train_label,
                            mode = 'train')

val_dataset = MMAC_task3_dataset(dataset_root=task3_data_root, 
                          patient_list = val_patient,
                          label_list = val_label,
                          mode = 'val')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=True)

#init model 
model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

x=torch.randn(1,3,800,800)
output = model(x)
print(output.shape)

model.cuda()
#optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=wght_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
#scheduler = ExponentialLR(optimizer, gamma=0.99)
#criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()

#Start Training
summary_dir = './logs'

torch.backends.cudnn.benchmark = True
print('cuda',torch.cuda.is_available())
print('gpu number',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
summaryWriter = SummaryWriter(summary_dir)


best_model_path_r2 = './weights/bestmodel_r2.pth'
best_model_path_mae = './weights/bestmodel_mae.pth'
best_mae = 1000.0
best_r2 = -10.0

for epoch in range(num_epochs):
    #print('lr now = ', get_learning_rate(optimizer))
    avg_loss_list = []
    logits_list = []
    labels_list = []
    
    mean_absolute_error = MeanAbsoluteError()
    r2score = R2Score()
    
    model.train()
    with torch.enable_grad():
        for batch_idx, data in enumerate(train_loader):
            
            img = (data[0])
            labels = (data[1])
            labels_list.extend(labels)
            
            
            img = img.cuda().float()
            labels = labels.cuda()
            #print(labels)
            #print(labels.shape)
            

            logits = model(img)
            logits = torch.squeeze(logits,dim=1)
            #print(logits)
            #print(logits.shape)
            logits_list.extend(logits.detach().cpu())
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad = None
                
            avg_loss_list.append(loss.item())

        
        #print(logits_list)
        #print(labels_list)

        avg_loss = np.array(avg_loss_list).mean()
        avg_r2 = r2score(torch.tensor(logits_list),torch.tensor(labels_list)).item()
        avg_mae = mean_absolute_error(torch.tensor(logits_list),torch.tensor(labels_list)).item()
        #print(avg_r2)
        #print(avg_mae)
        #print(abc)
        
        print("[TRAIN] epoch={}/{} avg_loss={:.4f} avg_r2={:.4f} avg_mae={:.4f}".format(epoch, num_epochs, avg_loss, avg_r2, avg_mae))
        summaryWriter.add_scalars('loss', {"loss": (avg_loss)}, epoch)
        summaryWriter.add_scalars('r2', {"r2": avg_r2}, epoch)
        summaryWriter.add_scalars('mae', {"mae": avg_mae}, epoch)
    
    model.eval()
    val_logits_list = []
    val_labels_list = []
    
    mean_absolute_error = MeanAbsoluteError()
    r2score = R2Score()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            
            img = (data[0])
            labels = (data[1])
            val_labels_list.extend(labels)
            
            img = img.cuda().float()
            labels = labels.cuda()
            logits = model(img)
            logits = torch.squeeze(logits,dim=1)
            val_logits_list.extend(logits.detach().cpu())
            
        
        r2 = r2score(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
        mae = mean_absolute_error(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
        print("[EVAL] epoch={}/{}  val_r2={:.4f} val_mae={:.4f}".format(epoch, num_epochs, r2, mae))
        summaryWriter.add_scalars('r2', {"val_r2": r2}, epoch)
        summaryWriter.add_scalars('mae', {"val_mae": mae}, epoch)
        
    # scheduler.step()
    filepath = './weights'
    folder = os.path.exists(filepath)
    if not folder:
        # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(filepath)
    if r2 >= best_r2:
        print('best r2 model epoch = ',epoch)
        print('best r2  =',r2)
        best_r2 = r2
        torch.save(model.state_dict(), best_model_path_r2) 

    if mae <= best_mae:
        print('best mae model epoch = ',epoch)
        print('best mae  =',mae)
        best_mae = mae
        torch.save(model.state_dict(), best_model_path_mae)
    

print('################# TRAIN FINISH, START TEST (R2) #################')
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=in_chans)
model.load_state_dict(torch.load(best_model_path_r2))
model.cuda()
model.eval()

mean_absolute_error = MeanAbsoluteError()
r2score = R2Score()

test_dataset = MMAC_task3_dataset(dataset_root=task3_data_root, 
                            patient_list = test_patient,
                            label_list = test_label,
                            mode = 'test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=True)


val_logits_list = []
val_labels_list = []
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        #print(batch_idx)

        img = (data[0])
        labels = (data[1])
        val_labels_list.extend(labels)

        img = img.cuda().float()
        labels = labels.cuda()
        logits = model(img)
        logits = torch.squeeze(logits,dim=1)
        val_logits_list.extend(logits.detach().cpu())

    r2 = r2score(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
    avg_mae = mean_absolute_error(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
    print("[TEST] val_r2={:.4f} val_mae={:.4f}".format(r2, avg_mae))


print('################# TRAIN FINISH, START TEST (MAE) #################')
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=in_chans)
model.load_state_dict(torch.load(best_model_path_mae))
model.cuda()
model.eval()

mean_absolute_error = MeanAbsoluteError()
r2score = R2Score()

test_dataset = MMAC_task3_dataset(dataset_root=task3_data_root,
                            patient_list = test_patient,
                            label_list = test_label,
                            mode = 'test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=True)


val_logits_list = []
val_labels_list = []
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        #print(batch_idx)

        img = (data[0])
        labels = (data[1])
        val_labels_list.extend(labels)

        img = img.cuda().float()
        labels = labels.cuda()
        logits = model(img)
        logits = torch.squeeze(logits,dim=1)
        val_logits_list.extend(logits.detach().cpu())

    r2 = r2score(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
    avg_mae = mean_absolute_error(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
    print("[TEST] val_r2={:.4f} val_mae={:.4f}".format(r2, avg_mae))
