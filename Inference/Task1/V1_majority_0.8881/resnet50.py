import torch.nn as nn



import torch.nn as nn

class Resnet50Model(nn.Module):  # 
    def __init__(self, backbone):
        super(Resnet50Model, self).__init__() 
        self.backbone = backbone
        self.fc = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.backbone(x)
        #x = torch.flatten(x, start_dim=1)  
        x = self.fc(x)
        return x




