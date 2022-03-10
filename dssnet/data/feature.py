import os
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision
import torch.nn as nn
import torch

device = 'cuda'
os.environ["TORCH_HOME"] = '/notebooks/data/mine/live/code_v7/model/'
USE_FPN = True
if USE_FPN:
    backbone = backbone_utils.resnet_fpn_backbone('resnet50', True)
    features = list(backbone.children())[:-1]  # 去掉最后的fpn层, 得到resnet的2,3,4层输出
    # features = list(backbone.children()) # 计算图像金字塔输出, 低层包括具体和抽像特征
    model = nn.Sequential(*features)
else:
    backbone = torchvision.models.resnet50(pretrained=True)
    features = list(backbone.children())[:-2]  # 去掉全连接和池化层, 得到最后卷积层输出
    model = nn.Sequential(*features)

model = model.to(device)
x = torch.rand([1, 3, 244, 244]).to(device)
out = model(x)

if USE_FPN:  # 多层输出
    for key, value in out.items():
        print(key, value.shape)
else:  # 单层输出
    print(out.shape)
