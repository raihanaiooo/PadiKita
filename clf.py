import flask
from flask import jsonify, render_template, request, url_for, redirect, session

import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch import nn
import torch
from torchvision import models


def predict(image_path):
    # setup
    CFG = {
        'device' : 'cpu',
        'model_name' : 'resnet34',
        }
    
    # predictive classes definition
    labels = {
        'Bacterial Leaf Blight Disease': 0, 'Bacterial Leaf Streak Disease': 1,
        'Bacterial Panicle Blight Disease': 2, 'Blast Disease': 3, 'Brown Spot Disease': 4,
        'Dead Heart Disease': 5, 'Downy Mildew Disease': 6, 'Hispa Disease': 7,
        'Normal Disease': 8, 'Tungro Disease': 9 
        }
    
    num_classes = len(labels.keys())
    reverse_labels = dict((v, k) for k, v in labels.items())
    
    # model preparation
    class CustomModel(nn.Module):
        def __init__(self, num_classes, model_name, pretrained=True):
            super(CustomModel, self).__init__()
            if model_name == 'efficientnet_b1':
                self.model = models.efficientnet_b1(pretrained=pretrained)
                in_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
            elif model_name == 'resnet34':
                self.model = models.resnet34(pretrained=pretrained)
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    
        def forward(self, x):
            x = self.model(x)
            return x
    
    
    # model definition
    model_path = "model/ORBIT-best-model-CV.pt"
    model = CustomModel(num_classes=num_classes, model_name=CFG['model_name'], pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(CFG['device'])
    model.eval()
    
    # image transformer
    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    print(image_path)
    image_pil = Image.open(image_path)
    image_np = numpy.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # img = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
    img = val_transforms(image=img)['image'] # apply same transforms of validation set
    img = img[None, ...].to(CFG['device']) # add batch dimension to image and use device
    # predict 
    pred_prob = model(img) 
    pred = torch.max(pred_prob, dim=1)[1]
    label = reverse_labels[pred.item()]
    skor = round(pred_prob.detach().cpu().numpy()[0][pred] * 1, 4)
    skor = 'skor tensor: ' + str(skor)

    return label, skor


        
