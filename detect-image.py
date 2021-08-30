# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 04:21:31 2021

@author: ASDF
"""

  
import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import model
from encoder import DataEncoder
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw


print('Loading model..')
net = model()
net.load_state_dict(torch.load('D:/Videos/pytorch-retinanet-master/csv_retinanet_99.pt'))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open('D:/Videos/clean/video-frame00001.jpg')
w = h = 600
img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()