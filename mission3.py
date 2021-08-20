import torch
# import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
# import torch.optim as optim  이거두개 안쓰는듯ㅋㅋ
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
import time
import numpy as np

from mission_all_library import *

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# trainset = ImageFolder("./train_dataset/",
#                          transform=transform)
# trainloader = data.DataLoader(trainset, batch_size=4 , shuffle=True)


# testset = ImageFolder("./test_dataset/",
#                          transform=transform)
# testloader = data.DataLoader(testset, batch_size=4, shuffle=True)


class VGG_E(nn.Module):
    def __init__(self,  init_weights: bool = True):
        super(VGG_E, self).__init__()

        
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3x64 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv64x64 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
        )
        self.conv64x128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
        )
        self.conv128x128 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
        )
        self.conv128x256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
        )
        self.conv256x256 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
        )        
        self.conv256x512 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
        )
        self.conv512x512 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
        )
        self.convnet = nn.Sequential(
            self.conv3x64,
            self.conv64x64,
            self.conv64x64,               
            self.conv64x128,
            self.maxpool, # 224 -> 112 
            self.conv128x128,
            self.conv128x128,
            self.conv128x256,
            self.maxpool, # 112 -> 56
            self.conv256x256,
            self.conv256x256,
            self.conv256x512,
            self.maxpool, # 56 -> 28
            self.conv512x512,
            self.conv512x512,
            self.maxpool # 28 -> 14
        )

        self.fclayer = nn.Sequential(
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 8),
        )

    def forward(self, x:torch.Tensor):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        return x





device = torch.device('cuda')
vgg11 = VGG_E()
vgg11 = vgg11.to(device)
PATH='./resnet_weight/' 
vgg11.load_state_dict(torch.load(PATH+'model.pt'))
classes =  ('altitude_descent', 'altitude_rise', 'circular_rotation', 'landing', 'stop', 'turn_left', 'turn_right', 'u_trun')

def cnt_reset():
    global altitude_descent_cnt, altitude_rise_cnt, circular_rotation_cnt, landing_cnt,stop_cnt, turn_left_cnt, turn_right_cnt, u_trun_cnt
    altitude_descent_cnt, altitude_rise_cnt, circular_rotation_cnt, landing_cnt = 0, 0, 0, 0 
    stop_cnt, turn_left_cnt, turn_right_cnt, u_trun_cnt = 0, 0, 0, 0

# reset 안되면 다시 

cnt_reset()

coordinate1=(100,100)
coordinate2=(150,100)

font=cv2.FONT_HERSHEY_SIMPLEX
with torch.no_grad():
    cap = cv2.VideoCapture(0)
    while True:
        # print(f'image type: {type(cap)}')
        
        _, img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        dst = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        dst = np.transpose(dst, (2, 0, 1))
        tensor = torch.Tensor(dst)
        tensor = tensor.unsqueeze(0)    
        images = tensor.cuda()
        outputs = vgg11(images)
        _, predicted = torch.max(outputs, 1)

        print(f'predicted is {classes[predicted.item()]}')
        text = classes[predicted.item()]
        
        if text == "stop":
            cnt_reset()
            cv2.putText(img, stop_cnt, coordinate2, font, 1, (255,0,0) ,2)

        elif text == "altitude_rise":
            altitude_rise_cnt += 1
            cv2.putText(img, altitude_rise_cnt, coordinate2, font, 1, (255,0,0), 2)
            if altitude_rise_cnt > 300:
                # library.altitude_rise( )
                pass

        elif text == "altitude_descent":
            altitude_descent_cnt += 1
            cv2.putText(img, altitude_descent_cnt, coordinate2, font, 1, (255,0,0), 2)
            if altitude_descent_cnt > 300:
                # library.altitude_descent( )
                pass


        elif text == "trun_left":
            turn_left_cnt += 1
            cv2.putText(img, turn_left_cnt, coordinate2, font, 1, (255,0,0), 2)
            if turn_left_cnt > 300:
                # library.turn_left( )
                pass

        elif text == "trun_right":
            turn_right_cnt += 1
            cv2.putText(img, turn_right_cnt, coordinate2, font, 1, (255,0,0), 2)
            if turn_right_cnt > 300:
                # library.turn_right( )
                pass

        elif text == "u_turn":
            u_trun_cnt += 1
            cv2.putText(img, u_trun_cnt, coordinate2, font, 1, (255,0,0), 2)
            if u_trun_cnt > 300:
                # library.altitude_descent( )
                pass

        elif text == "circular_rotation":
            circular_rotation_cnt += 1
            cv2.putText(img, circular_rotation_cnt, coordinate2, font, 1, (255,0,0), 2)
            if circular_rotation_cnt > 300:
                # library.altitude_descent( )
                pass

        elif text == "landing":
            landing_cnt += 1
            cv2.putText(img, landing_cnt, coordinate2, font, 1, (255,0,0), 2)
            if landing_cnt > 300:
                # library.altitude_descent( )
                pass

        # time.sleep(0.1)
        
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.putText(img, text, coordinate1, font, 1, (255,0,0), 2)
        cv2.imshow('test.jpg', img)

        k = cv2.waitKey(1)
        if k == 27:
            # ascii code 27 is ESC button
            break
