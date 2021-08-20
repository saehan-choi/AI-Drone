import torch
import torch.nn as nn
import cv2
import numpy as np
from mission_all_library import *

class my_convnet(nn.Module):
    def __init__(self,  init_weights: bool = True):
        super(my_convnet, self).__init__()

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3x64 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv64x64 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)           
        )
        self.conv64x128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)           
        )
        self.conv128x128 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)            
        )
        self.conv128x256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)            
        )
        self.conv256x256 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)           
        )        
        self.conv256x512 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)         
        )
        self.conv512x512 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)           
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
convnet = my_convnet()
convnet = convnet.to(device)
PATH='./weight_file/' 
convnet.load_state_dict(torch.load(PATH+'model.pt'))
classes =  ('altitude_descent', 'altitude_rise', 'circular_rotation', 'landing', 'stop', 'turn_left', 'turn_right', 'u_turn')

def cnt_reset():
    global altitude_descent_cnt, altitude_rise_cnt, circular_rotation_cnt, landing_cnt,stop_cnt, turn_left_cnt, turn_right_cnt, u_trun_cnt
    altitude_descent_cnt, altitude_rise_cnt, circular_rotation_cnt, landing_cnt = 0, 0, 0, 0 
    stop_cnt, turn_left_cnt, turn_right_cnt, u_trun_cnt = 0, 0, 0, 0

cnt_reset()
library = library()
coordinate1=(50,30)
coordinate2=(400,30)
font=cv2.FONT_HERSHEY_SIMPLEX
signal_alarm = cv2.imread(PATH+"signal_alarm.jpg")

with torch.no_grad():
    cap = cv2.VideoCapture(0)
    library.same_flying(mission3=True)

    while True:
        _, img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        dst = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        dst = np.transpose(dst, (2, 0, 1))
        tensor = torch.Tensor(dst)
        tensor = tensor.unsqueeze(0)
        images = tensor.cuda()
        outputs = convnet(images)
        _, predicted = torch.max(outputs, 1)
        text = classes[predicted.item()]
        
        if text == "stop":
            cnt_reset()
            cv2.putText(img, "COUNT RESET", coordinate2, font, 1, (255,0,0) ,2)
            
        elif text == "altitude_rise":
            altitude_rise_cnt += 1
            cv2.putText(img, str(altitude_rise_cnt+1), coordinate2, font, 1, (255,0,0), 2)
            if altitude_rise_cnt == 99:
                cv2.imshow("signal", signal_alarm)
            elif altitude_rise_cnt == 100:
                library.altitude_control(2,0.6,3)
                cv2.destroyAllWindows()

        elif text == "altitude_descent":
            altitude_descent_cnt += 1
            cv2.putText(img, str(altitude_descent_cnt+1), coordinate2, font, 1, (255,0,0), 2)
            if altitude_descent_cnt == 99:
                cv2.imshow("signal", signal_alarm)
            elif altitude_descent_cnt == 100:
                library.altitude_control(1,0.6,3)
                cv2.destroyAllWindows()

        elif text == "turn_left":
            turn_left_cnt += 1
            cv2.putText(img, str(turn_left_cnt+1), coordinate2, font, 1, (255,0,0), 2)
            if turn_left_cnt == 99:
                cv2.imshow("signal", signal_alarm)
            elif turn_left_cnt == 100:
                library.stright_flying(1.2,0.5,3)
                library.stright_flying(0.8,0.4,3,right_left=True)
                print('좌회전 비행완료 5초간 정지비행합니다.')
                sleep(5)
                print("제자리로 회귀합니다.")                
                drone.sendControlPosition(-1.2,-0.8,0,0.5,0,0)
                sleep(4)
                cv2.destroyAllWindows()

        elif text == "turn_right":
            turn_right_cnt += 1
            cv2.putText(img, str(turn_right_cnt+1), coordinate2, font, 1, (255,0,0), 2)
            if turn_right_cnt == 99:
                cv2.imshow("signal", signal_alarm)
            elif turn_right_cnt == 100:
                library.stright_flying(1.2,0.5,3)
                library.stright_flying(-0.8,0.4,3,right_left=True)
                print('우회전 비행완료 5초간 정지비행합니다.')
                sleep(5)
                print("제자리로 회귀합니다.")
                drone.sendControlPosition(-1.2,0.8,0,0.5,0,0)
                sleep(4)
                cv2.destroyAllWindows()

        elif text == "u_turn":
            u_trun_cnt += 1
            cv2.putText(img, str(u_trun_cnt+1), coordinate2, font, 1, (255,0,0), 2)
            if u_trun_cnt == 99:
                cv2.imshow("signal", signal_alarm)
            elif u_trun_cnt == 100:
                library.stright_flying(1,0.5,3)
                library.circle_flying(0.4,0.4,0.4,u_turn=True)
                library.stright_flying(-1,0.5,3)
                print("u_turn 비행완료 5초간 정지비행합니다.")
                sleep(5)
                print("제자리로 회귀합니다.")
                library.stright_flying(-0.8,0.3,4,right_left=True)
                # 이거 오른쪽 왼쪽인지 잘봐라 그리고 -0.6 등 fine_tuning 해라.
                cv2.destroyAllWindows()

        elif text == "circular_rotation":
            circular_rotation_cnt += 1
            cv2.putText(img, str(circular_rotation_cnt+1), coordinate2, font, 1, (255,0,0), 2)
            if circular_rotation_cnt ==99:
                cv2.imshow("signal", signal_alarm)
            elif circular_rotation_cnt == 100:
                library.circle_flying(1.75,0.7,0.3)
                cv2.destroyAllWindows()
    
        elif text == "landing":
            landing_cnt += 1
            cv2.putText(img, str(landing_cnt+1), coordinate2, font, 1, (255,0,0), 2)
            if landing_cnt == 99:
                cv2.imshow("signal", signal_alarm)
            elif landing_cnt == 100:
                library.landing()
                
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.putText(img, text, coordinate1, font, 1, (255,0,0), 2)
        cv2.imshow('MISSION_3', img)

        k = cv2.waitKey(1)
        if k == 27 or landing_cnt == 100:
            # ascii code 27 is ESC button or landing_cnt
            cv2.destroyAllWindows()
            break
