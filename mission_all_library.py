from time import sleep
from e_drone.drone import *
from e_drone.protocol import *
# import matplotlib.pyplot as plt
import math


drone = Drone()
drone.open()

class library:    

    def landing(self):
        print("미션완료 착륙합니다.")
        drone.sendLanding()
        sleep(0.01)
        drone.close()

    def eventAltitude(self,altitude):
        global height
        print(f'현재고도:{altitude.rangeHeight}m')
        height = altitude.rangeHeight
        return height
        
    def handler(self):
        drone.setEventHandler(DataType.Altitude, self.eventAltitude)
        drone.sendRequest(DeviceType.Drone, DataType.Altitude)
        sleep(0.1)
        # 짧으면 짧을수록 좋음. 시간 많이 X

    def stright_flying(self,positionX, m_s, delay_time):

        drone.sendControlPosition(positionX, 0, 0, m_s, 0,0)
        print(f"전방 {positionX}m, {m_s}m/s 속도로 이동합니다.")
        sleep(delay_time)


    def altitude_control(self,Hope_Altitude, m_s, delay_time):
        # 단위 : m/s
        self.handler()  # 고도 차이 알아내기
        altitude = []
        # 다시 실행되면 altitude를 []로 리셋하기때무에 [0]값을 받아도 상관x
        altitude.append(height)

        diffrence = Hope_Altitude-altitude[0]
        print(f'{Hope_Altitude}m까지 {diffrence}m 차이가 나므로 고도 조종합니다.')
        
        drone.sendControlPosition(0, 0, diffrence, m_s, 0, 0)
        sleep(delay_time)
        self.handler() # 고도 확인작업
        # Hope_Altitude -> 원하는고도 (m단위)
        # append로 한 이유는 변수가 자꾸 튕겨나가서
        print("고도조정완료 5초간 정지비행 합니다.")
        sleep(5)


    def square_flying(self,positionX, m_s,delay_time):    
        # 정사각형 회전 
        for i in range(4):
            print(f"go front start_{i+1}회")
            drone.sendControlPosition(positionX,0,0,m_s,0,0)
            sleep(delay_time)
            ###### 아직 0.5m이동으로 해놓았음 테스트하려고 ㅎㅎ...
            # 대회 제출떈 이동거리 1m로 바꾸고 0.5m/s로 바꿀것.

            # 좌로 90도 rotate
            print(f"rotate start_{i+1}회")
            drone.sendControlPosition(0,0,0,0,90,30)
            sleep(4)
            # 3초만에 돌고 1초 대기.
        print("정사각형 비행완료 5초간 정지비행 합니다.")
        sleep(5)
        # 정지비행 5초 포함

    def circle_flying(self, radius ,m_s, delay_time, eight_shape_rotate=False):
        x = [] 
        y = [] 
        
        # 지름이 1이라했으니 0.5 로 고쳐야함
        if eight_shape_rotate:
            for theta in range(0, 180):
                x.append(radius * math.cos(math.radians(theta)))   
                y.append(radius * math.sin(math.radians(theta)))         

            for theta in range(0, 360):
                x.append(-2*radius + radius * math.cos(math.radians(theta)))   
                y.append(-radius * math.sin(math.radians(theta))) 

            for theta in range(180, 360):
                x.append(radius * math.cos(math.radians(theta)))   
                y.append(radius * math.sin(math.radians(theta))) 

            for theta in range(0, 720):
                print(f'x좌표:{x[theta]}')
                print(f'y좌표:{y[theta]}')
                drone.sendControlPosition(x[theta],y[theta],0,m_s,0,0)
                sleep(delay_time)
            # plt.scatter(x, y)
            # plt.show()
            print("8자 비행완료 5초간 정지비행 합니다.")

        else:
            for theta in range(0, 360):
                # 0도에서 360도까지 1도(degree) 간격
                x.append(radius * math.cos(math.radians(theta)))   
                y.append(radius * math.sin(math.radians(theta))) 

                print(f'x좌표:{x[theta]}')
                print(f'y좌표:{y[theta]}')
                drone.sendControlPosition(x[theta],y[theta],0,m_s,0,0)
                sleep(delay_time)
            print("원 비행완료 5초간 정지비행 합니다.")
        sleep(5)
        # 정지비행 5초 포함

    def zigzag_flying(self, distance, m_s, angle, delay_time):
        
        angle = angle / 2

        positionX = distance * math.sin(math.radians(angle))
        positionY = distance * math.cos(math.radians(angle))

        for i in range(2):
            drone.sendControlPosition(positionX, positionY, 0, m_s, 0, 0)
            sleep(delay_time)
            drone.sendControlPosition(positionX, -positionY, 0, m_s, 0, 0)
            sleep(delay_time)
        for i in range(2):
            drone.sendControlPosition(positionY, -positionX, 0, m_s, 0, 0)
            sleep(delay_time)
            drone.sendControlPosition(-positionY, -positionX , 0, m_s, 0, 0)
            sleep(delay_time)
        for i in range(2):
            drone.sendControlPosition(-positionX, -positionY, 0, m_s, 0, 0)
            sleep(delay_time)
            drone.sendControlPosition(-positionX, positionY, 0, m_s, 0, 0)
            sleep(delay_time)
        for i in range(2):
            drone.sendControlPosition(-positionY, positionX, 0, m_s, 0, 0)
            sleep(delay_time)
            drone.sendControlPosition(positionY, positionX, 0, m_s, 0, 0)
            sleep(delay_time)
        print("지그재그 비행완료 5초간 정지비행합니다.")    
        sleep(5)
    # 정지비행 5s


    def same_flying(self):
        print("이륙")
        drone.sendTakeOff()
        sleep(3)   # waiting for takeoff

        print("호버링 3초")
        drone.sendControlWhile(0, 0, 0, 0, 3000)
        sleep(3)
        self.stright_flying(0.8,0.4,3)

        self.altitude_control(1.5,0.4,4)


