from time import sleep
from e_drone.drone import *
from e_drone.protocol import *
import math

drone = Drone()
drone.open()

class library:    
    
    def eventAltitude(self,altitude):
        global height
        print(f'현재고도:{altitude.rangeHeight}m')
        height = altitude.rangeHeight
        return height
        
    def handler(self):
        drone.setEventHandler(DataType.Altitude, self.eventAltitude)
        drone.sendRequest(DeviceType.Drone, DataType.Altitude)
        sleep(0.1)

    def same_flying(self,mission3=False):
        if mission3:
            print("이륙")
            drone.sendTakeOff()
            sleep(3)
            drone.sendControlWhile(0, 0, 0, 0, 1000)
            sleep(1)
        else:
            print("이륙")
            drone.sendTakeOff()
            sleep(3)
            print("호버링 3초")
            drone.sendControlWhile(0, 0, 0, 0, 3000)
            sleep(3)
            self.stright_flying(0.8,0.4,3)
            self.altitude_control(1.5,0.4,4)

    def stright_flying(self,distance, m_s, delay_time, right_left=False, sleepON=False):
        if right_left:
            drone.sendControlPosition(0, distance, 0, m_s, 0,0)
            print(f"측방 {distance}m, {m_s}m/s 속도로 이동합니다.")
        else:
            drone.sendControlPosition(distance, 0, 0, m_s, 0,0)
            print(f"전방 {distance}m, {m_s}m/s 속도로 이동합니다.")
        sleep(delay_time)
        if sleepON == True:
            print("5초간 정지비행합니다.")
            sleep(5)

    def altitude_control(self,Hope_Altitude, m_s, delay_time, sleepoff=False):
        self.handler()
        altitude = []
        altitude.append(height)
        diffrence = Hope_Altitude-altitude[0]
        print(f'{Hope_Altitude}m까지 {diffrence}m 차이가 나므로 고도 조종합니다.')
        drone.sendControlPosition(0, 0, diffrence, m_s, 0, 0)
        sleep(delay_time)
        self.handler()
        if sleepoff==True:
            pass
        else:
            print("고도조정완료 5초간 정지비행 합니다.")
            sleep(5)

    def square_flying(self,distance, m_s,delay_time):    
        for i in range(4):
            print(f"정사각형비행_전진_{i+1}회")
            drone.sendControlPosition(distance,0,0,m_s,0,0)
            sleep(delay_time)
            print(f"정사각형비행_회전__{i+1}회")
            drone.sendControlPosition(0,0,0,0,90,30)
            sleep(4)
        print("정사각형 비행완료 5초간 정지비행 합니다.")
        sleep(5)

    def circle_flying(self, radius ,m_s, delay_time, eight_shape_rotate=False, u_turn=False):
        x = [] 
        y = [] 
        x_sampling=[]
        y_sampling=[]

        if eight_shape_rotate:
            for theta in range(0, 360):
                x.append(-radius * math.cos(math.radians(theta)))   
                y.append(radius * math.sin(math.radians(theta)))         
            for theta in range(0, 361):
                x.append(-2*radius + radius * math.cos(math.radians(theta)))   
                y.append(radius * math.sin(math.radians(theta)))         
            for theta in range(0, 721):
                if theta == 0:
                    pass
                elif theta % 10 == 0:
                    drone.sendControlPosition(x[theta]-x[theta-10],y[theta]-y[theta-10],0,m_s,0,0)
                    sleep(delay_time)
            print("8자 비행완료 5초간 정지비행 합니다.")
            sleep(5)

        elif u_turn:
            for theta in range(0,181):
                x.append(radius - radius * math.cos(math.radians(theta)))   
                y.append(radius * math.sin(math.radians(theta))) 
            for theta in range(0,181):
                if theta == 0:
                    pass
                elif theta % 10 == 0:
                    drone.sendControlPosition(y[theta]-y[theta-10],x[theta]-x[theta-10],0,m_s,0,0)
                    sleep(delay_time)
            
        else:
            for theta in range(0, 361):
                x.append(radius - radius * math.cos(math.radians(theta)))   
                y.append(radius * math.sin(math.radians(theta))) 
            for theta in range(0,361):
                if theta == 0:
                    pass
                elif theta % 10 ==0:
                    drone.sendControlPosition(y[theta]-y[theta-10],x[theta]-x[theta-10],0,m_s,0,0)
                    sleep(delay_time)
            print("원 비행완료 5초간 정지비행 합니다.")
            sleep(5)

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

    def landing(self):
        print("미션완료 착륙합니다.")
        drone.sendLanding()
        sleep(0.1)
        drone.close()    