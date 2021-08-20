from mission_all_library import *


# mission 1
if __name__ =="__main__":

    library = library()

    library.same_flying()
    # mission 1, 2 동일요소

    library.square_flying(0.5, 0.25, 3)
    # square_flying(positionX, m/s, delay_time)

    library.circle_flying(0.3, 0.1, 0.04)
    # circle_flying(radius, m/s, delay_time)
    # 이 신호는 for문안에서 360도를 돌아야하기 때문에 360*0.04 = 14.4s 소요

    library.stright_flying(-1, 0.4, 3)
    # stright_flying(positionX, m/s, delay_time)
    
    library.altitude_control(1, 0.4, 3)
    # altitude_control(Hope_Altitude, m/s, delay_time)

    library.landing()