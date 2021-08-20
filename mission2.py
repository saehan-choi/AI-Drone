from mission_all_library import *


# mission 1

if __name__ =="__main__":

    library = library()

    library.same_flying()
    # mission 1, 2 동일요소
    # 고도 못받아오는거 배터리 부족하면 못받아오는거같음

    library.circle_flying(0.4, 0.1, eight_shape_rotate=True)
    # circle_flying(radius, m/s, delay_time, eight_shape_rotate=True):
    # 8자비행 신호 ON

    library.zigzag_flying(0.25, 0.15, 45, 1.5)
    # zigzag_flying(distance, m/s, angle, delay_time)

    library.stright_flying(0.6, 0.3, 3)
    # stright_flying(positionX, m/s, delay_time)

    library.altitude_control(0.5, 0.3, 3)
    # altitude_control(Hope_Altitude, m/s, delay_time)

    library.altitude_control(1,0.4,3)
    # altitude_control(Hope_Altitude, m/s, delay_time)

    library.stright_flying(0.5,0.25,3)
    # stright_flying(positionX, m/s, delay_time)
    
    library.landing()