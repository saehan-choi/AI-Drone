from mission_all_library import *


# mission 2
if __name__ =="__main__":

    library = library()

    library.same_flying()
    # mission 1, 2 동일요소 이륙, 호버링, 전진비행, 고도상승포함

    library.circle_flying(0.5, 0.5, 0.3, eight_shape_rotate=True)
    # circle_flying(radius, m/s, delay_time, 8자비행신호ON)

    library.zigzag_flying(0.5, 0.35, 45, 1.5)
    # zigzag_flying(distance, m/s, angle, delay_time)

    library.stright_flying(0.6, 0.3, 3)
    # stright_flying(distance, m/s, delay_time)

    library.altitude_control(0.5, 0.4, 4)
    # altitude_control(Hope_Altitude, m/s, delay_time)

    library.altitude_control(1,0.4,3,sleepoff=True)
    # altitude_control(Hope_Altitude, m/s, delay_time, 정지비행OFF)

    library.stright_flying(0.5,0.25,3,sleepon=True)
    # stright_flying(distance, m/s, delay_time, 정지비행ON)
    
    library.landing()