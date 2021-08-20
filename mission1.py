from mission_all_library import *


# mission 1
if __name__ =="__main__":

    library = library()

    library.same_flying()
    # mission 1, 2 동일요소 이륙, 호버링, 전진비행, 고도상승포함

    library.square_flying(1, 0.25, 5)
    # square_flying(distance, m/s, delay_time)

    library.circle_flying(0.5, 0.5, 0.3)
    # circle_flying(radius, m/s, delay_time)

    library.stright_flying(-1, 0.4, 3)
    # stright_flying(distance, m/s, delay_time)
    
    library.altitude_control(1, 0.4, 3)
    # altitude_control(Hope_Altitude, m/s, delay_time)

    library.landing()