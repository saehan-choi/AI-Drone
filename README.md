# AI-Drone
The product name is codrone2 DIY

you need to "pip install e_drone"

# version
pytorch - 1.7
opencv - 

# mission 1
이륙 ⇨ 호버링(3초) ⇨ 전진 비행(80cm) ⇨ 고도 상승(높이 1.5m) 
⇨ 정지 비행(5sec) ⇨ 정사각형 패턴 비행(각 동작마다 드론 
90°회전, 정지 1sec, 지름 1m) ⇨ 정지 비행(5sec) ⇨ 원 패턴 비
행(지름 1m) ⇨ 정지 비행(5sec) ⇨ 후진 비행(1m) ⇨ 고도 하강
(높이 1m) ⇨정지 비행(5sec) ⇨ 착륙

# mission 2
이륙 ⇨ 호버링(3초) ⇨ 전진 비행(80cm) ⇨ 고도 상승(높이 1.5m) 
⇨ 정지 비행(5sec) ⇨ 8자 원 비행(각 지름 1m) ⇨ 정지 비행
(5sec) ⇨ 지그재그 비행 4회(45°시계방향 0.5sec, 전진 1.5sec) 
⇨ 정지 비행(5sec) ⇨ 전진 비행(60cm) ⇨ 고도 하강(높이 50cm) 
⇨ 정지 비행(5sec) ⇨ 고도 상승(높이 1m) ⇨ 전진 비행(50cm) 
⇨정지 비행(5sec) ⇨ 착륙

# mission 3
웹캠으로 들어오는 이미지 분류 -> 맞는 동작수행

ex)좌회전, 우회전, 유턴, 원회전, 고도상승, 하강, 정지, 착륙

영상 : https://www.youtube.com/watch?v=Sy6B7mN5g4s
