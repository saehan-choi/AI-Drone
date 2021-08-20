# AI-Drone
Codrone2 DIY

# version

in mission 1,2

    only pip install e_drone
  
else mission 3

    cuda10.1
    torch1.7.1
    opencv4.5.1
    numpy1.19.5

# mission 1
이륙 ⇨ 호버링(3초) ⇨ 전진 비행(80cm) ⇨ 고도 상승(높이 1.5m) ⇨ 정지 비행(5sec) ⇨ 
정사각형 패턴 비행(각 동작마다 드론 90°회전, 정지 1sec, 지름 1m) ⇨ 정지 비행(5sec) ⇨ 
원 패턴 비행(지름 1m) ⇨ 정지 비행(5sec) ⇨ 후진 비행(1m) ⇨ 고도 하강(높이 1m) ⇨ 
정지 비행(5sec) ⇨ 착륙

# mission 2
이륙 ⇨ 호버링(3초) ⇨ 전진 비행(80cm) ⇨ 고도 상승(높이 1.5m) ⇨ 정지 비행(5sec) ⇨ 
8자 원 비행(각 지름 1m) ⇨ 정지 비행(5sec) ⇨ 지그재그 비행 4회(45°시계방향 0.5sec, 전진 1.5sec) 
⇨ 정지 비행(5sec) ⇨ 전진 비행(60cm) ⇨ 고도 하강(높이 50cm) ⇨ 정지 비행(5sec) ⇨ 
고도 상승(높이 1m) ⇨ 전진 비행(50cm) ⇨정지 비행(5sec) ⇨ 착륙

# mission 3
웹캠으로 들어오는 이미지 분류 -> 맞는 동작수행

![image](https://user-images.githubusercontent.com/70372577/130180265-9acd4882-3ac5-4332-9d07-a6a3b509a9db.png)
![image](https://user-images.githubusercontent.com/70372577/130180274-24263cd8-2c88-4ef5-8c02-0e97562a03a8.png)
![image](https://user-images.githubusercontent.com/70372577/130180318-7921f4e4-475c-4226-9115-6118142aee2f.png)
![image](https://user-images.githubusercontent.com/70372577/130180444-6a996d72-f5bd-4c3f-8829-92a2f5899668.png)
![image](https://user-images.githubusercontent.com/70372577/130180476-c15917d4-b9ab-4782-bf89-b700caed43cc.png)
![image](https://user-images.githubusercontent.com/70372577/130180538-4b3e946a-c86e-460f-b808-e3db920157a7.png)
![image](https://user-images.githubusercontent.com/70372577/130180492-2c75deb8-0672-47be-9f19-29658aa60d83.png)
![image](https://user-images.githubusercontent.com/70372577/130180498-062265fd-6b5f-479e-88af-7747b71ef43a.png)






영상 : https://www.youtube.com/watch?v=Sy6B7mN5g4s
