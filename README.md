# YoloV5 DeepSORT ROS

## Introduction
본 프로젝트는 ROS, yolov5를 활용하여 객체를 탐지해내고, Deepsort 알고리즘을 활용하여 단일 객체를 분류 및 추적하는 기능을 ROS 패키지로 구현하였습니다.
본 프로젝트의 원 소스코드는 아래와 같습니다.

* [mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git)
* [fcakyon/yolov5-pip](https://github.com/fcakyon/yolov5-pip.git)

## Tutorials
* [YoloV5로 원하는 Custom Dataset 만들기(외부링크)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Deepsort 관련 설명자료(외부링크)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)

## Before You Run Node
### 1. Configuring the Python3 environment for ROS Melodic
* YoloV5는 Python3 환경에서 구동됩니다. 그러므로 해당 패키지를 실행하기 전 ROS Medloic 환경에서 Python3 Node가 구동되도록 환경설정을 해주어야 합니다.
    ```s
    $ sudo apt-get install python3-pip python3-all-dev python3-yaml python3-rospkg
    $ sudo apt install ros-melodic-desktop-full --fix-missing
    $ sudo pip3 install rospkg catkin_pkg
    ```
* 