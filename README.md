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
* python3환경에서 cv_bridge를 사용하는 경우 에러가 발생합니다. 그러므로 cv_bridge를 melodic version에 맞춰 재 빌드를 해주어야 합니다.
    ```s
    $ sudo apt-get install python-catkin-tools python3-catkin-pkg-modules
    # Create catkin workspace
    $ mkdir catkin_workspace
    $ cd catkin_workspace
    $ catkin init
    
    # Instruct catkin to set cmake variables
    $ catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
    
    # Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
    $ catkin config --install
    
    # Clone cv_bridge src
    $ git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
    
    # Find version of cv_bridge in your repository
    $ apt-cache show ros-melodic-cv-bridge | grep Version
        Version: 1.13.0-0bionic.20210505.032238
    
    # Checkout right version in git repo. In our case it is 1.13.0
    $ cd src/vision_opencv/
    $ git checkout 1.13.0
    $ cd ../../
    
    # Build
    $ catkin build cv_bridge
    
    # Extend environment with new package
    $ echo "source install/setup.bash --extend" >> bashrc
    ```