# YoloV5 DeepSORT Pytorch ROS
<img src="./track_all.gif" width="400"/> <img src="./track_pedestrians.gif" width="400"/>

## Introduction
본 레포지토리는 ROS, yolov5를 활용하여 객체를 탐지해내고, Deepsort 알고리즘을 활용하여 단일 객체를 분류 및 추적하는 기능을 ROS 패키지로 구현하였습니다.
본 프로젝트의 원 소스코드는 아래와 같습니다.

* [mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

본 리포지토리는 한국어, 영어로 설명되어 있으며 다음의 링크를 참고하시기 바랍니다.
* [한국어](/doc/README_KOR.md)
* [영어](/README.md)


## Tutorials
* [YoloV5로 원하는 Custom Dataset 만들기(외부링크)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Deepsort 관련 설명자료(외부링크)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)

## Before You Run Node
### 1. Configuring the Python3 environment for ROS Melodic
* YoloV5는 Python3 환경에서 구동됩니다. 그러므로 해당 패키지를 실행하기 전 ROS Medloic 환경에서 Python3 Node가 구동되도록 환경설정을 해주어야 합니다.
    ```s
    sudo apt-get install python3-pip python3-all-dev python3-yaml python3-rospkg
    sudo apt-get install ros-melodic-desktop-full --fix-missing
    sudo pip3 install rospkg catkin_pkg
    ```
* python3환경에서 cv_bridge를 사용하는 경우 에러가 발생합니다. 그러므로 cv_bridge를 melodic version에 맞춰 재 빌드를 해주어야 합니다.
    ```s
    sudo apt-get install python-catkin-tools python3-catkin-pkg-modules
    # Create catkin workspace
    mkdir catkin_workspace
    cd catkin_workspace
    catkin init
    
    # Instruct catkin to set cmake variables
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
    
    # Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
    catkin config --install
    
    # Clone cv_bridge src
    git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
    
    # Find version of cv_bridge in your repository
    apt-cache show ros-melodic-cv-bridge | grep Version
        Version: 1.13.0-0bionic.20210505.032238
    
    # Checkout right version in git repo. In our case it is 1.13.0
    cd src/vision_opencv/
    git checkout 1.13.0
    cd ../../
    
    # Build
    catkin build cv_bridge
    
    # Extend environment with new package
    echo "source install/setup.bash --extend" >> bashrc
    ```

### 2. ROS Node Package Download
* 의존성이 있는 리포지토리와 함께 clone을 실행합니다.
    ```
    cd your_workspace/src
    git clone --recurse-submodules https://github.com/jungsuyun/yolov5_deepsort_ros.git
    cd ..
    catkin_make
    ```
    만약 `--recurse-submodules`를 하지 않았다면 `git submodule update --init`을 실행하여야 합니다.
* Node의 모든 의존성 정보를 충족하는지 확인해야 한다. 해당 패키지는 python3.6 버전 이상에서 동작하며 몇가지 의존성 패키지를 설치해야 합니다.
    ```
    pip3 install "numpy>=1.18.5,<1.20" "matplotlib>=3.2.2,<4"
    pip3 install yolov5
    pip3 install -r requirements.txt
    ```
* Github는 [100MB 이상의 파일을 업로드 하는 것을 막고 있습니다](https://docs.github.com/en/github/managing-large-files/working-with-large-files/conditions-for-large-files). 그러므로 Deepsort와 관련한 가중치 파일을 다운로드 해야 합니다.
    * [Deepsort 관련 가중치 파일을 다운로드 합니다](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6). ckpt.t7 파일을 `scripts/deep_sort_pytorch/deep_sort/deep/checkpoint/` 경로에 다운받습니다.

## Run YoloV5 Deepsort Node
### 1. Run Detection Node
* yolov5 detection node는 다음과 같이 실행합니다.

    ```
    roslaunch yolov5_deepsort detector.launch
    ```
    
    기본적으로 detection을 수행하는 image topic은 `/image_raw` 입니다. 만약 Subscribe image topic을 변경하고 싶다면 `detector.launch`의 다음 부분을 수정해야 합니다.

    ```
    <arg name="image_topic"	                default="/image_raw"/>
    ```
    여러개의 객체가 아닌 한개 또는 일부의 객체만 탐지하고 싶다면 `detector.launch`의 다음 부분을 수정해야 합니다. 전체 Class를 탐지하고자 한다면 None을 입력하고 특정 Class만 탐지를 하기 위해서는 Class명을 입력해야 합니다.
    ```
    <arg name="class_name"                  default='None'/>
    <!-- <arg name="class_name"                  default='person'/> -->
    ```
* __실행 결과__

### 2. Run Deep Sort Node
* yolov5 deepsort node는 다음과 같이 실행합니다.

    ```
    roslaunch yolov5_deepsort tracker.launch
    ```
    
    기본적으로 detection을 수행하는 image topic은 `/image_raw` 입니다. 만약 Subscribe image topic을 변경하고 싶다면 `detector.launch`의 다음 부분을 수정해야 합니다.

    ```
    <arg name="image_topic"	                default="/image_raw"/>
    ```
    Tracking을 수행하는 Class의 초기값은 `person`입니다. 만약 다른 Class를 Tracking 하고자 한다면 `detector.launch`의 다음 부분을 수정해야 합니다.
    ```
    <arg name="class_name"                  default='person'/>
    ```

### 3. Subscribe Topic
* [/image_raw](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html) : 객체를 검출하기 위한 Image topic을 Subscribe 합니다.

### 4. Published Topic
* [/detections_image_topic](https://github.com/jungsuyun/yolov5_deepsort_ros/blob/melodic/msg/BoundingBox.msg)
    * string Class : 인식된 객체 클래스 및 분류된 값
    * float64 probability : 인식된 객체의 정확도
    * int64 xmin : Bounding Box의 x축 최소값
    * int64 ymin : Bounding Box의 y축 최소값
    * int64 xmax : Bounding Box의 x축 최대값
    * int64 ymax : Bounding Box의 y축 최대값
* [/detections_image_topic](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html) : Bounding Box, Class명이 입력된 Image topic을 발행합니다.

## TroubleShooting
#### ImportError: dynamic module does not define module export function (PyInit_cv_bridge_boost)
해당 에러는 cv_bridge_boost관련 python 패키지를 찾지 못했을 경우에 발생합니다. cv_bridge 패키지를 설치한 폴더의 cv_bridge/CMakelist.txt파일을 열어 `find_package(Boost REQUIRED python3)`을 다음과 같이 수정합니다.
```CMake
find_package(Boost REQUIRED python3-py36)
```
그리고 다시 cv_bridge를 빌드한 뒤 실행합니다.

#### ImportError: cannot import name 'BoundingBox'
해당 에러는 yolov5_deepsort package를 빌드하지 않아 발생한 문제입니다. catkin_ws로 이동하여 `catkin_make`를 한 후 다시 패키지를 실행합니다.