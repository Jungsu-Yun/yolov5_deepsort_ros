# YoloV5 DeepSORT Pytorch ROS
<img src="./doc/track_all.gif" width="400"/> <img src="./doc/track_pedestrians.gif" width="400"/>

## Introduction
This repository implements ROS packages to detect objects using ROS, yolov5, and to classify and track single objects using Deepport algorithms.
The source code of this repository is as follows.

* [mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

This repository is described in Korean and English, and please refer to the following link.
* [한국어](./doc/README_KOR.md)
* [English](/README.md)

## Tutorials
* [Train Custom Data(External Link)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Training the RE-ID model(External Link)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)

## Before You Run Node
### 1. Configuring the Python3 environment for ROS Melodic
* YoloV5 runs in Python 3 environments. Therefore, before running the package, you must configure the node using Python 3 to run in the ROS Medloic environment.
    ```s
    sudo apt-get install python3-pip python3-all-dev python3-yaml python3-rospkg
    sudo apt-get install ros-melodic-desktop-full --fix-missing
    sudo pip3 install rospkg catkin_pkg
    ```
* Error occurs when using cv_bridge in python3 environment. Therefore, you should rebuild the CV_bridge according to the melodic version.
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
* Run clone with a repository that is dependent.
    ```
    cd your_workspace/src
    git clone --recurse-submodules https://github.com/jungsuyun/yolov5_deepsort_ros.git
    cd ..
    catkin_make
    ```
    If you did not do '--recurse-submodules', you must run 'git submodule update --init'.
* Ensure that all dependency information in Node is met. The package works with Python 3.6 or later and requires several dependency packages to be installed.
    ```
    pip3 install "numpy>=1.18.5,<1.20" "matplotlib>=3.2.2,<4"
    pip3 install yolov5
    pip3 install -r requirements.txt
    ```
* Github [prevents you from uploading more than 100MB of files](https://docs.github.com/en/github/managing-large-files/working-with-large-files/conditions-for-large-files). Therefore, you should download the weight file associated with Deepport.
* [Download the Depsort-specific weight file](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6). Download the ckpt.t7 file to the path 'scripts/deep_sort_pytorch/deep_sort/deep/checkpoint/'.

## Run YoloV5 Deepsort Node
### 1. Run Detection Node
* The yolov5 detection node runs as follows:

    ```
    roslaunch yolov5_deepsort detector.launch
    ```
    
    By default, the image topic that performs detection is '/image_raw'. If you want to change the Subscribe image topic, you need to modify the following part of 'detector.launch'.

    ```
    <arg name="image_topic"	                default="/image_raw"/>
    ```
    If you want to detect only one or a few objects, not multiple objects, you need to modify the following parts of 'detector.launch'. If you want to detect the entire class, you must enter None, and if you want to detect only a specific class, you must enter the class name.
    ```
    <arg name="class_name"                  default='None'/>
    <!-- <arg name="class_name"                  default='person'/> -->
    ```

### 2. Run Deep Sort Node
* yolov5 depthort node runs as follows:

    ```
    roslaunch yolov5_deepsort tracker.launch
    ```
    
    By default, the image topic that performs detection is '/image_raw'. If you want to change the Subscribe image topic, you need to modify the following part of 'detector.launch'.

    ```
    <arg name="image_topic"	                default="/image_raw"/>
    ```
    The initial value of the class that performs tracking is 'person'. If you want to track another class, you need to modify the next part of 'detector.launch'.
    ```
    <arg name="class_name"                  default='person'/>
    ```

### 3. Subscribe Topic
* [/image_raw](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html) : Subscribe Image topic to detect objects.

### 4. Published Topic
* [/detections_image_topic](https://github.com/jungsuyun/yolov5_deepsort_ros/blob/melodic/msg/BoundingBox.msg)
    * string Class
    * float64 probability
    * int64 xmin
    * int64 ymin
    * int64 xmax
    * int64 ymax
* [/detections_image_topic](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html) : Issue Image topic with bounding box, class name entered.

## TroubleShooting
#### ImportError: dynamic module does not define module export function (PyInit_cv_bridge_boost)
This error occurs when a python package related to cv_bridge_boost is not found. Open the cv_bridge/CMakelist.txt file in the folder where you installed the cv_bridge package and modify 'found_package(Boost REQUIRED python3)' as follows:
```CMake
find_package(Boost REQUIRED python3-py36)
```
Then build and run cv_bridge again.

#### ImportError: cannot import name 'BoundingBox'
This error is caused by not building yolov5_deepport package. Go to catkin_ws, 'catkin_make' and run the package again.