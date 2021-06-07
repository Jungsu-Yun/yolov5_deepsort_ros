#!/usr/bin/env python3

from __future__ import division

import torch
from torch.autograd import Variable

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import (non_max_suppression, set_logging, yolov5_in_syspath)
from yolov5.utils.torch_utils import (load_classifier, select_device)

import rospy
from rospkg import RosPack
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from yolov5_deepsort.msg import BoundingBox, BoundingBoxes
from skimage.transform import resize

import os, cv2

import numpy as np

package = RosPack()
package_path = package.get_path('yolov5_deepsort')

class DetectorManager():
    def __init__(self):
        # Load weights parameter
        weights_name = rospy.get_param('~weights_name', 'yolov5/weights/yolov5s.pt')
        self.weights_path = os.path.join(package_path, 'scripts', weights_name)
        rospy.loginfo("Found weights, loading %s", self.weights_path)

        # Raise error if it cannot find the model
        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))

        # Load image parameter and confidence threshold
        self.image_topic = rospy.get_param('~image_topic', '/image_raw')
        self.confidence_th = rospy.get_param('~confidence', 0.25)
        self.nms_th = rospy.get_param('~nms_th', 0.45)

        # Load publisher topics
        self.detected_objects_topic = rospy.get_param('~detected_objects_topic')
        self.published_image_topic = rospy.get_param('~detections_image_topic')

        # Load other parameters
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.network_img_size = rospy.get_param('~img_size', 640)
        self.publish_image = rospy.get_param('~publish_image')

        # Initialize
        set_logging()
        device = select_device(str(self.gpu_id))
        self.half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights_name, map_location=device)  # load FP32 model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        stride = int(self.model.stride.max())  # model stride
        if self.half:
            self.model.half()  # to FP16

        # Load Class and initialize
        self.class_name = rospy.get_param('~class_name', None)
        # print(self.names)
        if self.class_name == "None":
            self.class_name = None
        elif type(self.class_name) == str:
            self.class_name = self.names.index(self.class_name)
        else:
            self.class_name = self.class_name.split(', ')
            self.class_name = [self.names.index(int(i)) for i in self.class_name]

        print(self.class_name)

        # Initialize width and height
        self.h = 0
        self.w = 0

        self.classes_colors = {}

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            with yolov5_in_syspath():
                self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                    device).eval()
        self.bridge = CvBridge()

        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24)

        # Define publishers
        self.pub_ = rospy.Publisher(self.detected_objects_topic, BoundingBoxes, queue_size=10)
        self.pub_viz_ = rospy.Publisher(self.published_image_topic, Image, queue_size=10)
        rospy.loginfo("Launched node for object detection")

        rospy.spin()

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header

        input_img = self.imagePreProcessing(self.cv_image)
        if torch.cuda.is_available():
            input_img = Variable(input_img.type(torch.cuda.HalfTensor))

        with torch.no_grad():
            # img = torch.from_numpy(input_img).to(self.gpu_id)
            # img = img.half() if self.half else img.float()  # uint8 to fp16/32
            # img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # if img.ndimension() == 3:
            #     img = img.unsqueeze(0)
            detections = self.model(input_img, augment=False)[0]
            detections = non_max_suppression(detections, self.confidence_th, self.nms_th, classes=self.class_name, agnostic=False)

        if detections[0] is not None:
            for detection in detections[0]:
                # # Get xmin, ymin, xmax, ymax, confidence and class
                # print(detection)
                xmin, ymin, xmax, ymax, conf, det_class = detection
                pad_x = max(self.h - self.w, 0) * (self.network_img_size/max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * (self.network_img_size/max(self.h, self.w))
                unpad_h = self.network_img_size-pad_y
                unpad_w = self.network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*self.w
                xmax_unpad = ((xmax-xmin)/unpad_w)*self.w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*self.h
                ymax_unpad = ((ymax-ymin)/unpad_h)*self.h + ymin_unpad

                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = int(xmin_unpad.item())
                detection_msg.xmax = int(xmax_unpad.item())
                detection_msg.ymin = int(ymin_unpad.item())
                detection_msg.ymax = int(ymax_unpad.item())
                detection_msg.probability = conf.item()
                detection_msg.Class = self.names[int(det_class.item())]
                print(detection_msg.xmin, detection_msg.xmax, detection_msg.ymin, detection_msg.ymax, detection_msg.probability, detection_msg.Class)

                # Append in overall detection message
                detection_results.bounding_boxes.append(detection_msg)

        # Publish detection results
        self.pub_.publish(detection_results)

        # Visualize detection results
        if (self.publish_image):
            self.visualizeAndPublish(detection_results, self.cv_image)
        return True

    def imagePreProcessing(self, img):
        # Extract image and shape
        img = np.copy(img)
        img = img.astype(float)
        height, width, channels = img.shape

        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width

            # Determine image to be used
            self.padded_image = np.zeros((max(self.h, self.w), max(self.h, self.w), channels)).astype(float)

        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w - self.h) // 2: self.h + (self.w - self.h) // 2, :, :] = img
        else:
            self.padded_image[:, (self.h - self.w) // 2: self.w + (self.h - self.w) // 2, :] = img

        # Resize and normalize
        input_img = resize(self.padded_image, (self.network_img_size, self.network_img_size, 3)) / 255.

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = input_img[None]

        return input_img

    def visualizeAndPublish(self, output, imgIn):
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        for index in range(len(output.bounding_boxes)):
            label = output.bounding_boxes[index].Class
            x_p1 = output.bounding_boxes[index].xmin
            y_p1 = output.bounding_boxes[index].ymin
            x_p3 = output.bounding_boxes[index].xmax
            y_p3 = output.bounding_boxes[index].ymax
            confidence = output.bounding_boxes[index].probability

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0, 255, 3)
                self.classes_colors[label] = color
            # print(color[0], color[1], color[2], type(color[0]), type(color[1]), type(color[2]))

            # Create rectangle
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (int(color[0]), int(color[1]), int(color[2])),
                          thickness)
            text = ('{:s}: {:.3f}').format(label, confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1 + 20)), font, fontScale, (255, 255, 255), thickness,
                        cv2.LINE_AA)
        # Publish visualization image
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        self.pub_viz_.publish(image_msg)


if __name__=="__main__":
    # Initialize node
    rospy.init_node("detector_manager_node")

    # Define detector object
    dm = DetectorManager()