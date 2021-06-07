#!/usr/bin/env python3

from __future__ import division

import torch
from torch.autograd import Variable
import numpy as np

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import (non_max_suppression, set_logging, yolov5_in_syspath)
from yolov5.utils.torch_utils import (load_classifier, select_device)
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import rospy
from rospkg import RosPack
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from yolov5_deepsort.msg import BoundingBox, BoundingBoxes
from skimage.transform import resize
from multiprocessing import Process, Queue

import os, cv2, sys
# sys.path.insert(0, './yolov5')

package = RosPack()
package_path = package.get_path('yolov5_deepsort')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


class TrackerManager():
    def __init__(self):
        # Setup Deep sort
        config_name = rospy.get_param('~config_path', 'deep_sort_pytorch/configs/deep_sort.yaml')
        ckpt_name = rospy.get_param('~ckpt_path', 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7')
        self.ckpt_path = os.path.join(package_path, 'scripts', ckpt_name)
        self.config_path = os.path.join(package_path, 'scripts', config_name)
        cfg = get_config(config_file=self.config_path)
        cfg.merge_from_file(self.config_path)
        self.deepsort = DeepSort(self.ckpt_path,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

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
        if self.class_name == "None":
            self.class_name = None
        elif self.class_name.find(', ') == -1:
            self.class_name = self.names.index(self.class_name)
        else:
            self.class_name = self.class_name.split(', ')
            self.class_name = [self.names.index(i) for i in self.class_name]

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
            detections = self.model(input_img, augment=False)[0]
            detections = non_max_suppression(detections, self.confidence_th, self.nms_th, classes=self.class_name, agnostic=False)

            for i, detection in enumerate(detections):
                if detection is not None and len(detection):
                    xywh_bboxs = []
                    confs = []

                    for *xyxy, conf, cls in detection:
                        x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)

                    # pass detections to deepsort
                    try:
                        outputs = self.deepsort.update(xywhs, confss, self.cv_image)
                        print("output finished")
                        if len(outputs) > 0:
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -1]
                            confidence = outputs[:, -2]
                            offset = (0, 0)
                            for i, box in enumerate(bbox_xyxy):
                                x1, y1, x2, y2 = [int(i) for i in box]
                                x1 += offset[0]
                                x2 += offset[0]
                                y1 += offset[1]
                                y2 += offset[1]

                                pad_x = max(self.h - self.w, 0) * (self.network_img_size / max(self.h, self.w))
                                pad_y = max(self.w - self.h, 0) * (self.network_img_size / max(self.h, self.w))
                                unpad_h = self.network_img_size - pad_y
                                unpad_w = self.network_img_size - pad_x
                                xmin_unpad = ((x1 - pad_x // 2) / unpad_w) * self.w
                                xmax_unpad = ((x2 - x1) / unpad_w) * self.w + xmin_unpad
                                ymin_unpad = ((y1 - pad_y // 2) / unpad_h) * self.h
                                ymax_unpad = ((y2 - y1) / unpad_h) * self.h + ymin_unpad
                                # box text and bar
                                id = int(identities[i]) if identities is not None else 0
                                detection_msg = BoundingBox()
                                detection_msg.xmin = xmin_unpad
                                detection_msg.xmax = xmax_unpad
                                detection_msg.ymin = ymin_unpad
                                detection_msg.ymax = ymax_unpad
                                detection_msg.probability = float(confidence[i])
                                detection_msg.Class = self.names[int(cls.item())] + "_" + str(id)
                                print(detection_msg.xmin, detection_msg.xmax, detection_msg.ymin, detection_msg.ymax,
                                      detection_msg.probability, detection_msg.Class)
                                detection_results.bounding_boxes.append(detection_msg)

                        self.pub_.publish(detection_results)

                        if (self.publish_image):
                            self.visualizeAndPublish(detection_results, self.cv_image)
                    except Exception as e:
                        print(e)
                        return False
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

            # Create rectangle
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (int(color[0]), int(color[1]), int(color[2])),
                          thickness)
            text = ('{:s}: {:.3f}').format(label, confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1 + 20)), font, fontScale, (255, 255, 255), thickness,
                        cv2.LINE_AA)

        # Publish visualization image
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        self.pub_viz_.publish(image_msg)

    def xyxy_to_xywh(*xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[1].item(), xyxy[3].item()])
        bbox_top = min([xyxy[2].item(), xyxy[4].item()])
        bbox_w = abs(xyxy[1].item() - xyxy[3].item())
        bbox_h = abs(xyxy[2].item() - xyxy[4].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h


if __name__=="__main__":
    # Initialize node
    rospy.init_node("tracker_manager_node")

    # Define detector object
    tm = TrackerManager()