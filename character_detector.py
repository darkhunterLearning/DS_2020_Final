# character detector
import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Reshape
from utils import decode_netout, merge_file
from losses import  YoloLoss
from tensorflow.keras.applications import InceptionV3
from  tensorflow.keras.applications.inception_v3 import preprocess_input

class CharacterDetector:
    def __init__(self):
        self.model = None
        self.anchors = [2.97246,5.71198, 3.22137,8.57165, 4.26194,8.53863, 5.22358,8.50339, 5.42089,9.80336]
        self.labels = ['chars']
        self.num_classes = len(self.labels)
        self.batch_size = 16

    def load_model(self):
        merge_file('models/CharacterDetector.h5', 7)
        self.model = load_model(
            'models/CharacterDetector.h5',
            custom_objects={'yolo_loss': YoloLoss(
                self.anchors, [8, 8], self.batch_size, lambda_obj=5.0
                )
            }
        )

    def predict(self, image):
        """
        Autotest will call this function
        :param image: a PIL Image object
        :return: a list of boxes, each item is a tuple of (x_min, y_min, x_max, y_max)
        """
        w = image.width
        h = image.height
        image = image.resize([300, 300])
        image_np = np.array(image)
        preprocessed_image = np.array([self.preprocess_input(image_np)])
        netout = self.model.predict(preprocessed_image)[0]
        boxes = decode_netout(
            netout, self.anchors, self.num_classes, 0.5, 0.2
        )
        
        list_result = []
        for box in boxes:
            xmin = box.xmin * w
            ymin = box.ymin * h
            xmax = box.xmax * w
            ymax = box.ymax * h
            list_result.append((xmin, ymin, xmax, ymax))
        return list_result

    def preprocess_input(self, image):
        return preprocess_input(image)

    def infer(self, image, iou_threshold=0.5, score_threshold=0.5):
        image = cv2.resize(image, (300, 300))
        image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        image = self.preprocess_input(image)
        if len(image.shape) == 3:
            input_image = image[np.newaxis, :]
        else:
            input_image = image[np.newaxis, ..., np.newaxis]

        netout = self.model.predict(input_image)[0]

        boxes = decode_netout(netout, self.anchors, self.num_classes, score_threshold, iou_threshold)

        return boxes
