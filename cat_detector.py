import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Reshape
from utils import decode_netout, merge_file
from tensorflow.keras.applications import InceptionV3
from  tensorflow.keras.applications.inception_v3 import preprocess_input
# from yolo import FullYolo
# from yolo import preprocess_input 
# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.applications.mobilenet import preprocess_input
from losses import  YoloLoss
from callbacks import MapEvaluation
from data_generator import BatchGenerator, parse_annotation_xml
from tensorflow.keras.optimizers import Adam


class CatDetector:
    def __init__(self):
        self.model = None
        self.anchors = [5.67861,7.16968, 9.94545,11.93625, 13.77812,19.51335, 19.02503,14.69332, 22.18607,25.66493]
        self.labels = ['cat']
        self.num_classes = len(self.labels)
        self.batch_size = 16

    def build_model(self):
        num_anchors = len(self.anchors)//2
        backend = InceptionV3(
            include_top=False,
            input_shape=[500, 500, 3],
            weights='pretrained/inception_backend.h5'
        )
        # backend = FullYolo(
        #     [416, 416, 3],
        #     weights='pretrained/full_yolo_backend.h5'
        # )
        # backend = MobileNet(
        #     include_top=False,
        #     input_shape=[416, 416, 3],
        #     weights='pretrained/mobilenet_backend.h5'
        # )
        conv_layer_1 = Conv2D(filters=num_anchors * 6, kernel_size=[1, 1],)(backend.output)
        output_layer = Reshape([14, 14, num_anchors, 6])(conv_layer_1)
        self.model = Model(backend.input, output_layer)
        self.model.summary()

        loss = YoloLoss(
            self.anchors, [14, 14], self.batch_size,
            lambda_obj=5.0,
        )

        optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=loss, optimizer=optimizer)

    def save_model(self):
        self.model.save('models/cat_best_map.h5')

    def load_model(self):
        merge_file('models/cat_best_map.h5', 7)
        self.model = load_model(
            'models/cat_best_map.h5',
            custom_objects={'yolo_loss': YoloLoss(
                self.anchors, [14, 14], self.batch_size, lambda_obj=5.0
                )
            }
        )
    def train(self, **kwargs):
        num_anchors = len(self.anchors) // 2
        list_train_images, _ = parse_annotation_xml(
            'datasets/train/anns', 'datasets/train/images'
        )
        list_valid_images, _ = parse_annotation_xml(
            'datasets/valid/anns', 'datasets/valid/images'
        )
        generator_config = {
            'IMAGE_H': 500, 'IMAGE_W': 500, 'IMAGE_C': 3,
            'GRID_H': 14, 'GRID_W': 14, 'BOX': num_anchors,
            'LABELS': self.labels, 'CLASS': self.num_classes,
            'ANCHORS': self.anchors, 'BATCH_SIZE': self.batch_size
        }
        valid_generator = BatchGenerator(
            list_valid_images, generator_config, preprocess_input=preprocess_input
        )
        train_generator = BatchGenerator(
            list_train_images, generator_config, preprocess_input=preprocess_input
        )
        # generator = None
        map_evaluation = MapEvaluation(
            self, valid_generator, iou_threshold=0.6,
            save_best=True, save_name='models/cat_best_map.h5'
        )
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator) * 0.1,
            epochs=110,
            validation_data=valid_generator,
            validation_steps=len((valid_generator)) * 0.1,
            callbacks=[map_evaluation]
        )

    def predict(self, image):
        """
        Autotest will call this function
        :param image: a PIL Image object
        :return: a list of boxes, each item is a tuple of (x_min, y_min, x_max, y_max)
        """
        w = image.width
        h = image.height
        image = image.resize([500, 500])
        image_np = np.array(image)
        preprocessed_image = np.array([self.preprocess_input(image_np)])
        netout = self.model.predict(preprocessed_image)[0]
        boxes = decode_netout(
            netout, self.anchors, self.num_classes, 0.5, 0.5
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
        image = cv2.resize(image, (500, 500))
        image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        image = self.preprocess_input(image)
        if len(image.shape) == 3:
            input_image = image[np.newaxis, :]
        else:
            input_image = image[np.newaxis, ..., np.newaxis]

        netout = self.model.predict(input_image)[0]

        boxes = decode_netout(netout, self.anchors, self.num_classes, score_threshold, iou_threshold)

        return boxes
