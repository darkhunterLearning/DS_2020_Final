from tensorflow.keras.models import load_model
import numpy as np

class CharacterClassifier:
    def __init__(self):
        self.model = None
        self.labels = [str(idx) for idx in range(10)] + [chr(idx) for idx in range(65, 65+26)]
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}

    def load_model(self):
        self.model = load_model('models/CharacterClassifier.h5')

    def predict(self, image):
        """
        Autotest will call this function
        :param image: a PIL Image object
        :return: a list of boxes, each item is a tuple of (x_min, y_min, x_max, y_max)
        """
        # resize_image = np.array(image.resize([self.image_w, self.image_h]))
        resized_image = np.array(image.resize([50, 50]))
        output = self.model.predict(np.array([resized_image]))[0]
        label = self.labels[np.argmax(output)]
        return label