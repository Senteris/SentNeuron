import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tensorflow.python.keras.models import model_from_json
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

images_to_predict = [str(path) for path in pathlib.Path('../images_to_predict/').glob('*')]


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range

    return image


print("Loading model from file")
# Загружаем данные об архитектуре сети
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель
model = model_from_json(loaded_model_json)
# Загружаем сохраненные веса в модель
model.load_weights("model.h5")
print("Loading done")

data_root = pathlib.Path('../data/')

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))

print("Predict...")

for image_path in images_to_predict:
    prediction = model.predict(np.expand_dims(load_and_preprocess_image(image_path), axis=0))
    print(image_path, prediction)
    plt.figure()
    plt.imshow(load_img(image_path))
    plt.title(
        f'This is a {[key for key, value in label_to_index.items() if value == list(prediction[0]).index(max(prediction[0]))][0]} ({round(max(prediction[0]) * 100)}%)')
    plt.xlabel(image_path)
    plt.show()
