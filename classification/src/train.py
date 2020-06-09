#region Settings
BATCH_SIZE = 224
IMAGE_SIZE = 256
EPOCHS = 16
TEST = False
#endregion

#region imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

#endregion

print("Loading dataset from folder")

#region Get paths
data_root = pathlib.Path('../data/')

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
image_count = len(all_image_paths)

print(label_to_index)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
#endregion

#region Creating dataset
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# Установка размера буфера перемешивания, равного набору данных, гарантирует полное перемешивание данных.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
ds = ds.prefetch(buffer_size=AUTOTUNE)

#endregion

#region Splitting dataset
train_size = int(0.95 * image_count)

train_dataset = ds.take(train_size)
test_dataset = ds.skip(train_size)

#endregion

print(f"Loading done: {ds}")

#region Creating model
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False)
mobile_net.trainable = False

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(label_names), activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

#endregion

#region Training neuronet
steps_per_epoch = tf.math.ceil(train_size / BATCH_SIZE).numpy()

history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)

#endregion

print("Fit done")

#region Stats
acc = history.history['accuracy']
loss = history.history['loss']

if TEST: print(model.evaluate(test_dataset))

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#endregion

#region Saving
print("Saving")

# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("model.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("model.h5")
print("Save done")

#endregion