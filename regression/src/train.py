#region Settings
import sys
EPOCHS = 200
TEST = True
POPING_COLUMN = "Weighted_Price"
BATCH_SIZE = 60
#endregion

#region imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#endregion

#region Get paths
dataset_path = '../data/bitstampUSD_data_2012_to_2020.csv'

#endregion

#region Import dataset
print("Start importing the dataset")
column_names = ['Timestamp','Open','High','Low','Close',
                'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "NaN",
                      sep=",", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()

#endregion

#region Splitting dataset
train_dataset = dataset.sample(frac=0.95,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#endregion

#region Normalize data
print("Start normalizing of the data")
train_stats = train_dataset.describe()
train_stats.pop(POPING_COLUMN)
train_stats = train_stats.transpose()

train_labels = train_dataset.pop(POPING_COLUMN)
test_labels = test_dataset.pop(POPING_COLUMN)

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#endregion

#region Creating model
print("Start creating the model")
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

#endregion

#region Training neuronet
# Выведем прогресс обучения в виде точек после каждой завершенной эпохи
print("Start training the neuronet")
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
      print(f"\rProgress: {round(epoch / EPOCHS * 100)}%")

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=1, callbacks=[early_stop, PrintDot()], batch_size=BATCH_SIZE)

sys.stdout.write(f"\rProgress: 100%")
print('')
#endregion

#region Stats
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel(f'Mean Abs Error [{POPING_COLUMN}]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel(f'Mean Square Error [{POPING_COLUMN}^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)

if TEST:
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel(f'True Values [{POPING_COLUMN}]')
    plt.ylabel(f'Predictions [{POPING_COLUMN}]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.title("Testing set Mean Abs Error: {:5.2f} {}".format(mae, POPING_COLUMN))
    plt.show()
#endregion