PATH_DATA_TO_PREDICT  = '../data/pregictData.csv'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import pandas as pd
from tensorflow import keras

#Import the model
model = keras.models.load_model('my_model.h5')
model.summary()

#Getting predict data
column_names = ['Timestamp','Open','High','Low','Close',
                'Volume_(BTC)', 'Volume_(Currency)']

raw_dataset = pd.read_csv(PATH_DATA_TO_PREDICT, names=column_names,
                      na_values = "NaN",
                      sep=",", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

#endregion

#Answer
predict = model.predict(dataset)
print(float(predict[0]))

#endregion