import pandas as pd
import numpy as np
import tensorflow as tf 

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers,models

mnist_train = pd.read_csv("train.csv")
X_train = mnist_train.drop("label",axis=1)
y_train = mnist_train["label"]

X_train = tf.reshape(X_train,(-1,28,28,1))

X_train = np.repeat(X_train,3,axis=-1)
print(X_train.shape)
# base_model = ResNet50(weights="imagenet",include_top=False,input_shape=(32,32,3))

