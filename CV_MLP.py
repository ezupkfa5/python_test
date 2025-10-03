import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

#from tensorflow import keras

#print(tf.__version__) #2.20.0
#print(os.getcwd())  # 目前工作目錄
#print(__file__)     # 目前檔案完整路徑

mnist_train = pd.read_csv("train.csv")
mnist_test = pd.read_csv("test.csv")
#print(mnist_test.head())

X_train = mnist_train.drop("label",axis= 1)
y_train = mnist_train["label"]
#print(X_train.shape) #(42000, 784))

#print(y_train.shape) #(42000,)
X_test = mnist_test
#print(X_test.shape) #(28000, 784)


model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(units=128,activation="relu",input_shape=(784,)))
model.add(tf.keras.layers.Dense(units=64,activation='relu'))
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

model.summary()
model.compile(optimizer="adam",loss="crossentropy",metrics=["accuracy"])

history = model.fit(X_train,y_train,batch_size=200,epochs=10)

history_pd = pd.DataFrame(history.history)
history_pd.loc[:,["loss"]].plot()
history_pd.loc[:,["accuracy"]].plot()
#plt.show()
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred,axis=1)
y_pred_class_pd = pd.DataFrame({"ImageId":range(1, len(y_pred_class) + 1),"Label":y_pred_class})

y_pred_class_pd.to_csv("y_pred_class_MLP.csv",index= False)
print(y_pred_class_pd.head())