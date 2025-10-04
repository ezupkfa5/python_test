import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

mnist_train = pd.read_csv("train.csv")
mnist_test = pd.read_csv("test.csv")

X_train = mnist_train.drop("label",axis=1)
y_train = mnist_train["label"]
X_test = mnist_test

X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0

X_train_conv = tf.reshape(X_train,(-1,28,28,1))
X_test_conv = tf.reshape(X_test,(-1,28,28,1))

# print(X_train_conv.shape) #(42000, 28, 28, 1)
# print(X_test_conv.shape)  #(28000, 28, 28, 1)
#print(X_train.head(),y_train.head(),sep='\n')

model = tf.keras.Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=3,strides=1,padding="same",activation="relu",input_shape=(28,28,1)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=64,kernel_size=3,strides=1,padding="same",activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(1024,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

model.summary()

model.compile(optimizer="adam",loss="crossentropy",metrics=["accuracy"])
history = model.fit(X_train_conv,y_train,batch_size=16,epochs=5)

history_pd = pd.DataFrame(history.history)
history_pd.loc[:,"loss"].plot()
history_pd.loc[:,"accuracy"].plot()
plt.show()


# #最後才預測
# y_pred = model.predict(X_test_conv)
# y_pred_class = np.argmax(y_pred,axis=1)
# y_pred_class_pd = pd.DataFrame({"ImageId":range(1, len(y_pred_class) + 1),"Label":y_pred_class})

# y_pred_class_pd.to_csv("y_pred_class_CONV.csv",index= False)
# print(y_pred_class_pd.head())
