import os
import pandas as pd
#import tensorflow as tf

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

