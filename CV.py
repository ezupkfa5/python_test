import os
import numpy as np
import pandas as pd
#import tensorflow as tf

print(os.getcwd())  # 目前工作目錄
print(__file__)     # 目前檔案完整路徑

x_train = pd.read_csv("train.csv",index_col="label")
print(x_train.head())
