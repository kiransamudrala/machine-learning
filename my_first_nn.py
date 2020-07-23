import keras as ke
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist=tf.keras.datasets.mnist

(xtrain,ytrain),(xt,yt)=mnist.load_data()

xtrain=tf.keras.utils.normalize(xtrain)
ytrain=tf.keras.utils.normalize(ytrain)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

plt.imshow(xtrain[0])

