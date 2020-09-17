import keras
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
keras.backend.backend()
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
y_train.shape
y_test.shape
x_test.shape
x_train[0].shape
plt.matshow(x_train[1])
x_train=x_train/255
x_test=x_test/255
x_train[0]
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train[0].shape
y_train=keras.utils.to_categorical(y_train)
y_test=keras.utils.to_categorical(y_test)
y_train[0]
models=keras.models.Sequential()
models.add(keras.layers.InputLayer(input_shape=(784,)))
models.add(keras.layers.Dense(128,activation="relu"))
models.add(keras.layers.Dense(10,activation="softmax"))
models.summary()
models.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
models.fit(x_train,y_train,epochs=10)

yp=models.predict(x_test)
np.argmax(yp[1])


