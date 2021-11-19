import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# normalize the data to make it easier to handle
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# building the neural network
model = tf.keras.models.Sequential()  # create the model
# define individual layers
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # input layer, it transforms this shape into one dimension
model.add(tf.keras.layers.Dense(units=128, activation='relu'))  # hidden layer relu = rectified linear unit
model.add(tf.keras.layers.Dense(units=128, activation='relu'))  # returns 0 when the input is < 0, the input otherwise
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  # output layer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=3)

# model.save('handwritten.model')

loss, accuracy = model.evaluate(X_test, Y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)

image = cv.imread('digit.png')[:, :, 0]
image = np.invert(np.array([image]))

prediction = model.predict(image)
print(f"Prediction: {np.argmax(prediction)}")
plt.imshow(image[0])
plt.show()
