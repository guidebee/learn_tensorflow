import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

from convert_hz24_to_numpy import convert_hz24_to_numpy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)
print(x_test.shape)

image_numbers = np.zeros((28, 280))
image_numbers_24 = convert_hz24_to_numpy("０１２３４５６７８９０")
for k in range(10):
    for i in range(24):
        for j in range(24):
            image_numbers[i + 2, j + 2 + k * 28] = image_numbers_24[i, j + k * 24]

number_0 = image_numbers[:, :28]
number_1 = image_numbers[:, 28:56]
number_2 = image_numbers[:, 56:84]
number_3 = image_numbers[:, 84:112]
number_4 = image_numbers[:, 112:140]
number_5 = image_numbers[:, 140:168]
number_6 = image_numbers[:, 168:196]
number_7 = image_numbers[:, 196:224]
number_8 = image_numbers[:, 224:252]
number_9 = image_numbers[:, 252:280]
number_matrix = [number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8, number_9]

#map y_train to number_matrix
y_train_matrix = np.zeros((y_train.shape[0], 28, 28))
for i in range(y_train.shape[0]):
    y_train_matrix[i] = number_matrix[y_train[i]]

y_test_matrix = np.zeros((y_test.shape[0], 28, 28))
for i in range(y_test.shape[0]):
    y_test_matrix[i] = number_matrix[y_test[i]]


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


shape = x_test.shape[1:]
latent_dim = 64
autoencoder = Autoencoder(latent_dim, shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, y_train_matrix,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, y_test_matrix))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

