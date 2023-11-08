import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tqdm import tqdm

from convert_hz24_to_numpy import convert_hz24_to_numpy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train = (x_train > 0.5).astype(int)
# x_test = (x_test > 0.5).astype(int)
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

# map y_train to number_matrix
y_train_matrix = np.zeros((y_train.shape[0], 28, 28))
for i in range(y_train.shape[0]):
    y_train_matrix[i] = number_matrix[y_train[i]]

y_test_matrix = np.zeros((y_test.shape[0], 28, 28))
for i in range(y_test.shape[0]):
    y_test_matrix[i] = number_matrix[y_test[i]]


# class Autoencoder(Model):
#     def __init__(self, latent_dim, shape):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.shape = shape
#         self.encoder = tf.keras.Sequential([
#             layers.Flatten(),
#             layers.Dense(latent_dim, activation='relu'),
#         ])
#         self.decoder = tf.keras.Sequential([
#             layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
#             layers.Reshape(shape)
#         ])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#
# shape = x_test.shape[1:]
# latent_dim = 128
# autoencoder = Autoencoder(latent_dim, shape)
# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
input_img = tf.keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, y_train_matrix,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, y_test_matrix))

train_encoded_imgs = autoencoder.encoder(x_train).numpy()
train_decoded_imgs = autoencoder.decoder(train_encoded_imgs).numpy()
# train_decoded_imgs = (train_decoded_imgs > 0.5).astype(int)

test_encoded_imgs = autoencoder.encoder(x_test).numpy()
test_decoded_imgs = autoencoder.decoder(test_encoded_imgs).numpy()
# test_decoded_imgs = (test_decoded_imgs > 0.5).astype(int)

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
    plt.imshow(test_decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# use nuro network to predict

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


train_decoded_imgs_ds = train_decoded_imgs[..., tf.newaxis].astype("float32")
test_decoded_imgs_ds = test_decoded_imgs[..., tf.newaxis].astype("float32")

EPOCHS = 5
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_decoded_imgs_ds, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((test_decoded_imgs_ds, y_test)).batch(32)

# for epoch in range(EPOCHS):
#     # Reset the metrics at the start of the next epoch
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     test_loss.reset_states()
#     test_accuracy.reset_states()
#
#     with tqdm(train_ds) as t:
#         t.colour = 'green'
#         for images, labels in t:
#             train_step(images, labels)
#         t.write(
#             f'Epoch {epoch + 1}, '
#             f'Loss: {train_loss.result()}, '
#             f'Accuracy: {train_accuracy.result() * 100}, '
#
#         )
#
#     with tqdm(test_ds) as t:
#         t.colour = 'red'
#         for test_images, test_labels in t:
#             test_step(test_images, test_labels)
#         t.write(
#             f'Epoch {epoch + 1}, '
#             f'Test Loss: {test_loss.result()}, '
#             f'Test Accuracy: {test_accuracy.result() * 100}'
#
#         )


# use autoencoder to predict
class DirectModel(Model):
    def call(self, x):
        size = x.shape[0]
        ret = np.one(size)
        for index in range(size):
            image = tf.squeeze(x[index])
            predictions = np.ones(10)
            for num in range(10):
                matrix = number_matrix[num]
                mse = (np.square(matrix - image)).mean()
                predictions[num] = mse
            ret[index] = np.argmin(predictions).astype(np.int8)
        return ret


direct_model = DirectModel()

error_index = []
test_size = len(test_decoded_imgs)
print(f"test size = {test_size}")
test_decoded_imgs_dir = []

for index in tqdm(range(test_size)):
    test_images, test_labels = test_decoded_imgs[index], y_test[index]
    test_images = test_images[tf.newaxis,...]
    p = direct_model(test_images)
    if p[0] != test_labels:
        error_index.append(index)
        # print(f"index={index}, p={p[0]}, test_labels={y_test[index]}")


print(error_index)
# with tqdm(test_ds) as t:
#     t.colour = 'red'
#     for test_images, test_labels in t:
#         p = direct_model(test_images)
#         diff = tf.math.equal(p, labels)
#         accuracy = tf.math.reduce_mean(tf.cast(diff, tf.float32))
#         t.write(f'accuracy={accuracy}')
