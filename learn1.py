import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# add 3 channels dimension to x_train and x_test
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
x_train, x_test = x_train / 255.0, x_test / 255.0
# for i in range(28):
#     for j in range(28):
#         x_train[:, i, j, 1] = i
#         x_train[:, i, j, 2] = j
#         x_test[:, i, j, 1] = i
#         x_test[:, i, j, 2] = j


# # expand x_train and x_test to 4D array
# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]

# # display first 10 images of test data and labels
# num_test = 25
# num_grid = 5
# plt.figure(figsize=(10, 10))
# plt.clf()
# plt.title('MNIST Dataset')
# for i in range(num_test):
#     # show plot title
#
#     plt.subplot(int(num_test / num_grid), num_grid, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     # sequence of images x_test[i]
#
#     plt.imshow(x_test[i])
#     # display predicted label
#
#     plt.xlabel(y_test[i])
#     # save the plot
#
# plt.show()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

feature_extractors = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(9, activation='relu'),

])

feature_classifier = tf.keras.layers.Dense(10)
model = tf.keras.models.Sequential([
    feature_extractors,
    feature_classifier
])

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)
feature_classifier.trainable = False
for k in range(60):
    print(f'k={k}')
    if k % 2 == 0:
        feature_extractors = tf.keras.models.Sequential([
            # tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=(28, 28, 1)),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(9, activation='relu'),

        ])
    else:
        feature_extractors = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(9, activation='relu'),

        ])
    model = tf.keras.models.Sequential([
        feature_extractors,
        feature_classifier
    ])

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    num_test = 25
    num_grid = 5
    # display first 10 images of test data and predicted labels
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(x_test[:num_test])
    features = feature_extractors.predict(x_test[:num_test])
    plt.figure(figsize=(10, 10))
    plt.clf()
    plt.title(f'MNIST Feature Mappings {k}')
    for i in range(num_test):
        # show plot title

        plt.subplot(int(num_test / num_grid), num_grid, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # sequence of images x_test[i]

        plt.imshow(tf.squeeze(features[i].reshape(3, 3)), cmap='binary')
        # display predicted label

        plt.xlabel(np.argmax(predictions[i]))
        # save the plot
        plt.savefig(f'plot3x3_{k}.png')

# clear the plot
