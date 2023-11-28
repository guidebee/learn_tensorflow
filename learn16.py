# grader-required-cell

import csv
import string

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img


# grader-required-cell

# GRADED FUNCTION: parse_data_from_input
def parse_data_from_input(filename):
    """
    Parses the images and labels from a CSV file

    Args:
      filename (string): path to the CSV file

    Returns:
      images, labels: tuple of numpy arrays containing the images and labels
    """
    with open(filename) as file:
        ### START CODE HERE

        # Use csv.reader, passing in the appropriate delimiter
        # Remember that csv.reader can be iterated and returns one line in each iteration
        csv_reader = csv.reader(file, delimiter=None)
        #read all lines from csv_reader
        lines = list(csv_reader)
        # The first line contains the column headers, so remove it
        lines = lines[1:]
        # Separate the labels from the features
        labels = [line[0] for line in lines]
        images = [line[1:] for line in lines]
        # Convert the data to np.array format
        labels = np.array(labels)
        images = np.array(images)
        # Convert the data to float32
        labels = labels.astype(np.float32)
        images = images.astype(np.float32)
        # Reshape the images to 28x28
        images = images.reshape(images.shape[0], 28, 28)



        ### END CODE HERE

        return images, labels


# grader-required-cell

# GRADED FUNCTION: train_val_generators
def train_val_generators(training_images, training_labels, validation_images, validation_labels):
    """
    Creates the training and validation data generators

    Args:
      training_images (array): parsed images from the train CSV file
      training_labels (array): parsed labels from the train CSV file
      validation_images (array): parsed images from the test CSV file
      validation_labels (array): parsed labels from the test CSV file

    Returns:
      train_generator, validation_generator - tuple containing the generators
    """
    ### START CODE HERE


    #add new dimension for training_image
    training_images =training_images[:,:,:,np.newaxis]
    validation_images = validation_images[:,:,:,np.newaxis]

    # Instantiate the ImageDataGenerator class
    # Don't forget to normalize pixel values
    # and set arguments to augment the images (if desired)
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Pass in the appropriate arguments to the flow method
    train_generator = train_datagen.flow(x=training_images,
                                         y=training_labels,
                                         batch_size=32)

    # Instantiate the c class (don't forget to set the rescale argument)
    # Remember that validation data should not be augmented
    validation_datagen =  ImageDataGenerator(rescale=1.0/255.0)

    # Pass in the appropriate arguments to the flow method
    validation_generator = validation_datagen.flow(x=validation_images,
                                                   y=validation_labels,
                                                   batch_size=32)

    ### END CODE HERE

    return train_generator, validation_generator


def create_model():
    ### START CODE HERE

    # Define the model
    # Use no more than 2 Conv2D and 2 MaxPooling2D
    model = keras.Sequential(
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)
        ),
        # Max-pooling layer, using 2x2 pool size
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        keras.layers.Conv2D(
            32, (3, 3), activation='relu'
        ),
        # Max-pooling layer, using 2x2 pool size
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten units
        keras.layers.Flatten(),
        # Add a hidden layer with dropout
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        # Add an output layer with output units for all 10 digits
        keras.layers.Dense(25, activation='softmax')

    )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    ### END CODE HERE

    return model


# grader-required-cell

# GRADED FUNCTION: create_model
def create_model(total_words, max_sequence_len):
    """
    Creates a text generator model

    Args:
        total_words (int): size of the vocabulary for the Embedding layer input
        max_sequence_len (int): length of the input sequences

    Returns:
        model (tf.keras Model): the text generator model
    """
    model = Sequential()
    ### START CODE HERE
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    ### END CODE HERE

    return model

# Get the untrained model
model = create_model(total_words, max_sequence_len)

# Train the model
history = model.fit(features, labels, epochs=50, verbose=1)