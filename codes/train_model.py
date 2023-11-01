import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from data_analysis import x_train, x_test, y_train, y_test

# Normalize images
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10)
])

# Model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model training
model.fit(x_train, y_train, epochs=20)

# Save model
model.save('fashion_mnist_model.h5')
