# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 13:46:18 2026

@author: Dell
"""
# ==========================================
# Tiny CNN for CIFAR-10 (FPGA Friendly)
# ==========================================

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import time
import matplotlib.pyplot as plt

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training images shape:", x_train.shape)
print("Test images shape:", x_test.shape)

# Normalize images (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# ------------------------------------------
# 2. Build 3-Layer CNN (Higher Accuracy)
# ------------------------------------------

model = Sequential([

    # Conv Block 1
    Conv2D(16, (3,3), activation='relu', padding='same',
           input_shape=(32,32,3)),
    MaxPooling2D((2,2)),

    # Conv Block 2
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    # Conv Block 3  (NEW LAYER)
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Flatten(),

    # Small dense layer improves accuracy
    Dense(64, activation='relu'),

    Dense(10, activation='softmax')
])

# ------------------------------------------
# 3. Compile Model
# ------------------------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------
# 4. Train Model
# ------------------------------------------

start_train = time.time()

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

end_train = time.time()

print("CPU Training Time:", end_train - start_train, "seconds")

# ------------------------------------------
# 5. Evaluate Model
# ------------------------------------------

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# ------------------------------------------
# 6. Measure CPU Inference Time (1 Image)
# ------------------------------------------

sample = np.expand_dims(x_test[0], axis=0)

start_inf = time.time()
prediction = model.predict(sample)
end_inf = time.time()

print("CPU Inference Time per Image:",
      (end_inf - start_inf) * 1000, "ms")

print("Predicted Class:", np.argmax(prediction))
print("Actual Class:", y_test[0][0])

# ------------------------------------------
# 7. Save Model
# ------------------------------------------

model.save("tiny_cnn_model.h5")
print("Model saved as tiny_cnn_model.h5")

# ------------------------------------------
# 8. Export Weights for FPGA
# ------------------------------------------
# ------------------------------------------
# 8. Export Weights for NumPy Inference
# ------------------------------------------

weights = model.get_weights()

conv1_w = weights[0].transpose(3,2,0,1)
conv1_b = weights[1]

conv2_w = weights[2].transpose(3,2,0,1)
conv2_b = weights[3]

conv3_w = weights[4].transpose(3,2,0,1)
conv3_b = weights[5]

fc1_w = weights[6]
fc1_b = weights[7]

fc2_w = weights[8]
fc2_b = weights[9]

np.savez("trained_weights_3conv.npz",
         conv1_w=conv1_w, conv1_b=conv1_b,
         conv2_w=conv2_w, conv2_b=conv2_b,
         conv3_w=conv3_w, conv3_b=conv3_b,
         fc1_w=fc1_w, fc1_b=fc1_b,
         fc2_w=fc2_w, fc2_b=fc2_b)

print("trained_weights.npz saved successfully!")
weights = model.get_weights()

conv1_w = weights[0].transpose(3,2,0,1)
conv1_b = weights[1]

conv2_w = weights[2].transpose(3,2,0,1)
conv2_b = weights[3]

conv3_w = weights[4].transpose(3,2,0,1)
conv3_b = weights[5]

fc1_w = weights[6]
fc1_b = weights[7]

fc2_w = weights[8]
fc2_b = weights[9]

np.savez("trained_weights_3conv.npz",
         conv1_w=conv1_w, conv1_b=conv1_b,
         conv2_w=conv2_w, conv2_b=conv2_b,
         conv3_w=conv3_w, conv3_b=conv3_b,
         fc1_w=fc1_w, fc1_b=fc1_b,
         fc2_w=fc2_w, fc2_b=fc2_b)

# ------------------------------------------
# 9. Show Some Predictions
# ------------------------------------------

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

plt.figure(figsize=(10,5))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_test[i])
    pred = np.argmax(model.predict(np.expand_dims(x_test[i], axis=0)))
    plt.title("P: {} | A: {}".format(
        class_names[pred],
        class_names[y_test[i][0]]
    ))
    plt.axis('off')

plt.tight_layout()
plt.show()