import numpy as np
import pickle
import time
import os

# ==========================
# CIFAR-10 classes
# ==========================

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# ==========================
# Load CIFAR-10
# ==========================

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']
        images = images.reshape(-1,3,32,32)
        images = images.astype(np.float32)/255.0
        return images, np.array(labels)

# ==========================
# Activation
# ==========================

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# ==========================
# Convolution
# ==========================

def conv2d(x, weight, bias, stride=1, padding=0):

    N,C,H,W = x.shape
    F,_,HH,WW = weight.shape

    H_out = (H - HH + 2*padding)//stride + 1
    W_out = (W - WW + 2*padding)//stride + 1

    out = np.zeros((N,F,H_out,W_out))

    if padding > 0:
        x_padded = np.pad(x,
                          ((0,0),(0,0),
                           (padding,padding),
                           (padding,padding)),
                          mode='constant')
    else:
        x_padded = x

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    region = x_padded[n,
                                      :,
                                      i*stride:i*stride+HH,
                                      j*stride:j*stride+WW]
                    out[n,f,i,j] = np.sum(region * weight[f]) + bias[f]

    return out

# ==========================
# Max Pool
# ==========================

def maxpool(x, size=2, stride=2):

    N,C,H,W = x.shape
    H_out = (H - size)//stride + 1
    W_out = (W - size)//stride + 1

    out = np.zeros((N,C,H_out,W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    region = x[n,
                               c,
                               i*stride:i*stride+size,
                               j*stride:j*stride+size]
                    out[n,c,i,j] = np.max(region)

    return out

# ==========================
# Forward pass (3 Conv)
# ==========================

def forward(x, weights):

    conv1_w = weights['conv1_w']
    conv1_b = weights['conv1_b']

    conv2_w = weights['conv2_w']
    conv2_b = weights['conv2_b']

    conv3_w = weights['conv3_w']
    conv3_b = weights['conv3_b']

    fc1_w = weights['fc1_w']
    fc1_b = weights['fc1_b']

    fc2_w = weights['fc2_w']
    fc2_b = weights['fc2_b']

    


    # Conv Block 1
    x = conv2d(x, conv1_w, conv1_b, padding=1)
    x = relu(x)
    x = maxpool(x)

    # Conv Block 2
    x = conv2d(x, conv2_w, conv2_b, padding=1)
    x = relu(x)
    x = maxpool(x)

    # Conv Block 3  (NEW)
    x = conv2d(x, conv3_w, conv3_b, padding=1)
    x = relu(x)
    x = maxpool(x)

    # Flatten
    x = x.reshape(x.shape[0], -1)

    # Dense 1
    x = np.dot(x, fc1_w) + fc1_b
    x = relu(x)

    # Output layer
    x = np.dot(x, fc2_w) + fc2_b

    return softmax(x)

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    data_dir = r"D:\Downloads\arm\cifar-10-batches-py"
    weight_file =r"C:\Users\Dell\Desktop\Miniproject\ARM\PYTHON CODES\trained_weights.npz"

    print("Loading data...")
    X_test, y_test = load_cifar10_batch(
        os.path.join(data_dir, "test_batch"))

    print("Loading weights...")
    weights = np.load(weight_file)

    print("Running inference on ARM CPU...")

    start = time.time()

    outputs = forward(X_test, weights)
    preds = np.argmax(outputs, axis=1)

    end = time.time()

    acc = np.mean(preds == y_test) * 100

    print("\nAccuracy: %.2f%%" % acc)

    print("\nSample predictions:\n")
    for i in range(5):
        print("Original :", classes[y_test[i]])
        print("Predicted:", classes[preds[i]])
        print("----------")

    print("\nTotal inference time:", end-start, "seconds")