# -*- encoding= cp949 -*-

from keras import backend as K
import tensorflow as tf
import scipy.io as io
import numpy as np

def loadData():
    data = io.loadmat("face_landmark.mat")
    images = data["images"]
    landmarks = data["landmarks"]
    print("im_shape:", images.shape)
    print("landmarks_shape:", landmarks.shape)

    return images, landmarks

def makeData(data, size):
    indices = np.random.permutation(size)

    train_size = int(size * 0.6)
    val_size = int(size * 0.2)
    test_size = int(size * 0.2)

    train_ds = data[indices[ : train_size]]
    val_ds = data[indices[train_size : train_size + val_size]]
    test_ds = data[indices[size - test_size : ]]
    
    return train_ds, val_ds, test_ds

def makeModel():    
    # true 실제 값, pred 예측 값
    def L1_loss(true, pred):
        return K.mean(tf.abs(pred - true))

    def L2_loss(true, pred):
        return K.mean(tf.square(pred - true))

    def cosine_loss(true, pred):
        true = tf.nn.l2_normalize(true, axis = -1)
        pred = tf.nn.l2_normalize(pred, axis = -1)
        return K.mean(1 - K.sum(true * pred, axis = -1))

    def combined_loss(L2_weight, cosine_weight):
        def loss_fn(true, pred):
            L2 = L2_loss(true, pred)
            cosine = cosine_loss(true, pred)
            return L2_weight * L2 + cosine_weight * cosine
        return loss_fn

    regularizer = None
    #regularizer = tf.keras.regularizers.l1(0.01)
    #regularizer = tf.keras.regularizers.l2(0.01)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (96, 96, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer = regularizer),
        tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer = regularizer),
        tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer = regularizer),
        tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer = regularizer),
        tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu', kernel_regularizer = regularizer),
        # tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(15 * 2, activation = None),
        tf.keras.layers.Reshape((15, 2))
    ])

    # MSE 예측값과 실제값 간의 평균 제곱 오차, MAE 예측값과 실제값 간의 평균 절대 오차
    # lambda_value = 0.5
    # loss_fn = combined_loss(lambda_value, 1 - lambda_value)
    model.compile(optimizer = 'adam', loss = cosine_loss, metrics = ['mse', 'mae'])

    return model

images, landmarks = loadData()
train_images, val_images, test_images = makeData(images, images.shape[0])
train_landmarks, val_landmarks, test_landmarks = makeData(landmarks, landmarks.shape[0])

model = makeModel()

history = model.fit(train_images, train_landmarks, validation_data = (val_images, val_landmarks), epochs = 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
mse = history.history["mse"]
val_mse = history.history["val_mse"]
mae = history.history["mae"]
val_mae = history.history["val_mae"]

test_loss, test_mse, test_mae = model.evaluate(test_images, test_landmarks)