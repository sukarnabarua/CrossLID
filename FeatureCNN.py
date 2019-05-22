from keras import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
from utilities import scale_value


def get_cnn_mnist_model():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def get_pretrained_cnn_mnist_model():
    model = get_cnn_mnist_model()
    model.load_weights('cnn.h5')
    return model

def get_deep_features(X):
    my_input_min_max = [0.0, 1.0]
    layer_idx = 5 # Feature extraction layer
    feature_len = 128
    model = get_pretrained_cnn_mnist_model()
    X_scaled = scale_value(X, my_input_min_max)
    intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[layer_idx].output)
    features = intermediate_layer_model.predict(X_scaled)
    np.random.shuffle(features)
    return features.reshape((-1, feature_len))