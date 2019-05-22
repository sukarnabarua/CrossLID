from keras.initializers import RandomNormal

from utilities import scale_value
from image_utils import dim_ordering_shape
from keras.models import Model
from keras.layers import BatchNormalization, Input
from keras.layers import Conv2DTranspose, Reshape, Activation
import numpy as np
from utilities import get_bn_axis

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

usegbn = True
usedbn = True

def dcgan_generator(bnmode=0):
    global usegbn, conv_init

    d_input = Input(shape=(100,))
    L = Reshape(target_shape=dim_ordering_shape((100, 1, 1)))(d_input)
    L = Conv2DTranspose(filters=128, kernel_size=3, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)
    L = Conv2DTranspose(filters=64, kernel_size=3, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)
    L = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)
    L = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    d_output = Activation('tanh')(L)
    return Model(d_input, d_output)


def get_pretrained_generator_model():
    model = dcgan_generator(bnmode=1)
    weight_file = 'generator_dcgan.h5'
    print('Loading dcgan generator weight file: ' + weight_file)
    model.load_weights(weight_file)
    return model


def get_fake_images(size=20000):
    model = get_pretrained_generator_model()
    print('Generating fake images using the dcgan mnist toy generator')
    noise = np.random.normal(size=(size, 100))
    images = model.predict(noise)
    images = scale_value(images, [0, 255.0])
    return images