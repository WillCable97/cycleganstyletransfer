import tensorflow as tf
import keras
from keras.layers import Conv2D, LeakyReLU, Input, BatchNormalization
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
import warnings

weight_initializer = RandomNormal(stddev=0.02)

"""*******************************DESCRIMINATOR*******************************"""
def DescriminatorBlock(filters:int, kernel_size:int, strides:int) -> tf.Tensor:
    ret_block =  Sequential()
    ret_block.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=weight_initializer))
    ret_block.add(BatchNormalization(axis=-1))
    ret_block.add(LeakyReLU(0.2))
    return ret_block


def Descriminator(input_shape:tuple, kernel_size:int=3, strides:int=1) -> keras.Model:
    discriminator_input = Input(shape=input_shape)
    ret_block = discriminator_input
    ret_block = DescriminatorBlock(64, kernel_size, strides)(ret_block)
    ret_block = DescriminatorBlock(128, kernel_size, strides)(ret_block)
    ret_block = DescriminatorBlock(256, kernel_size, strides)(ret_block)
    ret_block = DescriminatorBlock(512, kernel_size, strides)(ret_block)
    return Model(discriminator_input, ret_block)




import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

weight_initializer = RandomNormal(stddev=0.02)

def discriminator_block(x, filters, kernel_size=4, strides=2, padding='same'):
    """Single block of the discriminator"""
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=weight_initializer
    )(x)
    x = LeakyReLU(0.2)(x)
    return x

def build_discriminator(input_shape=(256, 256, 3)):
    """Build the discriminator model"""
    inputs = Input(shape=input_shape)
    
    # First layer without stride
    x = discriminator_block(inputs, 64, strides=1)
    
    # Downsampling layers
    x = discriminator_block(x, 128)
    x = discriminator_block(x, 256)
    x = discriminator_block(x, 512)
    
    # Final layer
    x = Conv2D(
        filters=1,
        kernel_size=4,
        strides=1,
        padding='same',
        kernel_initializer=weight_initializer
    )(x)
    
    # Flatten and add sigmoid for binary classification
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs, outputs, name='discriminator')