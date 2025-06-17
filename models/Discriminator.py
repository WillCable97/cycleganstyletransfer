import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
import matplotlib.pyplot as plt
from pathlib import Path
from cycleganstyletransfer.config import DATA_DIR

# Constants
BATCH_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
data_dir = DATA_DIR / "raw"

# Weight initializer
weight_initializer = RandomNormal(stddev=0.02)

# Dataset creation and preprocessing
def normalize(x):
    x = tf.cast(x, tf.float32)
    return (x / 127.5) - 1

def random_jitter(image):
    # Resize to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Random crop back to 256 x 256 x 3
    image = tf.image.random_crop(image, size=[256, 256, 3])
    # Random mirroring
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

# Create datasets
monet_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir / "Monet",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode=None
)

images_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir / "Images",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode=None
)

# Apply preprocessing
monet_ds = monet_ds.map(
    preprocess_image_train, 
    num_parallel_calls=tf.data.AUTOTUNE
).cache().shuffle(1000).repeat()

images_ds = images_ds.map(
    preprocess_image_train, 
    num_parallel_calls=tf.data.AUTOTUNE
).cache().shuffle(1000).repeat()

# Create iterators
monet_iterator = iter(monet_ds)
images_iterator = iter(images_ds)

# Discriminator implementation
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
    """Build the discriminator model for CycleGAN"""
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
    
    return Model(inputs, x, name='discriminator')

# Visualization function
def visualize_batch(monet_batch, image_batch):
    plt.figure(figsize=(10, 5))
    
    # Plot Monet
    plt.subplot(1, 2, 1)
    plt.imshow((monet_batch[0] + 1) * 0.5)  # Convert from [-1,1] to [0,1]
    plt.title('Monet')
    plt.axis('off')
    
    # Plot Photo
    plt.subplot(1, 2, 2)
    plt.imshow((image_batch[0] + 1) * 0.5)
    plt.title('Photo')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Test the setup
def test_setup():
    # Test dataset
    print("Testing dataset...")
    monet_batch = next(monet_iterator)
    image_batch = next(images_iterator)
    print("Monet batch shape:", monet_batch.shape)
    print("Image batch shape:", image_batch.shape)
    print("Monet batch range:", tf.reduce_min(monet_batch), "to", tf.reduce_max(monet_batch))
    visualize_batch(monet_batch, image_batch)
    
    # Test discriminator
    print("\nTesting discriminator...")
    discriminator = build_discriminator()
    monet_output = discriminator(monet_batch)
    image_output = discriminator(image_batch)
    print("Discriminator output shape:", monet_output.shape)
    print("Discriminator output range:", tf.reduce_min(monet_output), "to", tf.reduce_max(monet_output))
    
    # Print model summary
    print("\nDiscriminator summary:")
    discriminator.summary()

# Run the test
if __name__ == "__main__":
    test_setup()