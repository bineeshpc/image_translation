# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Problem statement
#
# <b>To build a Generative adversarial model(modified U-Net) which can generate artificial MRI images of different contrast levels from existing MRI scans<b>

# %%
# !nvidia-smi

# %%
# Install all the necessary libraries
# # !pip install matplotlib
# # !pip install scikit-image
# # !pip install glob
# # !pip install keras
# # !pip install imageio

# %%
# Import necessary libraries

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
import imageio
import glob
import os

# %% [markdown]
# ### Data Loading

# %%
pth = "/datasets/mri"
t1_path = f"{pth}/Tr1/TrainT1"
t2_path = f"{pth}/Tr2/TrainT2"

t1_images = os.listdir(t1_path)
t2_images = os.listdir(t2_path)
# Validating filename extraction
t1_images[2]

# %%
# Checking how an image looks like as a numpy array
imageio.imread(t1_path + '/' + t1_images[2])


# %%
# Definition to read and extract t1 and t2 train images
# Returns a numpy array
def get_numpy_array(ospath, imgfolder):
    """
    Reads images from a specified folder and returns them as a numpy array.

    Args:
        ospath (str): The path to the folder containing the images.
        imgfolder (list): A list of image file names to be read from the folder.

    Returns:
        numpy.ndarray: A numpy array containing the images.
    """
    img_list = []
    for file in imgfolder:
        img = imageio.imread(ospath + '/' + file)
        img_list.append(img)
    return np.asarray(img_list)  # converts list to a numpy array


# %%
# Data loading
t1 = get_numpy_array(t1_path, t1_images)
t2 = get_numpy_array(t2_path, t2_images)
print('T1 Shape: ', t1.shape)
print('T2 Shape: ', t2.shape)
print('Dataset size: ', t1.shape[0]+t2.shape[0])

# %% [markdown]
# It can bee seen that the number of images is very less for conventional training but an important advantage of GANs is its ability to generate images and create datasets out of it. Hence, instead of data augmentation, increasing the number of epochs should be fine.

# %% [markdown]
# ### Data Visualization
#

# %%
# Visualizing t1 image
def visualize_image(t1, n):
    """
    Visualizes a single image from a tensor.

    Parameters:
    t1 (np.ndarray): A numpy array containing image data.
    n (int): The index of the image to visualize.

    Returns:
    None
    """
    plt.figure(figsize=(2, 2))
    plt.imshow(t1[n], cmap='gray')
    plt.axis('off')
    plt.show()

visualize_image(t1, 2)

# %%
# VIsualizing t2 image
visualize_image(t2, 2)

# %% [markdown]
# ### Data Preprocessing

# %%
# Definition to normalize image
def normalize(img):
    """
    Normalize the input image.

    This function normalizes the input image by scaling its pixel values
    from the range [0, 255] to the range [-1.0, 1.0].

    Parameters:
    img (numpy.ndarray): The input image to be normalized.

    Returns:
    numpy.ndarray: The normalized image with pixel values in the range [-1.0, 1.0].
    """
    img = (img / 127.5) - 1.0  # Normalize the images to [-1.0, 1.0]
    return img


# %%
# Normalizing t1 and t2 images
t1 = normalize(t1)
t2 = normalize(t2)

# %%
# Visualizing t1 image
visualize_image(t1, 2)
# %%
# Visualizing t2 image
visualize_image(t2, 2)

# %%
# Resizing images to (64,64)
def resize_img(imgs):
    """
    Resize a batch of images to 64x64 pixels.

    Parameters:
    imgs (numpy.ndarray): A 4D numpy array of shape (N, H, W, C) where N is the number of images,
                          H is the height, W is the width, and C is the number of channels.

    Returns:
    numpy.ndarray: A 4D numpy array of shape (N, 64, 64, C) containing the resized images.
    """
    img_resized = np.zeros((imgs.shape[0], 64, 64))
    for index, img in enumerate(imgs):
        img_resized[index, :, :] = resize(img, (64, 64))
    return img_resized


# %%
t1_resized = resize_img(t1)
t2_resized = resize_img(t2)
print('T1 Shape: ', t1_resized.shape)
print('T2 Shape: ', t2_resized.shape)

# %%
# Reshape Images to (64, 64, 1) with float pixel values
t1_resized = t1_resized.reshape(
    t1_resized.shape[0], 64, 64, 1).astype('float32')
t2_resized = t2_resized.reshape(
    t2_resized.shape[0], 64, 64, 1).astype('float32')

# %%
# Batch and shuffle data
BATCH_SIZE = 32
t1_resized = tf.data.Dataset.from_tensor_slices(
    t1_resized).shuffle(t1.shape[0], seed=42).batch(BATCH_SIZE)
t2_resized = tf.data.Dataset.from_tensor_slices(
    t2_resized).shuffle(t2.shape[0], seed=42).batch(BATCH_SIZE)

# %%
# Randomly visualizing resized images
t1_sample = next(iter(t1_resized))

def visualize_sample(sample, n, title):
    """
    Visualizes a sample image from the provided dataset.

    Parameters:
    sample (array-like): The dataset containing the images.
    n (int): The index of the image to visualize.

    Returns:
    None
    """
    plt.figure(figsize=(2, 2))
    plt.imshow(t1_sample[n].numpy()[:, :, 0], cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

visualize_sample(t1_sample, 0, 'T1 MRI')

# %%
t2_sample = next(iter(t2_resized))
visualize_sample(t2_sample, 0, 'T2 MRI')


# %% [markdown]
# ### Model Building & Training

# %%
# Class to implement instance normalization on images
class InstanceNormalization(tf.keras.layers.Layer):
    """
    Instance Normalization Layer.
    This layer normalizes the input tensor for each instance in a batch independently.
    It computes the mean and variance of each instance and normalizes the input tensor
    using these statistics.
    Attributes:
        epsilon (float): A small float added to variance to avoid dividing by zero.
    Methods:
        build(input_shape):
            Creates the scale and offset weights for the layer.
        call(x):
            Applies instance normalization to the input tensor.
    Args:
        epsilon (float, optional): A small float added to variance to avoid dividing by zero. Defaults to 1e-5.
    """
    # Initialization of Objects
    def __init__(self, epsilon=1e-5):
        # calling parent's init
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        """
        Builds the layer by adding trainable weights for scaling and offsetting the input.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Adds:
            scale (tf.Variable): Trainable weight for scaling the input.
            offset (tf.Variable): Trainable weight for offsetting the input.
        """
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        """
        Applies instance normalization to the input tensor.

        Args:
            x (tf.Tensor): Input tensor to be normalized.

        Returns:
            tf.Tensor: Normalized tensor with the same shape as the input.
        """
        # Compute Mean and Variance, Axes=[1,2] ensures Instance Normalization
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


# %%
# Downsampling method: Leads to reduction in dimensions, achieved with convolutions
def downsample(filters, size, apply_norm=True):
    """
    Creates a downsampling layer using Conv2D, optional normalization, and LeakyReLU activation.

    Args:
        filters (int): The number of filters in the Conv2D layer.
        size (int): The size of the kernel in the Conv2D layer.
        apply_norm (bool, optional): Whether to apply normalization after the Conv2D layer. Defaults to True.

    Returns:
        tf.keras.Sequential: A Keras Sequential model containing the downsampling layers.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    # Add Conv2d layer
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    # Add Normalization layer
    if apply_norm:
        result.add(InstanceNormalization())
    # Add Leaky Relu Activation
    result.add(tf.keras.layers.LeakyReLU())
    return result


# %%
# Upsampling method: Achieved with Conv2DTranspose
def upsample(filters, size, apply_dropout=False):
    """
    Creates an upsampling layer using a transposed convolutional layer followed by instance normalization,
    optional dropout, and ReLU activation.

    Args:
        filters (int): The number of filters in the transposed convolutional layer.
        size (int): The size of the kernel for the transposed convolutional layer.
        apply_dropout (bool, optional): If True, a dropout layer is added after the normalization layer. Defaults to False.

    Returns:
        tf.keras.Sequential: A Keras Sequential model representing the upsampling layer.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    # Add Transposed Conv2d layer
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    # Add Normalization Layer
    result.add(InstanceNormalization())
    # Conditionally add Dropout layer
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    # Add Relu Activation Layer
    result.add(tf.keras.layers.ReLU())
    return result


# %%
# Unet Generator: A combination of Convolution + Transposed Convolution Layers
def unet_generator():
    """
    Creates a U-Net generator model for image translation.
    The U-Net architecture consists of a series of downsampling and upsampling layers with skip connections between
    corresponding layers in the downsampling and upsampling paths.
    Returns:
        tf.keras.Model: A U-Net generator model.
    """
    down_stack = [
        downsample(128, 4, False),  # (bs, 32, 32, 128)
        downsample(256, 4),  # (bs, 16, 16, 256)
        downsample(256, 4),  # (bs, 8, 8, 256)
        downsample(256, 4),  # (bs, 4, 4, 256)
        downsample(256, 4),  # (bs, 2, 2, 256)
        downsample(256, 4)  # (bs, 1, 1, 256)
    ]
    up_stack = [
        upsample(256, 4, True),  # (bs, 2, 2, 256)
        upsample(256, 4, True),  # (bs, 4, 4, 256)
        upsample(256, 4),  # (bs, 8, 8, 256)
        upsample(256, 4),  # (bs, 16, 16, 256)
        upsample(128, 4)  # (bs, 32, 32, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 64, 64, 1)
    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=[64, 64, 1])
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


# %% [markdown]
# #### Creating generator

# %%
generator_g = unet_generator()
generator_f = unet_generator()

# %%
generator_g.summary()


# %% [markdown]
# #### Creating Discriminator

# %%
# Discriminators only contain Convolutional Layers and no Transposed Convolution is used
def discriminator():
    """
    Builds a discriminator model for image translation.
    The discriminator is a convolutional neural network that classifies 
    whether the input image is real or generated. It uses several 
    downsampling layers, padding layers, and convolutional layers.
    Returns:
        tf.keras.Model: A Keras Model representing the discriminator.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    # add input layer of size (64, 64, 1)
    inp = tf.keras.layers.Input(shape=[64, 64, 1], name='input_image')
    x = inp

    # add downsampling step here
    down1 = downsample(128, 4, False)(x)  # (bs, 32, 32, 128)
    down2 = downsample(256, 4)(down1)  # (bs, 16, 16, 128)
    # add a padding layer here
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)  # (bs, 18, 18, 256)

    # implement a concrete downsampling layer here
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 15, 15, 512)
    norm1 = InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    # apply zero padding layer
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 17, 17, 512)

    # add a last pure 2D Convolution layer
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2)  # (bs, 14, 14, 1)
    return tf.keras.Model(inputs=inp, outputs=last)


# %%
discriminator_x = discriminator()
discriminator_y = discriminator()

# %%
discriminator_x.summary()

# %% [markdown]
# ### Model Training

# %%
# Verify output of untrained Generator models which should be a random noise
conv_to_t2 = generator_g(t1_sample)
conv_to_t1 = generator_f(t2_sample)
plt.figure(figsize=(4, 4))

imgs = [t1_sample, conv_to_t2, t2_sample, conv_to_t1]
title = ['t1_image', 'conv_to_t2', 't2_image', 'conv_to_t1']

for i in range(len(imgs)):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    plt.imshow(imgs[i][0].numpy()[:, :, 0], cmap='gray')
    plt.axis('off')
plt.show()

# %% [markdown]
# #### Creating loss objects and functions

# %%
# Classification loss of discriminator: BCE Loss
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# %%
# Discriminator Loss: Real Loss + Fake Loss)
def discriminator_loss(real, generated):
    """
    Computes the discriminator loss for a GAN.

    The loss is calculated as the mean of the loss for real images and the loss for generated images.
    The real images are compared against a tensor of ones (indicating they are real), and the generated
    images are compared against a tensor of zeros (indicating they are fake).

    Args:
        real (tf.Tensor): The discriminator's output on real images.
        generated (tf.Tensor): The discriminator's output on generated (fake) images.

    Returns:
        tf.Tensor: The total discriminator loss, averaged over the real and generated losses.
    """
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5  # mean of losses


# %%
# Generator Loss: Discriminator loss on generated data
def generator_loss(generated):
    """
    Computes the generator loss for a generated image.

    Args:
        generated (tf.Tensor): The generated image tensor.

    Returns:
        tf.Tensor: The computed loss value.
    """
    return loss_obj(tf.ones_like(generated), generated)


# %%
# Cycle Loss: When we use both of Generators sequentially on a Input Image, we get Cycle Image and
# the L1 Loss between these two is called Cycle Loss.
def calc_cycle_loss(real_image, cycled_image):
    """
    Calculates the cycle consistency loss for image translation models.

    Cycle consistency loss ensures that the translated image, when translated back to the original domain, 
    is similar to the original image. This helps in maintaining the content and structure of the original image.

    Args:
        real_image (tf.Tensor): The original image tensor.
        cycled_image (tf.Tensor): The image tensor obtained after translating the real image to another domain 
                                  and then translating it back to the original domain.

    Returns:
        tf.Tensor: The cycle consistency loss value, scaled by a factor of 10.
    """
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return 10.0 * loss1


# %%
# Identity Loss: When we provide input image to the Generator such that no translation is needed
# because the image is already transformed, here also we take L1 Loss between Input and Output Image.
def identity_loss(real_image, same_image):
    """
    Computes the identity loss between the real image and the same image.

    Identity loss is calculated as the mean absolute difference between the 
    real image and the same image, scaled by a factor of 0.5.

    Args:
        real_image (tf.Tensor): The original image tensor.
        same_image (tf.Tensor): The image tensor that should be identical to the original image.

    Returns:
        tf.Tensor: The computed identity loss.
    """
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return 0.5*loss


# %%
# Leveraging Adam Optimizer to smoothen gradient descent
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# %%
# Training model for 300 epochs
EPOCHS = 300

# %% [markdown]
# #### Checkpoint Initialization

# %%
# Initialize checkpoints to save models
checkpoint_path = "./Trained_Model"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


# %%
# Definition to show generated output whilst training
def generate_images(model1, test_input1, model2, test_input2):
    """
    Generates and displays images using two models and their respective test inputs.
    Args:
        model1 (tf.keras.Model): The first trained model for generating images.
        test_input1 (tf.Tensor): The input tensor for the first model.
        model2 (tf.keras.Model): The second trained model for generating images.
        test_input2 (tf.Tensor): The input tensor for the second model.
    Displays:
        A matplotlib figure with four subplots:
            - The first subplot shows the first input image.
            - The second subplot shows the image generated by the first model.
            - The third subplot shows the second input image.
            - The fourth subplot shows the image generated by the second model.
    Saves:
        A PNG file named 'image_at_epoch_{:04d}.png' where {:04d} is the epoch number.
    """
    prediction1 = model1(test_input1)
    prediction2 = model2(test_input2)
    plt.figure(figsize=(8, 4))
    display_list = [test_input1[0], prediction1[0],
                    test_input2[0], prediction2[0]]
    title = ['Input Image', 'Generated Image',
             'Input Image', 'Generated Image']
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i].numpy()[:, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# %% [markdown]
# #### Defining Training Function

# %%
@tf.function
def train_step(real_x, real_y):
    """
    Performs a single training step for the cycle-consistent generative adversarial network (CycleGAN).
    Args:
        real_x (tf.Tensor): Real images from domain X.
        real_y (tf.Tensor): Real images from domain Y.
    Returns:
        None
    """
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(
            real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = BCE loss + cycle loss + identity loss
        total_gen_g_loss = gen_g_loss + \
            total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + \
            total_cycle_loss + identity_loss(real_x, same_x)

        # Discriminator's loss
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(
        total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(
        total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(
        disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(
        disc_y_loss, discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(
        zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(
        zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(
        zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(
        zip(discriminator_y_gradients, discriminator_y.trainable_variables))


# %%
# Verify if GPU is set as the default device for training
tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(log_device_placement=True))

# %% [markdown]
# ### Training the CycleGAN

# %%
# %%time
for epoch in range(1, EPOCHS+1):
    for image_x, image_y in tf.data.Dataset.zip((t1_resized, t2_resized)):
        train_step(image_x, image_y)
    generate_images(generator_g, t1_sample, generator_f, t2_sample)
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch', epoch, 'at', ckpt_save_path)

# %% [markdown]
# ### Generating GIF to VIsualize Results

# %%
anim_file = 'cyclegan_mri.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)



# %%
from IPython.display import Image
Image(open(anim_file, 'rb').read())
