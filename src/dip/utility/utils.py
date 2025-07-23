# utils.py - Updated for TensorFlow 2.18.0

import cv2
import numpy as np
import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


@tf.function
def train_step_mse(sample, inj_model, optimizer_inj):
    """MSE training of the injective sub-network"""
    with tf.GradientTape() as tape:
        z, _ = inj_model(sample, reverse=False, training=True)
        recon, _ = inj_model(z, reverse=True, training=True)
        mse_loss = tf.reduce_mean(tf.square(sample - recon))
        
    gradients = tape.gradient(mse_loss, inj_model.trainable_variables)
    optimizer_inj.apply_gradients(zip(gradients, inj_model.trainable_variables))
    return mse_loss

@tf.function
def train_step_ml(sample, bij_model, pz, optimizer_bij):
    """ML training of the bijective sub-network"""

    with tf.GradientTape() as tape:
        latent_sample, obj = bij_model(sample, reverse=False)
        p = -tf.reduce_mean(pz.prior.log_prob(latent_sample))
        j = -tf.reduce_mean(obj) # Log-det of Jacobian
        loss =  p + j
        variables = tape.watched_variables()
        grads = tape.gradient(loss, variables)
        optimizer_bij.apply_gradients(zip(grads, variables))

    return loss

def PSNR(x_true, x_pred):
    """Calculate PSNR between true and predicted images"""
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += psnr(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min())
    return s / np.shape(x_pred)[0]

def SSIM(x_true, x_pred):
    """Calculate SSIM between true and predicted images"""
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += ssim(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min(),
                  channel_axis=-1 if len(x_true[i].shape) == 3 else None)
    return s / np.shape(x_pred)[0]

def Dataset_preprocessing(dataset='mnist', img_size=32, batch_size=64, ood_experiment=False):
    """Preprocess dataset for training"""
    if dataset == 'mnist':
        (train_images, train_labels), (test_images, _) = tf.keras.datasets.mnist.load_data()
        
        if ood_experiment:
            np.random.seed(0)
            sorted_labels_ind = np.argsort(train_labels)
            sorted_labels = train_labels[sorted_labels_ind]
            test_ind = np.where(sorted_labels == 6)[0][0]
            
            train_images = train_images[sorted_labels_ind, :, :]
            test_images = train_images[test_ind:]
            train_images = train_images[:test_ind]
            np.random.shuffle(test_images)
            np.random.shuffle(train_images)

        train_images = np.expand_dims(train_images, axis=3)
        test_images = np.expand_dims(test_images, axis=3)
        
    elif dataset == 'ellipses':
        images = np.load('datasets/ellipses_64.npy')
        train_images, test_images = np.split(images, [55000])

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
    
    needs_resize = train_images.shape[1] != img_size

    def preprocess_image(image):
        """Resize and normalize images using TensorFlow operations."""
        if needs_resize:
            image = tf.image.resize(image, [img_size, img_size])
        
        image = tf.cast(image, tf.float32)
        # Normalize per-image to [-1, 1]
        image -= (tf.reduce_max(image) + tf.reduce_min(image)) / 2.0
        max_val = tf.reduce_max(tf.abs(image))
        image = tf.cond(max_val > 0.0, lambda: image / max_val, lambda: image)
        return image

    train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    SHUFFLE_BUFFER_SIZE = 256
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(
        batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, test_dataset