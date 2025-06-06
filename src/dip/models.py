# models.py - Updated for TensorFlow 2.18.0
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp
from .glow_ops import *


tfd = tfp.distributions

class prior(keras.Model):
    """Defines the low dimensional distribution as Gaussian"""
    
    def __init__(self, latent_dim=64, **kwargs):
        super(prior, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        self.mu = self.add_weight(name='mu',
                                shape=(latent_dim,),
                                initializer='zeros',
                                trainable=True,
                                dtype=self.dtype)
        self.logsigma = self.add_weight(name='logsigma',
                                      shape=(latent_dim,),
                                      initializer='zeros',
                                      trainable=True,
                                      dtype=self.dtype)
        
        self.prior = tfd.MultivariateNormalDiag(self.mu, tf.math.exp(self.logsigma))

class bijective(keras.Model):
    def __init__(self, revnet_depth=3, **kwargs):
        super(bijective, self).__init__(**kwargs)
        self.depth = revnet_depth
        
        self.squeeze = upsqueeze(factor=2)
        self.revnets = [revnet(depth=self.depth, latent_model=True) 
                       for _ in range(6)]

    def call(self, x, reverse=False, training=None, **kwargs):
        # Reshape to 4D tensor for processing
        if len(x.shape) == 2:
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, [batch_size, 4, 4, 4])
        
        ops = self.revnets
        if reverse:
            ops = ops[::-1]

        objective = tf.constant(0.0, dtype=x.dtype)
        for op in ops:
            x, curr_obj = op(x, reverse=reverse, training=training)
            objective = objective + tf.cast(curr_obj, x.dtype)

        # Reshape back to 2D
        if not reverse:
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, [batch_size, 64])
        
        return x, objective

class injective(keras.Model):
    def __init__(self, revnet_depth=3, image_size=32, **kwargs):
        super(injective, self).__init__(**kwargs)
        self.depth = revnet_depth
        self.image_size = image_size
        
        self.squeeze = upsqueeze(factor=2)
        self.revnets = [revnet(depth=self.depth, latent_model=False) 
                       for _ in range(6)]
        
        self.inj_rev_steps = [revnet_step(layer_type='injective',
                                          latent_model=False, 
                                          activation='linear') 
                             for _ in range(6)]
        
    def call(self, x, reverse=False, training=None, **kwargs):
        if reverse:
            # Reshape from 1D latent to 4D tensor
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, [batch_size, 4, 4, 4])
            
        ops = [
            self.squeeze,
            self.revnets[0],
            self.inj_rev_steps[0],
            self.squeeze,
            self.revnets[1],
            self.inj_rev_steps[1],
            self.squeeze,
            self.revnets[2],
            self.inj_rev_steps[2],
            self.revnets[3],
            self.inj_rev_steps[3],
        ]

        if self.image_size == 64:
            ops += [
                self.inj_rev_steps[4],
                self.revnets[4],
                self.squeeze,
                self.inj_rev_steps[5],
                self.revnets[5]
            ]
   
        if reverse:
            ops = ops[::-1]

        objective = tf.constant(0.0, dtype=x.dtype)
        for op in ops:
            x, curr_obj = op(x, reverse=reverse, training=training)
            objective = objective + tf.cast(curr_obj, x.dtype)

        if not reverse:
            # Reshape to 1D latent vector
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, [batch_size, 4*4*4])

        return x, objective