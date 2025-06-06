# Unet_util.py - Updated for TensorFlow 2.18.0
import tensorflow as tf
from tensorflow import keras


class Conv_block(keras.layers.Layer):
    def __init__(self, num_filters, **kwargs):
        super(Conv_block, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(num_filters, (3, 3), padding="same", use_bias=False, activation=None)
        self.conv2 = keras.layers.Conv2D(num_filters, (3, 3), padding="same", use_bias=False, activation=None)
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation("relu")
        self.act2 = keras.layers.Activation("relu")

    def call(self, x, training=None, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        return x

class Unet(keras.layers.Layer):
    def __init__(self, output_channels, **kwargs):
        super(Unet, self).__init__(**kwargs)
        
        self.num_filters = [32, 64]
        
        self.conv_blocks1 = [Conv_block(num_filters=f) 
                           for f in self.num_filters]
        
        self.conv_blocks2 = [Conv_block(num_filters=f) 
                           for f in self.num_filters[::-1]]
            
        self.conv_block_bridge = Conv_block(self.num_filters[-1])
        self.maxpool = keras.layers.MaxPool2D((2, 2))
        self.upsample = keras.layers.UpSampling2D((2, 2))
        self.concat = keras.layers.Concatenate()
        self.conv = keras.layers.Conv2D(output_channels, (1, 1), padding="same", use_bias=False)
        self.act = keras.layers.Activation("sigmoid")

    def call(self, x, training=None, **kwargs):
        skip_x = []
        
        # Encoder
        for i in range(len(self.num_filters)):
            x = self.conv_blocks1[i](x, training=training)
            skip_x.append(x)
            x = self.maxpool(x)
        
        # Bridge
        x = self.conv_block_bridge(x, training=training)

        skip_x = skip_x[::-1]
        # Decoder
        for i in range(len(self.num_filters)):
            x = self.upsample(x)
            xs = skip_x[i]
            x = self.concat([x, xs])
            x = self.conv_blocks2[i](x, training=training)

        # Output
        x = self.conv(x)
        x = self.act(x)
        return x