import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Activation, Concatenate, BatchNormalization, Conv2DTranspose, Dropout

def encoder_block(layer_in, n_filters, batchnorm=True):	
	init = tf.random_normal_initializer(0., 0.02) 
	g = Conv2D(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)	
	if batchnorm:
		g = BatchNormalization()(g, training=True)	
	g = LeakyReLU(alpha=0.2)(g)
	return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):	
	init = tf.random_normal_initializer(0., 0.02)	
	g = Conv2DTranspose(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)	
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Concatenate()([g, skip_in])
	g = Activation('relu')(g)
	return g