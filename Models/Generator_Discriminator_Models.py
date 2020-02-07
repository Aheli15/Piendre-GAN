from Encoder_Decoder_Blocks import *
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Activation, Concatenate, BatchNormalization, Conv2DTranspose, Dropout
from tensorflow.keras.utils import plot_model

def define_discriminator(image_shape_1 = (256,256,1), image_shape_2 = (256,256,3)):

	init = tf.random_normal_initializer(0., 0.02)	
	in_src_image = Input(shape=image_shape_1) # source image input	
	in_target_image = Input(shape=image_shape_2) # target image input
	merged = Concatenate()([in_src_image, in_target_image]) 	# concatenate images channel-wise

	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d) # patch output
	patch_out = Activation('sigmoid')(d)
	
	model = Model([in_src_image, in_target_image], patch_out)
	return model

def define_generator(image_shape=(256,256,1)):

	init = tf.random_normal_initializer(0., 0.02)
	in_image = Input(shape=image_shape)
 
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = encoder_block(in_image, 64, batchnorm=False)
	e2 = encoder_block(e1, 128)
	e3 = encoder_block(e2, 256)
	e4 = encoder_block(e3, 512)
	e5 = encoder_block(e4, 512)
	e6 = encoder_block(e5, 512)
	e7 = encoder_block(e6, 512)
	
	# bottleneck, no batch norm and relu
	b = Conv2D(512, 4, strides=2, padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	
	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	
	g = Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	
	model = Model(in_image, out_image)
	return model