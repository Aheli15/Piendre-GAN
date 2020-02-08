import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model

opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
def define_gan(g_model, d_model, g_image_shape, d_image_shape):

	d_model.trainable = False  # make weights in the discriminator not trainable
	in_src = Input(shape=g_image_shape)	
	gen_out = g_model(in_src)  # connect the source image to the generator input
	dis_out = d_model([in_src, gen_out])  # connect the source input and generator output to the discriminator input
	
	model = Model(in_src, [dis_out, gen_out])  # src image as input, generated image and classification output
	opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model