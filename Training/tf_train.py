import tensorflow as tf
from utils import *
from losses import *
from Generator_Discriminator_Models import *

src_shape = (256,256,1)
tar_shape = (256,256,3)

discriminator = define_discriminator(src_shape, tar_shape)
generator = define_generator(src_shape)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

EPOCHS = 50

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

def fit(img_list, epochs, batch_size):
    for epoch in range(epochs):
        times = len(img_list)//batch_size
        for t in range(times):
            train_ds = load_data(img_list, batch_size)                
            #for n, images in enumerate(train_ds):
            input_image, target = train_ds
            train_step(input_image, target, epoch)
        plot_generated_images(generator, img_list)

fit(img_list, EPOCHS, 1)