import tensorflow as tf
from utils import *
from losses import *
from combined_model import *
from sample_generators import *
from Generator_Discriminator_Models import *

folder = '/data/train/'
img_list = glob.glob(folder)

src_shape = (256,256,1)
tar_shape = (256,256,3)

discriminator = define_discriminator(src_shape, tar_shape)
generator = define_generator(src_shape)

opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
generator.compile(loss=generator_loss, optimizer=opt, loss_weights=[0.5])
discriminator.compile(loss=discriminator_loss, optimizer=opt, loss_weights=[0.5])

gan_model = define_gan(generator, discriminator, src_shape, tar_shape)

def train(discriminator, generator, gan_model, img_list, n_epochs=80, n_batch=1, n_patch=16):
    bat_per_epo = int(len(img_list) / n_batch)
    n_steps = bat_per_epo * n_epochs
    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(img_list, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(generator, X_realA, n_patch)
        d_loss1 = discriminator.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = discriminator.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        if (i%10==0):
            plot_generated_images(generator, img_list)
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

train(discriminator, generator, gan_model, img_list)

