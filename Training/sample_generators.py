import numpy as np
from utils import load_data

def generate_real_samples(img_list, n_samples, patch_shape):
    dataset = load_data(img_list, n_samples)
    trainA, trainB = dataset
    y = np.ones((len(trainA), patch_shape, patch_shape, 1))
    return [trainA, trainB], y

def generate_fake_samples(g_model, samples, patch_shape):
	X = g_model.predict(samples)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y