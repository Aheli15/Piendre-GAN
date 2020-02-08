import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def load_data(img_list, batch_size):
    A = []
    B = []
    ix = random.sample(range(len(img_list)), batch_size)
    for i in ix:
        pic = plt.imread(img_list[i], format="np.uint8")
        if pic.ndim == 3:
            bw = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            bw = cv2.resize(bw, (256,256)).reshape((256,256,1))
            A.append(bw)
            B.append(cv2.resize(pic, (256,256)))
        else:
            dataset = load_data(img_list, batch_size)
            return dataset
    dataset = [np.asarray(A, dtype='float32')/255,np.asarray(B, dtype='float32')/255]
    return dataset

def plot_generated_images(generator, img_list):

    bw_img, _ = load_data(img_list, 10)
      
    generated_images = generator.predict(bw_img)
    generated_images = generated_images.reshape(10, 256, 256, 3)

    fig, axes = plt.subplots(2,5, figsize=(18,8))
    for i, ax in zip(range(10), axes.flat):
        img = generated_images[i]
        ax.imshow(img)
    plt.show()