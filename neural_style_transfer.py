import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf # Must import TF after Scipy 
from PIL import Image
from load_vgg import LoadVGG
from loss import Loss

STYLE_IMAGE = 'images/style.jpg'
CONTENT_IMAGE = 'images/content.jpg'
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
COLOR_CHANNELS = 3
OUTPUT_DIR = 'output/'

### Algorithm constants

NOISE_RATIO = 0.6 # Amount of noise to mix into the content image
ITERATIONS = 5000 # Number of training iterations
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat' # Pre-trained CNN model

# Initialize helper classes
model_loader = LoadVGG()
loss_calculator = Loss()

### Helper functions

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(low = -20, high = 20, size=(1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image

def load_image(path):
    image = scipy.misc.imread(path)

    # add an extra dimension so its compatible with the convnet
    image = np.reshape(image, ((1,) + image.shape))

    # subtract the mean from VGG-training to input into VGG
    image = image - model_loader.MEAN_VALUES
    return image

def save_image(path, image):

    # add back the VGG-training mean
    image = image + model_loader.MEAN_VALUES

    # remove the first useless dimension
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def save_current_generated_image(sess, model):

    # create an output folder if none exists
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    # save generated image
    mixed_image = loss_calculator.current_input(sess, model)
    filename = '%s%d.png' % (OUTPUT_DIR, itr)
    save_image(filename, mixed_image)



### Prepare session

sess = tf.InteractiveSession()

# load images
content_image = load_image(CONTENT_IMAGE)
style_image = load_image(STYLE_IMAGE)

# load model
model = model_loader.load_vgg_model(VGG_MODEL, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)

# randomly generate initial generated image
input_image = generate_noise_image(content_image)

# optimize
total_loss = loss_calculator.total_loss(sess, model, content_image, style_image)
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(total_loss)

# monitor
summary_op = tf.summary.scalar('loss', total_loss)
writer = tf.summary.FileWriter('summaries', graph=tf.get_default_graph())
#step_var = tf.train.create_global_step()

### Run

sess.run(tf.global_variables_initializer())
loss_calculator.assign_to_input(sess, model, input_image)


for itr in range(ITERATIONS):
    sess.run(train_step)

    # update summary every 5 iterations
    if itr % 5 == 0:
        print("writing summary!")
        summary = sess.run(summary_op)
        writer.add_summary(summary, global_step=itr)
        writer.flush()

    # save image every 50 iterations
    if itr % 50 == 0:
        save_current_generated_image(sess, model)
        print("Iteration %d" % (itr))
        print("Cost: ", sess.run(total_loss))
        
        