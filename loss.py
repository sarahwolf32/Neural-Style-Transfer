import numpy as np
import tensorflow as tf

class Loss():

    ### Config

    STYLE_LAYER_WEIGHTS = {
        'conv1_1': 0.5,
        'conv2_1': 1.0,
        'conv3_1': 1.5,
        'conv4_1': 3.0,
        'conv5_1': 4.0
    }

    CONTENT_LAYER = 'conv4_2'
    INPUT_LAYER = 'input'

    CONTENT_WEIGHT = 5
    STYLE_WEIGHT = 100

    ### Compute loss

    def total_loss(self, sess, model, content_image, style_image):

        # construct content loss using content image
        sess.run(tf.global_variables_initializer())
        self.assign_to_input(sess, model, content_image)
        content_loss = self._content_loss(sess, model)

        # construct style loss using style image
        self.assign_to_input(sess, model, style_image)
        style_loss = self._style_loss(sess, model)

        # compute total loss
        total_loss = self.CONTENT_WEIGHT * content_loss + self.STYLE_WEIGHT * style_loss
        return total_loss

    def assign_to_input(self, sess, model, image):
        sess.run(model[self.INPUT_LAYER].assign(image))

    def current_input(self, sess, model):
        return sess.run(model[self.INPUT_LAYER])


    ### PRIVATE 

    ### Content loss 

    def _content_loss(self, sess, model):

        # generated image activation
        x = sess.run(model[self.CONTENT_LAYER])

        # content image activation
        p = model[self.CONTENT_LAYER]

        # N is the number of channels (at layer l)
        N = p.get_shape().as_list()[3]

        # M is the size (height * width) at layer l
        M = p.get_shape().as_list()[1] * p.get_shape().as_list()[2]

        content_loss = (1. / (4. * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
        return content_loss

    ### Style loss 

    def _gram_matrix(self, activation_tensor, N, M):

        # unroll activation into one long vector per channel
        unrolled_activation = tf.reshape(activation_tensor, (M, N))

        # compute gram matrix
        return tf.matmul(tf.transpose(unrolled_activation), unrolled_activation)

    def _layer_style_loss(self, sess, model, layer):

        # activation at layer l of original image.
        a = sess.run(model[layer])

        # activation at layer l of generated image.
        x = model[layer]

        # num channels in activation
        N = a.shape[3]

        # activation height * width
        M = a.shape[2] * a.shape[1]

        # A is the style representation of the original image (at layer l)
        A = self._gram_matrix(a, N, M)

        # G is the style representation of the generated image (at layer l)
        G = self._gram_matrix(x, N, M)

        result = (1. / (4. * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    def _style_loss(self, sess, model):

        # Compute each layer's style loss and weight
        style_layer_names = self.STYLE_LAYER_WEIGHTS.keys()
        style_layer_losses = [self._layer_style_loss(sess, model, l) for l in style_layer_names]
        style_layer_weights = self.STYLE_LAYER_WEIGHTS.values()

        # compute weighted sum of losses 
        loss = np.dot(style_layer_losses, style_layer_weights)

        return loss





