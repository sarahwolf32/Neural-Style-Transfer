
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

# to use: 
#   from load_vgg import LoadVGG
#   loader = LoadVGG()

class LoadVGG():

    """
        Returns a model for the purpose of 'painting' the picture.
        Takes only the convolution layer weights and wrap using the TensorFlow
        Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
        the paper indicates that using AveragePooling yields better results.
        The last few fully connected layers are not used.

        Here is the detailed configuration of the VGG model:
            0 is conv1_1 (3, 3, 3, 64)
            1 is relu
            2 is conv1_2 (3, 3, 64, 64)
            3 is relu    
            4 is maxpool
            5 is conv2_1 (3, 3, 64, 128)
            6 is relu
            7 is conv2_2 (3, 3, 128, 128)
            8 is relu
            9 is maxpool
            10 is conv3_1 (3, 3, 128, 256)
            11 is relu
            12 is conv3_2 (3, 3, 256, 256)
            13 is relu
            14 is conv3_3 (3, 3, 256, 256)
            15 is relu
            16 is conv3_4 (3, 3, 256, 256)
            17 is relu
            18 is maxpool
            19 is conv4_1 (3, 3, 256, 512)
            20 is relu
            21 is conv4_2 (3, 3, 512, 512)
            22 is relu
            23 is conv4_3 (3, 3, 512, 512)
            24 is relu
            25 is conv4_4 (3, 3, 512, 512)
            26 is relu
            27 is maxpool
            28 is conv5_1 (3, 3, 512, 512)
            29 is relu
            30 is conv5_2 (3, 3, 512, 512)
            31 is relu
            32 is conv5_3 (3, 3, 512, 512)
            33 is relu
            34 is conv5_4 (3, 3, 512, 512)
            35 is relu
            36 is maxpool
            37 is fullyconnected (7, 7, 512, 4096)
            38 is relu
            39 is fullyconnected (1, 1, 4096, 4096)
            40 is relu
            41 is fullyconnected (1, 1, 4096, 1000)
            42 is softmax
    """

    # Mean to subract from the VGG input. 
    # These values were used for normalization when the VGG was trained, and are important.
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    def _weights(self, layer, expected_layer_name, vgg_layers):
        """
        Returns the weights and biases from the VGG model for a given layer
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(self, conv2d_layer):
        """
        Return the RELU function wrapped over a Tensorflow layer. 
        Expects a Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(self, prev_layer, layer, layer_name, vgg_layers):
        """
        Return the Conv2D layer using the weights, biases from the VGG model at 'layer'.
        """
        W, b = self._weights(layer, layer_name, vgg_layers)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1,1,1,1], padding='SAME') + b

    def _conv2d_relu(self, prev_layer, layer, layer_name, vgg_layers):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG model at 'layer'.
        """
        return self._relu(self._conv2d(prev_layer, layer, layer_name, vgg_layers))

    def _avgpool(self, prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def load_vgg_model(self, path, image_height, image_width, color_channels):

        vgg = scipy.io.loadmat(path)
        vgg_layers = vgg['layers']

        # Constructs the graph model.
        graph = {}
        graph['input']   = tf.Variable(np.zeros((1, image_height, image_width, color_channels)), dtype = 'float32')
        graph['conv1_1']  = self._conv2d_relu(graph['input'], 0, 'conv1_1', vgg_layers)
        graph['conv1_2']  = self._conv2d_relu(graph['conv1_1'], 2, 'conv1_2', vgg_layers)
        graph['avgpool1'] = self._avgpool(graph['conv1_2'])
        graph['conv2_1']  = self._conv2d_relu(graph['avgpool1'], 5, 'conv2_1', vgg_layers)
        graph['conv2_2']  = self._conv2d_relu(graph['conv2_1'], 7, 'conv2_2', vgg_layers)
        graph['avgpool2'] = self._avgpool(graph['conv2_2'])
        graph['conv3_1']  = self._conv2d_relu(graph['avgpool2'], 10, 'conv3_1', vgg_layers)
        graph['conv3_2']  = self._conv2d_relu(graph['conv3_1'], 12, 'conv3_2', vgg_layers)
        graph['conv3_3']  = self._conv2d_relu(graph['conv3_2'], 14, 'conv3_3', vgg_layers)
        graph['conv3_4']  = self._conv2d_relu(graph['conv3_3'], 16, 'conv3_4', vgg_layers)
        graph['avgpool3'] = self._avgpool(graph['conv3_4'])
        graph['conv4_1']  = self._conv2d_relu(graph['avgpool3'], 19, 'conv4_1', vgg_layers)
        graph['conv4_2']  = self._conv2d_relu(graph['conv4_1'], 21, 'conv4_2', vgg_layers)
        graph['conv4_3']  = self._conv2d_relu(graph['conv4_2'], 23, 'conv4_3', vgg_layers)
        graph['conv4_4']  = self._conv2d_relu(graph['conv4_3'], 25, 'conv4_4', vgg_layers)
        graph['avgpool4'] = self._avgpool(graph['conv4_4'])
        graph['conv5_1']  = self._conv2d_relu(graph['avgpool4'], 28, 'conv5_1', vgg_layers)
        graph['conv5_2']  = self._conv2d_relu(graph['conv5_1'], 30, 'conv5_2', vgg_layers)
        graph['conv5_3']  = self._conv2d_relu(graph['conv5_2'], 32, 'conv5_3', vgg_layers)
        graph['conv5_4']  = self._conv2d_relu(graph['conv5_3'], 34, 'conv5_4', vgg_layers)
        graph['avgpool5'] = self._avgpool(graph['conv5_4'])
        return graph


        
