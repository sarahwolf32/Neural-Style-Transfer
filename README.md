# Neural_Style_Transfer
A Tensorflow implementation 

This is an implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys, et al.

To use:

1. Download a pretrained VGG19 model [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat), and place it in the project directory.
2. Replace images/content.jpg with a photo of your choice (600x600).
3. Replace images/style.jpg with an art image of your choice (600x600).
4. Navigate into the project directory in terminal
5. Type 'python neural_style_transfer.py'
6. That's it! Training could take several hours. Intermediate images will be generated at regular intervals.
7. Check the 'output' folder for results.
