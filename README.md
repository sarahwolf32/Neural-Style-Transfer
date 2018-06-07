# Neural_Style_Transfer

This is an implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys, et al.

To use:

1. Download a pretrained VGG19 model [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat), and place it in the project directory.
2. Replace images/content.jpg with a photo of your choice (600x600).
3. Replace images/style.jpg with an art image of your choice (600x600).
4. Navigate into the project directory in terminal
5. Type 'python neural_style_transfer.py'
6. That's it! Training could take several hours. Intermediate images will be generated at regular intervals.
7. Check the 'output' folder for results.

Below is a piece of art I created with this code. Full disclosure: some of the results I did not post were quite terrible looking, so do not be discouraged if your first few attempts do not come out as you had hoped. I recommend experimenting a bit to get a feel for what works well and what doesn't. The same style image can have a very different effect on different content images, or with different hyperparameters.

Style Image:
![Style Image](images/style.jpg)

Content Image:
![Content Image](images/content.jpg)

Resulting Generated Image:
![Generated Image](finished/tree_990.png)
