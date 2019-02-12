# artsandcrafts2

Autoencoder for upsampling images based on Manu Mathew Thomas's autoencoder code. 

# in model.py: 

We first resize the images to the ground truth size, (128 × 90) which causes the images to become pixelated, and then we train the network to reduce the pixel-y-ness.  

```bash
def SRNetwork(input_data):
    input_data = tf.image.resize_images(input_data, [IMAGE_HEIGHT_GT, IMAGE_WIDTH_GT])
    hidden_layer1 = tf.layers.conv2d(input_data, filters=64, kernel_size=9, activation=tf.nn.relu, padding='SAME')
    hidden_layer2 = tf.layers.conv2d(hidden_layer1, filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME')
    hidden_layer2 = hidden_layer1 + hidden_layer2
    hidden_layer3 = tf.layers.conv2d(hidden_layer2, filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME')
    hidden_layer3 = hidden_layer2 + hidden_layer3
    hidden_layer4 = tf.layers.conv2d(hidden_layer3, filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME')
    hidden_layer4 = hidden_layer3 + hidden_layer4

    hidden_layer5 = tf.layers.conv2d(hidden_layer4, filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME')
    hidden_layer5 = hidden_layer4 + hidden_layer5
    hidden_layer6 = tf.layers.conv2d(hidden_layer5, filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME')
    hidden_layer6 = hidden_layer5 + hidden_layer6
    hidden_layer7 = tf.layers.conv2d(hidden_layer6, filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME')
    hidden_layer7 = hidden_layer1 + hidden_layer7

    output_layer = tf.layers.conv2d(hidden_layer7, filters=3, kernel_size=9, activation=tf.nn.sigmoid, padding='SAME')
    print(output_layer.shape)
    output_layer = output_layer + input_data
    
    return output_layer
```


# Basic Usage


train:
```bash
python train.py
```
run:
```bash
python test.py <path to image>
```

# Sample input (from running validation) 

Images at 64 × 45 pixels 

![alt text](https://github.com/sarahmfrost/artsandcrafts2/blob/master/autoencoder_images/1_input.jpg)
![alt text](https://github.com/sarahmfrost/artsandcrafts2/blob/master/autoencoder_images/2_input.jpg)
![alt text](https://github.com/sarahmfrost/artsandcrafts2/blob/master/autoencoder_images/3_input.jpg)
![alt text](https://github.com/sarahmfrost/artsandcrafts2/blob/master/autoencoder_images/4_input.jpg)


# Sample output 

Images at 128 × 90 pixels

![alt text](https://github.com/sarahmfrost/artsandcrafts2/blob/master/autoencoder_images/1_output.jpg)
![alt text](https://github.com/sarahmfrost/artsandcrafts2/blob/master/autoencoder_images/2_output.jpg)
![alt text](https://github.com/sarahmfrost/artsandcrafts2/blob/master/autoencoder_images/3_output.jpg)
![alt text](https://github.com/sarahmfrost/artsandcrafts2/blob/master/autoencoder_images/4_output.jpg)
