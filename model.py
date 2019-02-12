import tensorflow as tf
import numpy as np

from utils import *


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


