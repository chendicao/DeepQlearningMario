import tensorflow as tf
import numpy as np
from collections import deque

ACTIONS = 3
GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 20000
FINAL_EPSILON = 0.001
INITIAL_EPSILON = 0.99
REPLAY_MEMORY_SIZE = 50000
BATCH = 32
FRAME_PER_ACTION = 1

class Model:
	def __init__(self):
		def conv_layer(x, conv, stride = 1):
			return tf.nn.conv2d(x, conv, [1, stride, stride, 1], padding = 'SAME')
		
		def pooling(x, k = 2, stride = 2):
			return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding = 'SAME')

		self.middle_game = False
		self.memory = deque()
		self.initial_stack_images = np.zeros((80, 80, 4))
		self.X = tf.placeholder("float", [None, 80, 80, 4])
		self.action_space = tf.placeholder("float", [None, 2])
		self.action_left = tf.placeholder("float", [None, 2])
		self.action_right = tf.placeholder("float", [None, 2])
		self.Y = tf.placeholder("float", [None])

		w_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.1))
		b_conv1 = tf.Variable(tf.truncated_normal([32], stddev = 0.01))
		conv1 = tf.nn.relu(conv_layer(self.X, w_conv1, stride = 4) + b_conv1)
		pooling1 = pooling(conv1)

		w_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.1))
		b_conv2 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
		conv2 = tf.nn.relu(conv_layer(pooling1, w_conv2, stride = 2) + b_conv2)

		w_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.1))
		b_conv3 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
		conv3 = tf.nn.relu(conv_layer(conv2, w_conv3) + b_conv3)

		conv3 = tf.reshape(conv3, [-1, 1600])
		w_fc1 = tf.Variable(tf.truncated_normal([1600, 512], stddev = 0.1))
		b_fc1 = tf.Variable(tf.truncated_normal([512], stddev = 0.01))
		fc_512 = tf.nn.relu(tf.matmul(conv3, w_fc1) + b_fc1)

		w_space = tf.Variable(tf.truncated_normal([512, 2], stddev = 0.1))
		b_space = tf.Variable(tf.truncated_normal([2], stddev = 0.01))
		self.logits_space = tf.matmul(fc_512, w_space) + b_space
		
		w_left = tf.Variable(tf.truncated_normal([512, 2], stddev = 0.1))
		b_left = tf.Variable(tf.truncated_normal([2], stddev = 0.01))
		self.logits_left = tf.matmul(fc_512, w_left) + b_left
		
		w_right = tf.Variable(tf.truncated_normal([512, 2], stddev = 0.1))
		b_right = tf.Variable(tf.truncated_normal([2], stddev = 0.01))
		self.logits_right = tf.matmul(fc_512, w_right) + b_right

		readout_space = tf.reduce_sum(tf.multiply(self.logits_space, self.action_space), reduction_indices = 1)
		readout_left = tf.reduce_sum(tf.multiply(self.logits_left, self.action_left), reduction_indices = 1)
		readout_right = tf.reduce_sum(tf.multiply(self.logits_right, self.action_right), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.Y - readout_space)) + tf.reduce_mean(tf.square(self.Y - readout_left)) + tf.reduce_mean(tf.square(self.Y - readout_right))
		self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)








