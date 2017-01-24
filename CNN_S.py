# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
slim = tf.contrib.slim
import pdb

FLAGS = tf.app.flags.FLAGS
import flags

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999		 # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0			# Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1	# Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1			 # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	Args:
		x: Tensor
	Returns:
		nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scalar_summary(tensor_name + '/sparsity',
																			 tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable

	Returns:
		Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
				decay is not added for this Variable.

	Returns:
		Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
			name,
			shape,
			tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def inputs(eval_data):
	"""Construct input for CIFAR evaluation using the Reader ops.

	Args:
		eval_data: bool, indicating if one should use the train or eval data set.

	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.

	Raises:
		ValueError: If no data_dir
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
	images, labels = cifar10_input.inputs(eval_data=eval_data,
																				data_dir=data_dir,
																				batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels


def inference( images, num_classes ):
	"""Build the CNN_S model.

	Args:
		images: Images returned from distorted_inputs() or inputs().

	Returns:
		Logits.
	"""
	end_points = {}
	with tf.variable_scope('CNN_S') as sc:
		end_points['conv1'] = slim.conv2d( images, 96, [7, 7], stride=2, padding='VALID', scope='conv1')
		end_points['lrn'] = tf.nn.local_response_normalization( end_points['conv1'] )
		end_points['pool1'] = slim.max_pool2d(end_points['lrn'], [3, 3], stride=3, scope='pool1')
		end_points['conv2'] = slim.conv2d( end_points['pool1'], 256, [5, 5], stride=1, padding='SAME', scope='conv2')
		end_points['pool2'] = slim.max_pool2d(end_points['conv2'], [2, 2], stride=2, scope='pool2')
		end_points['conv3'] = slim.conv2d( end_points['pool2'], 512, [5, 5], stride=1, padding='SAME', scope='conv3')
		end_points['conv4'] = slim.conv2d( end_points['conv3'], 512, [5, 5], stride=1, padding='SAME', scope='conv4')
		end_points['conv5'] = slim.conv2d( end_points['conv4'], 512, [5, 5], stride=1, padding='SAME', scope='conv5')
		end_points['pool5'] = slim.max_pool2d(end_points['conv5'], [3, 3], stride=3, scope='pool5')
		# Use conv2d instead of fully_connected layers.
		end_points['fc6'] = slim.conv2d(end_points['pool5'], 4096, [6, 6], padding='VALID', scope='fc6')
		end_points['dropout6'] = slim.dropout(end_points['fc6'], scope='dropout6')
		end_points['fc7'] = slim.conv2d(end_points['dropout6'], 4096, [1, 1], scope='fc7' )
		end_points['dropout7'] = slim.dropout(end_points['fc7'], scope='dropout7')
		end_points['fc8'] = slim.conv2d(end_points['fc7'], num_classes, [1, 1], scope='fc8' )
		net = end_points['fc8']
		net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
	return net, end_points

def loss(logits, labels, batch_size=None ):
	"""Add L2Loss to all the trainable variables.

	Add summary for "Loss" and "Loss/avg".
	Args:
		logits: Logits from inference().
		labels: Labels from distorted_inputs or inputs(). 1-D tensor
						of shape [batch_size]

	Returns:
		Loss tensor of type float.
	"""
	if not batch_size:
		batch_size = FLAGS.batch_size

	# Reshape the labels into a dense Tensor of
	# shape [FLAGS.batch_size, num_classes].
	sparse_labels = tf.reshape(labels, [batch_size, 1])
#	indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
#	concated = tf.concat(1, [indices, sparse_labels])
#	num_classes = logits[0].get_shape()[-1].value
#	dense_labels = tf.sparse_to_dense(concated,[batch_size, num_classes],1.0, 0.0)

	# Cross entropy loss for the main softmax prediction.
	slim.losses.sparse_softmax_cross_entropy(logits[0],
									sparse_labels,
									weight=1.0)

def _add_loss_summaries(total_loss):
	"""Add summaries for losses in CIFAR-10 model.

	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.

	Args:
		total_loss: Total loss from loss().
	Returns:
		loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.scalar_summary(l.op.name + ' (raw)', l)
		tf.scalar_summary(l.op.name, loss_averages.average(l))

	return loss_averages_op


