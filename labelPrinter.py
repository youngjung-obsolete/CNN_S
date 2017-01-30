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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System				| Step Time (sec/batch)	|		 Accuracy
------------------------------------------------------------------
1 Tesla K20m	| 0.35-0.60							| ~86% at 60K steps	(5 hours)
1 Tesla K40m	| 0.25-0.35							| ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import pdb

import CNN_S
from imagenet_data import *

FLAGS = tf.app.flags.FLAGS
import flags

def printLabels( dataset ):
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()

		# Get images and labels for CIFAR-10.
		images, labels = distorted_inputs( dataset )
		out_labels = tf.identity( labels )

	return images, labels

def main(argv=None):	# pylint: disable=unused-argument
	dataset= ImageNetData( subset=FLAGS.subset )
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	images, labels = printLabels( dataset )

	x = tf.placeholder( tf.float32 )
	y = tf.placeholder( tf.int32 )
	sess = tf.Session()
	sess.run( labels ) 


if __name__ == '__main__':
	tf.app.run()
