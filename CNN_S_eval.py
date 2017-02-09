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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import sys

import pdb

import CNN_S
from imagenet_data import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/cvpr-gb/hdd4TBmount/eval_dir/CNN_S',
							 """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/cvpr-gb/hdd4TBmount/train_dir/CNN_S',
							 """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
							"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
								"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
							 """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, k, summary_op):
	"""Run Eval once.

	Args:
		saver: Saver.
		summary_writer: Summary writer.
		top_k_op: Top K op.
		summary_op: Summary op.
	"""
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#	 /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return

		# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
																				 start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0	# Counts the number of correct predictions.
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				sys.stdout.write( "%d/%d\r"%(step,num_iter) )
				sys.stdout.flush()
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1

			# Compute precision @ 1.
			precision = true_count / total_sample_count
			print('%s: precision @ %d = %.3f' % (datetime.now(), k, precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ 1', simple_value=precision)
			summary_writer.add_summary(summary, global_step)
		except Exception as e:	# pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)


def evaluate( dataset ):
	"""Eval CIFAR-10 for a number of steps."""
	with tf.Graph().as_default() as g:
		# Get images and labels for CIFAR-10.
		images, labels = inputs( dataset )
		labels = labels-1

		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits, _ = CNN_S.inference(images, dataset.num_classes() )

		# Calculate predictions.
		top_1_op = tf.nn.in_top_k(logits, labels, 1)
		top_5_op = tf.nn.in_top_k(logits, labels, 5)

		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(
				CNN_S.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.summary.merge_all()

		summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

		while True:
			print( "evaluating..." )
			eval_once(saver, summary_writer, top_1_op, 1, summary_op)
			eval_once(saver, summary_writer, top_5_op, 5, summary_op)
			if FLAGS.run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):	# pylint: disable=unused-argument
	dataset = ImageNetData( subset="validation" )
	tf.gfile.MakeDirs(FLAGS.eval_dir)
	evaluate( dataset )


if __name__ == '__main__':
	tf.app.run()
