import tensorflow as tf

# General parameters.
tf.app.flags.DEFINE_string('train_dir', '/home/cvpr-gb/hdd4TBmount/train_dir/CNN_S',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
								"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/cvpr-gb/hdd4TBmount/DataSet/ImageNet',
								"""Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
								"""Train the model using fp16.""")

# Dataset parameters
tf.app.flags.DEFINE_string('subset', 'train',
								"""Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_integer('image_size', 224,
								"""Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
								"""Number of preprocessing threads per tower. """
								"""Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
								"""Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
								"""Size of the queue of preprocessed images. """
								"""Default is ideal but try smaller values, e.g. """
								"""4, 2 or 1, if host memory is constrained. See """
								"""comments in code for more details.""")


