# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in training mode.
This is intended as a toy example that demonstrates how to use the VGGish model
definition within a larger model that adds more layers on top, and then train
the larger model. If you let VGGish train as well, then this allows you to
fine-tune the VGGish model parameters for your application. If you don't let
VGGish train, then you use VGGish as a feature extractor for the layers above
it.
For this toy task, we are training a classifier to distinguish between three
classes: sine waves, constant signals, and white noise. We generate synthetic
waveforms from each of these classes, convert into shuffled batches of log mel
spectrogram examples with associated labels, and feed the batches into a model
that includes VGGish at the bottom and a couple of additional layers on top. We
also plumb in labels that are associated with the examples, which feed a label
loss used for training.
Usage:
  # Run training for 100 steps using a model checkpoint in the default
  # location (vggish_model.ckpt in the current directory). Allow VGGish
  # to get fine-tuned.
  $ python vggish_train_demo.py --num_batches 100
  # Same as before but run for fewer steps and don't change VGGish parameters
  # and use a checkpoint in a different location
  $ python vggish_train_demo.py --num_batches 50 \
                                --train_vggish=False \
                                --checkpoint /path/to/model/checkpoint
"""

from __future__ import print_function

from random import shuffle

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

import vggish_input
import vggish_params
import vggish_slim

flags = tf.app.flags
slim = tf.contrib.slim

training_name = 'temp'
net_dir = './graph/'+training_name

flags.DEFINE_integer(
    'num_batches', 7000,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'audiohero.ckpt-0',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

_NUM_CLASSES = 8
onehot_label = np.eye(_NUM_CLASSES)

s = (os.path.dirname(__file__)+",path,to,the,training,folder").split(",")
training_data_path = os.path.join(*s)
filename = 'train_label_table.csv'
labeled_data = pd.read_csv(filename)
num_of_data = labeled_data.shape[0]
batch_size = int(num_of_data/FLAGS.num_batches)

def load_shuffled_label(data):
  idx = np.arange(num_of_data)
  np.random.shuffle(idx)
  np.random.shuffle(idx)
  return idx


def _get_examples_batch(loop_i, idx):
  print('current loop:', loop_i)
  """Returns a shuffled batch of examples of all audio classes.
  Note that this is just a toy function because this is a simple demo intended
  to illustrate how the training code might work.
  Returns:
    a tuple (features, labels) where features is a NumPy array of shape
    [batch_size, num_frames, num_bands] where the batch_size is variable and
    each row is a log mel spectrogram patch of shape [num_frames, num_bands]
    suitable for feeding VGGish, while labels is a NumPy array of shape
    [batch_size, num_classes] where each row is a multi-hot label vector that
    provides the labels for corresponding rows in features.
  """
  # Make a waveform for each class.

  start_idx = batch_size*loop_i
  end_idx = batch_size*(loop_i+1)
  if(end_idx > num_of_data):
    end_idx = num_of_data

  all_examples = np.array([]).reshape(0, 96, 64)
  all_labels = np.array([]).reshape(0, 8)
  for _ in range(start_idx, end_idx):
    print("loading audio file...      ",_ - start_idx + 1,"/",batch_size)
    i = idx[_]
    audio_name = labeled_data.iloc[i,0]
    context_num = labeled_data.iloc[i,1]
    try:
      audio_example = vggish_input.wavfile_to_examples(training_data_path+'/'+audio_name)
    except:
      # print("no file name of",audio_name)
      continue
    label = np.array([onehot_label[context_num-1]] * audio_example.shape[0])
    all_examples = np.concatenate((all_examples, audio_example))
    all_labels = np.concatenate((all_labels, label))
  labeled_examples = list(zip(all_examples, all_labels))
  shuffle(labeled_examples)

  # Separate and return the features and labels.
  features = [example for (example, _) in labeled_examples]
  labels = [label for (_, label) in labeled_examples]
  return (features, labels)


def main(_):
  with tf.Graph().as_default(), tf.Session() as sess:
    # Define VGGish.
    embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with tf.variable_scope('mymodel'):
      # Add a fully connected layer with 100 units.
      num_units = 100
      fc = slim.fully_connected(embeddings, num_units)

      # Add a classifier layer at the end, consisting of parallel logistic
      # classifiers, one per class. This allows for multi-class tasks.
      logits = slim.fully_connected(
          fc, _NUM_CLASSES, activation_fn=None, scope='logits')
      prediction = tf.sigmoid(logits, name='prediction')

      # Add training ops.
      with tf.variable_scope('train'):
        global_step = tf.Variable(
            0, name='global_step', trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.GLOBAL_STEP])

        # Labels are assumed to be fed as a batch multi-hot vectors, with
        # a 1 in the position of each positive class label, and 0 elsewhere.
        labels = tf.placeholder(
            tf.float32, shape=(None, _NUM_CLASSES), name='labels')

        # Cross-entropy label loss.
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xent')
        loss = tf.reduce_mean(xent, name='loss_op')
        tf.summary.scalar('loss', loss)

        # We use the same optimizer and hyperparameters as used to train VGGish.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=vggish_params.LEARNING_RATE,
            epsilon=vggish_params.ADAM_EPSILON)
        optimizer.minimize(loss, global_step=global_step, name='train_op')

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

    # Locate all the tensors and ops we need for the training loop.
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
    global_step_tensor = sess.graph.get_tensor_by_name(
        'mymodel/train/global_step:0')
    loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
    train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

    # The training loop.
    idx = load_shuffled_label(labeled_data)
    startTime = time.time()
    for _ in range(int(FLAGS.num_batches*2/5)):
      print(tf.trainable_variables())
      startTime_in = time.time()
      (features, labels) = _get_examples_batch(_, idx)
      [num_steps, loss, _, p] = sess.run(
          [global_step_tensor, loss_tensor, train_op, prediction],
          feed_dict={features_tensor: features, labels_tensor: labels})
      endTime_in = time.time()
      timePassed_in = endTime_in - startTime_in
      print('Inner loop time: ',timePassed_in)
      print('Step %d: loss %g' % (num_steps, loss))
      freq_in_one_slot = np.argmax(p,axis=1)+1
      print('prediction:', freq_in_one_slot)
      if(num_steps % 100 == 0):
        num_checkpoint = int(num_steps/100)+1
        checkpoint_prefix = "./checkpoint_folder/"+training_name+"/audiohero_{}.ckpt".format(num_checkpoint)
        checkpoint_state_name = "checkpoint_{}".format(num_checkpoint)
        input_graph_name = "input_graph_{}.pb".format(num_checkpoint)
        checkpoint_path = tf.train.Saver().save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
        tf.train.write_graph(sess.graph_def, net_dir, input_graph_name, as_text=False)
    checkpoint_prefix = "./checkpoint_folder/"+training_name+"/audiohero_final.ckpt"
    checkpoint_state_name = "checkpoint_final"
    input_graph_name = "input_graph_final.pb"
    checkpoint_path = tf.train.Saver().save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
    tf.train.write_graph(sess.graph_def, net_dir, input_graph_name, as_text=False)

    endTime = time.time()
    timePassed = endTime - startTime
    print('Total time: ',timePassed)

if __name__ == '__main__':
  tf.app.run()