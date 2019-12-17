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
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(this_dir)
sys.path.append(root_dir)

from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

import vggish_input
import vggish_params
import vggish_slim

from visualize import visualize_mel_log

flags = tf.app.flags
slim = tf.contrib.slim

training_name = 'temp'
net_dir = './graph/'+training_name
ckpt_dir = './audihero_ckpt/'

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




###################
##[AI501 Project]##
##  "AudioHero"  ##
## Detect Danger ##
##  From  Audio  ##
###################



_NUM_CLASSES = 8
audio_name = 'human_crying.wav'










onehot_label = np.eye(_NUM_CLASSES)
eval_data_path = '../sample_audio'
filename = 'evaluation_label_table.csv'
labeled_data = pd.read_csv(filename)
num_of_data = labeled_data.shape[0]

def _get_examples(i):
  # audio_name = labeled_data.iloc[i,0]
  context_num = labeled_data.iloc[i,1]
  try:
    all_examples = vggish_input.wavfile_to_examples(os.path.join(eval_data_path,audio_name))
  except:
    return None, None, None
  all_labels =  np.array([onehot_label[context_num-1]] * all_examples.shape[0])
  labeled_examples = list(zip(all_examples, all_labels))
  # Separate and return the features and labels.
  features = [example for (example, _) in labeled_examples]
  labels = [label for (_, label) in labeled_examples]
  return (features, labels, context_num)

def load_audiohero_checkpoint(session, checkpoint_path, variable_list):
  """Loads a pre-trained VGGish-compatible checkpoint.

  This function can be used as an initialization function (referred to as
  init_fn in TensorFlow documentation) which is called in a Session after
  initializating all variables. When used as an init_fn, this will load
  a pre-trained checkpoint that is compatible with the VGGish model
  definition. Only variables defined by VGGish will be loaded.

  Args:
    session: an active TensorFlow session.
    checkpoint_path: path to a file containing a checkpoint that is
      compatible with the VGGish model definition.
  """
  # Get the list of names of all VGGish variables that exist in
  # the checkpoint (i.e., all inference-mode VGGish variables).
  with tf.Graph().as_default():
    audiohero_var_names = [v.name for v in variable_list]

  # Get the list of all currently existing variables that match
  # the list of variable names we just computed.
  audiohero_vars = [v for v in tf.all_variables() if v.name in audiohero_var_names]

  # Use a Saver to restore just the variables selected above.
  saver = tf.train.Saver(audiohero_vars, name='audiohero_load_pretrained',
                         write_version=1)
  saver.restore(session, checkpoint_path)

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

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    load_audiohero_checkpoint(sess, FLAGS.checkpoint,tf.all_variables())

    # Locate all the tensors and ops we need for the training loop.
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)

    # f=open('result.csv','ab')
    all_count = 0
    true_count = 0
    acc = [0]
    for i in range(1):
      print("Data num ",i,"/",num_of_data)
      (_features, _labels, true_label) = _get_examples(i)
      if _features == None:
        continue
      p = sess.run(
          prediction,
          feed_dict={features_tensor: _features})
      print(p*100)
      print('--------------------------')
      # freq_in_one_slot = np.argmax(p,axis=1)+1
      # print("Predict label:" , freq_in_one_slot)
      # counts = np.bincount(freq_in_one_slot)
      # pred_label = np.argmax(counts)
      prob = np.sum(p,axis=0)
      pred_label = np.argmax(prob)+1
      all_count = all_count + 1
      if(pred_label == true_label):
        true_count = true_count + 1
      # print("Predict: ",pred_label)
      # print("Groundtruth: ",true_label)
      # acc[0] = true_count/all_count*100
      # print("Acc: ", acc[0])
      # np.savetxt(f, [[pred_label,true_label]], delimiter=',')   # X is an array
      visualize_mel_log(os.path.join(eval_data_path,audio_name),pred_label)
    # np.savetxt('accuracy.txt', acc)
    # f.close()

    # print('Step %d: loss %g' % (num_steps, loss))

if __name__ == '__main__':
  tf.app.run()