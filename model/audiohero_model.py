from __future__ import print_function
import os
from random import shuffle
import numpy as np
import tensorflow as tf
import time
import pandas as pd

import vggish_input
import vggish_params
import vggish_slim

flags = tf.app.flags
slim = tf.contrib.slim

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
    'checkpoint', 'audiohero_ckpt/audiohero.ckpt-0',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS


###################
##[AI501 Project]##
##  "AudioHero"  ##
## Detect Danger ##
##  From  Audio  ##
###################

class AudioHero():
  def __init__(self, isTrain=False):
    self._NUM_CLASSES = 8
    self.isTrain = isTrain
    self.best_loss = 99999
    self.labeled_data = pd.read_csv('evaluation_label_table.csv')
    self.onehot_label = np.eye(self._NUM_CLASSES)
    self.label_num_and_name = [
                                "1. Construction",
                                "2. Siren",
                                "3. Vehicle",
                                "4. Crying",
                                "5. Scream",
                                "6. Explosion",
                                "7. Animal Threat",
                                "8. Vehicle Horn"
                              ]

  def load_train_dataset(self,path):
    train_metadata_name = path+'train_label_table.csv'
    labeled_data = pd.read_csv(train_metadata_name)
    idx = self.load_shuffled_label(labeled_data)

    num_of_data = labeled_data.shape[0]
    batch_size = int(num_of_data/FLAGS.num_batches)

  def load_shuffled_label(data):
    idx = np.arange(num_of_data)
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    return idx

  def get_examples_(self, i, audio_name):
    # audio_name = self.labeled_data.iloc[i,0]
    context_num = self.labeled_data.iloc[i,1]
    all_examples = vggish_input.wavfile_to_examples(audio_name)
    all_labels =  np.array([self.onehot_label[context_num-1]] * all_examples.shape[0])
    labeled_examples = list(zip(all_examples, all_labels))
    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    if features == None:
      print("ERROR: None features")
      exit()
    return (features, labels, context_num)

  def load_shuffled_label(data):
    idx = np.arange(num_of_data)
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    return idx

  def get_examples(self, i, audio_name, shuffle=True):
    all_examples = vggish_input.wavfile_to_examples(audio_name)
    return (all_examples, None, None)

  def load_audiohero_checkpoint(self, sess):
    """Loads a pre-trained VGGish-compatible checkpoint.

    This function can be used as an initialization function (referred to as
    init_fn in TensorFlow documentation) which is called in a Session after
    initializating all variables. When used as an init_fn, this will load
    a pre-trained checkpoint that is compatible with the VGGish model
    definition. Only variables defined by VGGish will be loaded.

    Args:
      sess: an active TensorFlow session.
      checkpoint_path: path to a file containing a checkpoint that is
        compatible with the VGGish model definition.
    """
    # Get the list of names of all VGGish variables that exist in
    # the checkpoint (i.e., all inference-mode VGGish variables).
    checkpoint_path = FLAGS.checkpoint
    variable_list = tf.all_variables()

    with tf.Graph().as_default():
      audiohero_var_names = [v.name for v in variable_list]

    # Get the list of all currently existing variables that match
    # the list of variable names we just computed.
    audiohero_vars = [v for v in tf.all_variables() if v.name in audiohero_var_names]

    # Use a Saver to restore just the variables selected above.
    saver = tf.train.Saver(audiohero_vars, name='audiohero_load_pretrained',
                           write_version=1)
    saver.restore(sess, checkpoint_path)

    # Locate all the tensors and ops.
    self.features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)

  def define_model(self,sess):
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
      self.logits = slim.fully_connected(
          fc, self._NUM_CLASSES, activation_fn=None, scope='logits')
      self.prediction = tf.sigmoid(self.logits, name='prediction')

      if (self.isTrain):
        self.add_training_op(sess)

  def init_variables(self,sess):
    sess.run(tf.global_variables_initializer())

  def add_training_op(self,sess):
    # Add training ops.
    with tf.variable_scope('train'):
      global_step = tf.Variable(
          0, name='global_step', trainable=False,
          collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                       tf.GraphKeys.GLOBAL_STEP])

      # Labels are assumed to be fed as a batch multi-hot vectors, with
      # a 1 in the position of each positive class label, and 0 elsewhere.
      labels = tf.placeholder(
          tf.float32, shape=(None, self._NUM_CLASSES), name='labels')

      # Cross-entropy label loss.
      xent = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self.logits, labels=labels, name='xent')
      loss = tf.reduce_mean(xent, name='loss_op')
      tf.summary.scalar('loss', loss)

      # We use the same optimizer and hyperparameters as used to train VGGish.
      optimizer = tf.train.AdamOptimizer(
          learning_rate=vggish_params.LEARNING_RATE,
          epsilon=vggish_params.ADAM_EPSILON)
      optimizer.minimize(loss, global_step=global_step, name='train_op')

      # Locate all the tensors and ops we need for the training loop.
      self.labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
      self.global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
      self.loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
      self.train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')


  def predict(self, sess, features, labels=None):
    if self.isTrain:
      if labels is None:
        print("Please set \'labels\' in predict function.")
        exit()
      [num_steps, loss, _, p] = sess.run(
          [self.global_step_tensor, self.loss_tensor, self.train_op, self.prediction],
          feed_dict={self.features_tensor: features, self.labels_tensor: labels})
      return (num_steps, loss, _, p)
    else:
      p = sess.run(
          self.prediction,
          feed_dict={self.features_tensor: features})

      prob = np.sum(p,axis=0)
      pred_label = np.argmax(prob)+1

      return prob, pred_label

  def get_label_name(self):
    return self.label_num_and_name

  def save_best_ckpt(self, sess, loss):
    if loss < self.best_loss:
      self.best_loss = loss
      checkpoint_prefix = "./trained_model/audiohero.ckpt"
      checkpoint_state_name = "checkpoint"
      input_graph_name = "input_graph.pb"
      checkpoint_path = tf.train.Saver().save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
      tf.train.write_graph(sess.graph_def, './graph/temp', input_graph_name, as_text=False)