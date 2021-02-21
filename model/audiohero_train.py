from __future__ import print_function

from random import shuffle

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

import vggish_input

from audiohero_model import AudioHero

# Parameter Parser
parser = argparse.ArgumentParser()
parser.add_argument('--train_folder',required=True,help="Please put audio file name")
args = parser.parse_args()

training_data_path = args.train_folder # '../../AI4ML/dataset/train/'
train_metadata_name = '../dataset/train_label_table.csv'

_NUM_CLASSES = 8
onehot_label = np.eye(_NUM_CLASSES)


labeled_data = pd.read_csv(train_metadata_name)
num_of_data = labeled_data.shape[0]
batch_size = int(num_of_data/7000)

def load_shuffled_label(data):
  idx = np.arange(num_of_data)
  np.random.shuffle(idx)
  np.random.shuffle(idx)
  return idx


def _get_examples_batch(loop_i, idx):
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
    # print("loading audio file...      ",_ - start_idx + 1,"/",batch_size)
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
    ah = AudioHero(isTrain=True)

    # Load Model. VGGish + additional layers
    ah.define_model(sess)

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    ah.init_variables(sess)
    ah.load_audiohero_checkpoint(sess)

  # The training loop.
  ah.load_train_dataset()
  # idx = load_shuffled_label(labeled_data)

  for ep in range(100):
    for _ in range(int(7000*2/5)):
      # Generate input example from audio data
      (features, labels) = _get_examples_batch(_, idx)

      # Run (Prediction and Feed)
      num_steps, loss, _, p = ah.predict(sess, features, labels)

      # Print loss
      print('Step %d: loss %g' % (num_steps, loss))

      # Save Best ckpt
      ah.save_best_ckpt(sess, loss)

if __name__ == '__main__':
  tf.app.run()