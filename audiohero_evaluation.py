from __future__ import print_function
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import argparse
import model.vggish_input as vggish_input

###################
##[AI501 Project]##
##  "AudioHero"  ##
## Detect Danger ##
##  From  Audio  ##
###################

_NUM_CLASSES = 8
filename = 'human_crying.wav'

def generate_examples(filename):
  foldername = "sample_audio"
  path = os.path.join(foldername,filename)
  examples = vggish_input.wavfile_to_examples(path)
  labels = np.array([onehot_label[context_num-1]] * all_examples.shape[0])

  label
  return all_examples

def main():
  examples = generate_examples(filename)


if __name__ == "__main__":
  main()