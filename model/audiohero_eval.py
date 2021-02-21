from audiohero_model import AudioHero
import tensorflow as tf
import vggish_slim
import vggish_params
import pandas as pd
import numpy as np
import os
import argparse

from visualize import visualize_mel_log

# Parameter Parser
parser = argparse.ArgumentParser()
parser.add_argument('--filename',required=True,help="Please put audio file name")
args = parser.parse_args()
audio_name = args.filename

def main(_):
  with tf.Graph().as_default(), tf.Session() as sess:
    # Init AudioHero
    ah = AudioHero()

    # Load Model. VGGish + additional layers
    ah.define_model(sess)

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    ah.init_variables(sess)
    ah.load_audiohero_checkpoint(sess)

    # Generate input example from audio data
    (features, _, _) = ah.get_examples(0,audio_name)

    # Evaluate the input example.
    prob, pred_label = ah.predict(sess,features)
    label_name = ah.get_label_name()

    print('--------------------------')
    prob_with_name = pd.Series(prob, index=label_name)

    print("probability:\n{}\npredicted label:{}".format(prob_with_name,
                                        label_name[pred_label-1]))

    # Visualize input and prediction result
    visualize_mel_log(audio_name,pred_label-1)


if __name__ == '__main__':
  tf.app.run()