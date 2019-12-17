import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(this_dir)
sys.path.append(root_dir)

import vggish_input
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def visualize_mel_log(filename,pred_label):
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    data, samplerate = sf.read(filename)
    example = vggish_input.waveform_to_examples(data,samplerate)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(example[0], y_axis='mel', sr=samplerate, hop_length=hop_length, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    title_size = 50
    if(pred_label == 1): 
        plt.title('Drilling', fontsize=title_size)
    elif(pred_label == 2): 
        plt.title('Siren', fontsize=title_size)
    elif(pred_label == 3): 
        plt.title('Vehicle Approaching', fontsize=title_size)
    elif(pred_label == 8): 
        plt.title('Vehicle Horn', fontsize=title_size)
    elif(pred_label == 4): 
        plt.title('Human Sign (Crying)', fontsize=title_size)
    elif(pred_label == 5): 
        plt.title('Human Sign (Screaming)', fontsize=title_size)
    elif(pred_label == 6): 
        plt.title('Explosion', fontsize=title_size)
    elif(pred_label == 7): 
        plt.title('Animal Threat', fontsize=title_size)
    plt.tight_layout()
    # plt.savefig('Mel-Spectrogram example.png')
    plt.show()