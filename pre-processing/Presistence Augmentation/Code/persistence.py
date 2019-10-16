import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
from pysndfx import AudioEffectsChain

# Find options for each room
# Recital Hall

fx = (
    AudioEffectsChain()
    .reverb(reverberance=45,
    		hf_damping=10,
    		room_scale=28,
    		stereo_depth=28,
    		pre_delay=6,
    		wet_gain=2,
    		wet_only=False)
)

'''

# Small Bathroom Preset

fx = (
    AudioEffectsChain()
    .reverb(reverberance=32,
    		hf_damping=30,
    		room_scale=3,
    		stereo_depth=3,
    		pre_delay=0,
    		wet_gain=2,
    		wet_only=False)
)

# Bed Room
fx = (
    AudioEffectsChain()
    .reverb(reverberance=22,
    		hf_damping=20,
    		room_scale=6,
    		stereo_depth=6,
    		pre_delay=0,
    		wet_gain=2,
    		wet_only=False)
)
'''
#


inputfile_names = glob.glob('*000.wav')

for inputfile_name in inputfile_names :
	outputfile_name = inputfile_name.replace('.wav','') + '_processed.wav'

	fx(inputfile_name, outputfile_name)
