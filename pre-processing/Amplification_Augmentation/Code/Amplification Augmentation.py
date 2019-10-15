#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf


# In[42]:


file_name='--cB2ZVjpnA_30.000.wav'


# In[43]:


def quieter(file_name):
    samplerate, data = sio.wavfile.read(file_name)
    times = np.arange(len(data))/float(samplerate)
    data=(data*0.25).round(0).astype(int)
    arr=np.asarray(data, dtype=np.int16)
    rem_wav=file_name[:-4]
    write(rem_wav+'_quiter.wav',samplerate, arr)


# In[44]:


quieter(file_name)


# In[49]:


### Modified Part ###
def db_to_ratio(db):
    return 10**(db/20.0)


def louder(file_name):
    data, samplerate = sf.read(file_name)
    times = np.arange(len(data))/float(samplerate)
    target_peak_value = db_to_ratio(-0.1)
    #print(target_peak_value)
    max_val = np.max(np.abs(data))
    #print(max_val)
    
    to_target=max(max_val,target_peak_value)
    
    ratio = to_target/max_val
    data= data*ratio
    rem_wav=file_name[:-4]
    sf.write(rem_wav+'_louder.wav', data, samplerate)
### Modified Part ###


# In[50]:


louder(file_name)


# In[47]:


def plot (file_name):
    samplerate, data = sio.wavfile.read(file_name)
    times = np.arange(len(data))/float(samplerate)
    plt.figure(figsize=(30, 4))
    plt.fill_between(times, data[:,0], data[:,1], color='k') 
    plt.xlim(times[0], times[-1])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    rem_wav=file_name[:-4]
    plt.savefig(rem_wav+'.png', dpi=100)


# In[48]:


plot(file_name)


# In[ ]:


plot('--aaILOrkII_200.000_quiter.wav')


# In[38]:


plot('--aaILOrkII_200.000_louder.wav')


# In[ ]:





# In[ ]:




