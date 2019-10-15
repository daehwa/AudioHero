'''
Please run this code
in the same folder
with data to augment
'''

import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import glob

samplerate = 16000
subtype = 'PCM_16'
src_format = '.wav'
out_format = '_preprocessed.wav'


def get_files_in_folder(form):
    return glob.glob(form)

def check_already_exist(file,files):
    if file+out_format in files :
        return True
    return False

def load_wav(file,typeoffile):
    data = sf.SoundFile(file)
    print('-----'+typeoffile+' File-----')
    print('Sample rate: {}'.format(data.samplerate))
    print('Channels: {}'.format(data.channels))
    print('Subtype: {}'.format(data.subtype))
    print('---------------------')
    return data

def down_sample(input_wav, outputfile, origin_sr, resample_sr):
    y, sr = librosa.load(input_wav, sr=origin_sr)
    resample = librosa.resample(y, sr, resample_sr)
    return resample


def plot (filename):
    data, samplerate = sf.read(filename)
    times = np.arange(len(data))/float(samplerate)
    plt.figure(figsize=(30, 4))
    plt.plot(times, data[:,], color='b')
    plt.xlim(times[0], times[-1])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    rem_wav=filename[:-4]
    plt.savefig(rem_wav+'.png', dpi=100)

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def start_end(vals):
    result = []
    one_second_in_frame = second2frame(1)
    for v in vals:
        if v[len(v)-1] - v[0] > one_second_in_frame:
            result.append([v[0],v[len(v)-1]])
    return result

def second2frame(second):
    return samplerate*second

def cut_silence(data, index):
    if not index:
        return data
    index.append([data.shape[0]-1,data.shape[0]-1])
    data_no_slience = np.array([])
    start = 0
    for i in index:
        print(i)
        part = data[start:i[0]]
        data_no_slience = np.concatenate((data_no_slience, part), axis=None)
        start = i[1]
    return data_no_slience


def remove_silence(data):
    slience = np.argwhere(np.absolute(data[:,]) < 0.02).flatten()
    index = group_consecutives(slience)
    index = start_end(index)
    print("silent parts:",index)
    data_no_slience = cut_silence(data, index)
    return data_no_slience


def save_wav(outputfile,data):
    sf.write(outputfile, data, samplerate, subtype='PCM_16')



def main():
    files = get_files_in_folder('*000.wav')
    files_processed = get_files_in_folder('*'+out_format)
    for file in files:
        print('') # just for beauty
        file = file.replace(src_format,'')
        print("Processing with "+file)
        if check_already_exist(file,files_processed):
            print("already exist.")
            continue
        sourcefile = file+src_format
        outputfile = file+out_format
        data = load_wav(sourcefile,'Source')
        data = down_sample(sourcefile,outputfile,data.samplerate,samplerate)
        data = remove_silence(data)
        save_wav(outputfile,data)
        load_wav(outputfile,'Output')
        plot(sourcefile)
        plot(outputfile)

if __name__ == '__main__':
    main()