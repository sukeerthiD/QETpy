import pandas as pd
import numpy as np
import os

# plt.style.use("belle2")

# get input files

identifier = '1000'
folder = '/nfs/dust/belle2/user/khanmuni/wavedumpdata/'


def get_record_length(filename, pattern='RECORD_LENGTH'):
    with open(filename) as f:
        for line in f:
            # skip comment lines
            if not line.lstrip().startswith('#') and pattern in line:
                return int(line.split()[1]) #split and return second entry
    print('Could not find file {} or could not find pattern {} in file.'.format(filename, pattern))
    return -1


def read_waveforms(waveforms, cfg, n):
    
    # get record length from config file, return -1 if there was a problem
    recordlength = get_record_length(cfg)
    if recordlength < 0:
        return None

    # read the waveform files (one entry per line, this is not a csv file)
    arr = pd.read_csv(waveforms, header=None).to_numpy()
    nlines_in_arr = len(arr)
    
    # discard last event in case it was incomplete
    nevents = int(np.floor(nlines_in_arr/recordlength))
    arr = arr[:(nevents*recordlength),:]
    return arr.reshape(nevents, recordlength)


configfilename = os.path.join(folder, 'Config_' + identifier + '.txt')
filename_Tl = os.path.join(folder, 'wavedump_' + identifier + '.txt')
filename_pure = os.path.join(folder, 'wavedump1_' + identifier + '.txt')

sig_Tl = read_waveforms(waveforms = filename_Tl, cfg=configfilename, n=None)
print('Input shape CsI(Tl): ', sig_Tl.shape)

sig_pure = read_waveforms(waveforms = filename_pure, cfg=configfilename, n=None)
print('Input shape CsI(pure): ', sig_pure.shape)