#!/usr/bin/env python3

import librosa
from IPython.display import Audio
from scipy.io import wavfile
import noisereduce as nr
import numpy as np
import os

path = "./datasets"
# https://github.com/timsainb/noisereduce
# load data
# rate, data = wavfile.read(path + "/test/441.wav")
# # perform noise reduction
# reduced_noise = nr.reduce_noise(y=data, sr=rate, n_fft=2**7)
# wavfile.write(path + "/test_reduced/441.wav", rate, reduced_noise)
# y, sr = librosa.load(path + "/test_reduced/441.wav")
# mfccs = librosa.feature.mfcc(y=y, sr=sr)
# feacture_vector = np.array(mfccs.mean(axis=1))
# print("Feacture Vector of the 441 sound with noise reduced :", feacture_vector)


def reduce_noise(path, data_type="/train"):
    directory_1 = os.listdir(path + data_type)
    # print(directory_1)
    for f in directory_1:
        rate, data = wavfile.read(path + data_type + "/" + f)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, n_fft=2**7)
        wavfile.write(path + data_type + "_reduced/" + f, rate, reduced_noise)


# reduce_noise(path, "/train")
reduce_noise(path, "/test")
