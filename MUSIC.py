import numpy as np
import scipy.io as sio
import time
import torch
import math
import matplotlib.pyplot as plt
import pyroomacoustics as pra



class MUSIC:

    def __init__(self, coordinates, parameters):

        self.sample_rate = parameters['sample_rate']
        self.frame_length = parameters['frame_length']
        self.num_classes = parameters['num_classes']
        self.base_dir = parameters['base_dir']
        self.use_informed = parameters['use_informed']
        self.percentile = parameters['mask_percentile']
        self.num_channels = len(coordinates)

        self.speed_of_sound = 344

        theta = np.divide(range(self.num_classes), self.num_classes) * 2 * math.pi

        self.music = pra.doa.music.MUSIC(L=np.transpose(coordinates), fs=self.sample_rate, nfft=self.frame_length,
                                         c=self.speed_of_sound, num_src=1, mode='far', azimuth=theta)

    def forward(self, data):
        if self.use_informed:
            return self.calculate_informed(signal=data)
        else:
            return self.calculate_uninformed(signal=data)

    def calculate_uninformed(self, signal):

        spec_frames = np.expand_dims(a=np.fft.rfft(signal[:self.num_channels, :], n=self.frame_length, axis=-1), axis=-1)
        self.music.locate_sources(spec_frames)
        return self.music.grid.values

    def calculate_informed(self, signal):

        spec_frames = np.expand_dims(a=np.fft.rfft(signal, n=self.frame_length, axis=-1), axis=-1)
        magnitudes = np.abs(spec_frames[self.num_channels, :, 0])

        threshold = np.quantile(magnitudes, self.percentile/100)
        mask = np.zeros(shape=(int(self.frame_length/2+1),))
        for freq in range(int(self.frame_length / 2 + 1)):
            if magnitudes[freq] >= threshold:
                mask[freq] = 1

        for channel in range(self.num_channels):
            spec_frames[channel, :, 0] *= np.transpose(mask)

        self.music.locate_sources(spec_frames[:self.num_channels, :, :])
        return self.music.grid.values

    def eval(self):
        pass