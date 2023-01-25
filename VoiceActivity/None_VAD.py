import numpy as np

class None_VAD:

    def __init__(self, sample_rate, framesize):

        self.sample_rate = sample_rate
        self.frame_length_s = framesize
        self.frame_length = int(framesize * sample_rate)

    def get_vadvector_for_allframes(self, data):

        timevec = np.arange(int(len(data) / self.sample_rate / self.frame_length_s)) * self.frame_length_s
        num_blocks = int(np.floor(len(data) / self.frame_length))
        vad_vec = np.ones(shape=(num_blocks,))

        return timevec, vad_vec