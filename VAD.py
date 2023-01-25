import numpy as np

from VoiceActivity import DenkVAD
from VoiceActivity import MaKo_VAD
from VoiceActivity import Energy_VAD
from VoiceActivity import testNet_VAD
from VoiceActivity import None_VAD


class VAD:

    def __init__(self, name, sample_rate, frame_length, device):

        self.name = name
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        if name == 'Denk':
            self.vad = DenkVAD.DenkVAD(samplerate=self.sample_rate,
                                       framesize=self.frame_length,
                                       stepsize=self.frame_length)
        elif name == 'MaKo':
            self.vad = MaKo_VAD.MaKo_VAD(samplerate=self.sample_rate,
                                         framesize=self.frame_length)
        elif name == 'Energy':
            self.vad = Energy_VAD.Energy_VAD(sample_rate=self.sample_rate,
                                             framesize=self.frame_length, device=device)
        elif name == 'testNet':
            self.vad = testNet_VAD.testNet_VAD(samplerate=self.sample_rate,
                                               framesize=self.frame_length)
        elif name == 'None':
            self.vad = None_VAD.None_VAD(sample_rate=self.sample_rate,
                                         framesize=self.frame_length)

    def get_vad(self, data):
        time_vector, vad_vector = self.vad.get_vadvector_for_allframes(data)

        return time_vector, vad_vector
