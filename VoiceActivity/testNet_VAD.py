import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy.signal as sc
import soundfile as sf
import torch.nn as nn
import torch
import librosa
import librosa.display
# from zmq import device
#
# import segmentstools as st

# each algorithm is defined by one NN and a specific set of features
# changing one of them is a new algorith with ist own name

algoname = 'testNet'

class testNet_features():
    def __init__(self, samplerate_Hz):
        self.samplerate = samplerate_Hz
        self.framesize_s = 0.032
        self.stepsize_s = 0.032
        self.nr_of_mfccs = 13
        self.nr_of_output_features = self.nr_of_mfccs
        self.fftsize = 1024
        self.winlen = int(self.framesize_s*self.samplerate)
        self.hoplen = int(self.stepsize_s*self.samplerate)
        

    def get_features(self, data):
        # here feature extraction
        featdata = librosa.feature.mfcc(y=data, sr=self.samplerate, n_fft= self.fftsize, n_mfcc=self.nr_of_mfccs, win_length=self.winlen,
                                        hop_length=self.hoplen)
        return featdata
        
    def get_features_for_one_file(self, filename):
        data, sampleratefile = sf.read(str(filename))
        if (sampleratefile != self.samplerate):
            print("The samplerate of the file is differnet to the desired sample rate (SRC aplied)")
            data = sc.resample_poly(data,self.samplerate/math.gcd(self.samplerate, sampleratefile),sampleratefile/math.gcd(self.samplerate, sampleratefile))

        featdata = self.get_features(data)
        return featdata        

    def get_algoname(self):
        return "feat_" + algoname
        
    def save_features_for_one_file(self, filename, recompute = False):

        basefilename = os.path.splitext(str(filename))[0]
        algoname = self.get_algoname()
        fullpathtosegmentfile = basefilename + '.' + algoname 
        if (os.path.isfile(fullpathtosegmentfile+'.npy')) & (recompute == False):
            featdata = np.load(fullpathtosegmentfile+'.npy')
            return featdata
        featdata = self.get_features_for_one_file(filename)
        np.save(fullpathtosegmentfile, featdata)
        return featdata
        
    def plot_feature(self, featdata):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(featdata, x_axis='s', ax=ax, sr = self.samplerate, hop_length= self.hoplen,)
        fig.colorbar(img, ax=ax)
        ax.set(title='MFCC')
        plt.show()
        
def saveNNParams(model):
    # print("save NN")
    path = os.path.abspath(__file__)
    path_to_data = os.path.dirname(path)
    torch.save(model.state_dict(), os.path.join(path_to_data,algoname+'_nndata.pt'))

def loadNNParams(model):
    # print("load NN")
    path = os.path.abspath(__file__)
    path_to_data = os.path.dirname(path)
    model.load_state_dict(torch.load(os.path.join(path_to_data,algoname+'_nndata.pt')))
    return model


class testNet_NN(nn.Module):
    def __init__(self):
        super(testNet_NN, self).__init__()
        #define net
        featinfo = testNet_features(1)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(featinfo.nr_of_output_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
            
    def forward(self, featdata):
        logits = self.linear_relu_stack(featdata)
        return logits
        

class testNet_VAD():
    def __init__(self, samplerate, framesize):
        """ init 
        """
        self.framesize = framesize
        self.samplerate = samplerate
        self.decision_threhold = 0.7 #0.5
        self.device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Using {self.device} device")
        model = testNet_NN().to(self.device)
        self.features = testNet_features(self.samplerate)
        
        self.model = loadNNParams(model)
        self.model.eval()
        
    def get_algoname(self):
        return "vad_" + algoname
    
    def need_features(self):
        return True
    
    def get_decision_for_oneframe(self, data):
        curData = torch.from_numpy(data).to(self.device)
        speech_prob = self.model(curData)
        if speech_prob > self.decision_threhold:
            return 1
        else:
            return 0
        
    def get_vadvector_for_allframes(self, data):
        features = self.features.get_features(data=data)
        nr_of_data  = data.shape[0]
        num_frames = int(np.floor(nr_of_data / (self.framesize*self.samplerate)))
        timevec = np.arange(num_frames)*self.framesize
        vad_vec = np.zeros_like(timevec)

        model_output = self.model(torch.from_numpy(features).transpose(1, 0).to(self.device))

        # # Schleife über alle Daten
        for kk in range(num_frames):
            if model_output[kk] > self.decision_threhold:
                vad_vec[kk] = 1
            else:
                vad_vec[kk] = 0

        return timevec, vad_vec
        
    
    