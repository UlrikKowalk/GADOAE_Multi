import numpy as np

__author__ = "Jule Pohlhausen jule.pohlhausen@jade-hs.de"
__copyright__ = "Copyright (C) 2022 Jule Pohlhausen"
__version__ = "0.0.1"
__info__ = "VAD implementation of Denk et al. (2014)"

from matplotlib import pyplot as plt


class DenkVAD:
    def __init__(self, samplerate, framesize = 0.02, stepsize = 0.01):
        """ init 
        """
        self.samplerate = samplerate
        self.framesize = framesize
        self.stepsize = stepsize 
        self.nr_of_output_features = 1
        self.winlen = int(self.framesize*self.samplerate)
        self.hoplen = int(self.stepsize*self.samplerate)
        self.SNR1 = -1000
        self.SNR2 = -1000
        self.sub_weight = 0.7

    def get_algoname(self):
        return "vad_denk"

    def need_features(self):
        return False

    def get_rms(self, data):
        if data.ndim > 1:
            #Row vector with RMS of each frame
            rms = np.sqrt(np.mean(data**2, axis=1))
        else:
            rms = np.sqrt(np.mean(data**2))
        return rms

    def init_rms_threshold(self, vRMS):
        # Defining 1st Threshold value using Relation between 
        # highest and lowest RMS (~Dynamic of signal)
        nMax = np.max(vRMS)         
        nMin = np.min(vRMS)
        nRel = nMax/nMin

        if nRel < 1/0.3: #Safety for low dynamics
            self.rms_thres = nMin*1.2
        else:
            if np.isinf(nRel):
                self.rms_thres = nMax*0.2
            else:
                self.rms_thres = nMax*0.3
        return 
    
    def best_thr(self):
    # Best threshold function - from taylor fit
        if self.SNR2 > 30:
            self.rms_thres = 0.07
        else:
            self.rms_thres =  0.79379 - 0.0218*self.SNR2  - 0.00317*self.SNR2**2 + 1.296e-4*self.SNR2**3 + 7.116e-6*self.SNR2**4 - 6.0044e-7*self.SNR2**5 + 1.627e-8*self.SNR2**6 - 1.986e-10*self.SNR2**7 + 9.247e-13*self.SNR2**8 + 3.96e-26*self.SNR2**9

    def get_vadvector_for_allframes(self, data):
        timevec = np.arange(int(len(data)/self.samplerate/self.framesize))*self.framesize

        # rearrange data
        data_blocks = data[0:int(timevec.size*self.winlen)].reshape(timevec.size,self.winlen)
        n_blocks = len(timevec)

        # init RMS
        vRMS = self.get_rms(data_blocks)

        self.init_rms_threshold(vRMS)

        # first VAD prediction
        vad_vec = vRMS > self.rms_thres

        # iterative SNR estimation
        for kk in range(100):
            #print('iteration: ', kk)
            # Estimating RMS of Speech only
            mSpeech = data_blocks[vad_vec == 1, :]
            rmss = self.get_rms(mSpeech.reshape(mSpeech.size))
    
            # Estimating RMS of Noise only
            mNoise = data_blocks[vad_vec == 0, :]
            rmsn = self.get_rms(mNoise.reshape(mNoise.size))
            
            # SNR estimation
            self.SNR2 = 20*np.log10((rmss-(self.sub_weight*rmsn))/rmsn)
            
            # Safety mechanisms!
            if np.isnan(self.SNR2):
                self.SNR2 = 80
            self.SNR2  = np.min([np.max([self.SNR2, -30]), 80])

            # Convergence if Estimation of SNR does not change any more
            if np.abs((self.SNR1-self.SNR2)) < 0.1: 
                break

            # Saving old SNR
            self.SNR1 = self.SNR2
            
            # Determining new threshold for VAD
            self.best_thr()
            
            # next step in Speech activity detection using new threshold
            vad_vec = vRMS > self.rms_thres*rmss   

            # Avoiding false off-switching
            # Eliminating off-switching processes smaller than 20 frames
            for ii in range(0, n_blocks-20):
                falseswitch = 0
                # Detecting egative flank between ii and ii+1
                if (vad_vec[ii] and not(vad_vec[ii+1])):  
                    # Checking next 20 index instances after ii
                    for jj in range((ii+1), (ii+21)):                
                        if vad_vec[jj]:
                            # Switching on, if next positive flank is within 20 frames 
                            # (interpretation as false off-switching)
                            falseswitch = 1 
                            # Point after positive flank             
                            endpoint = jj                
                    
                    # If falseswitch should be detected, filling up the gap with ones
                    if falseswitch: 
                        vad_vec[(ii+1):endpoint] = 1
        
        return timevec, vad_vec