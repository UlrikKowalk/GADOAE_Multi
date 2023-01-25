import numpy as np
import matplotlib.pyplot as plt

__author__ = "Jule Pohlhausen jule.pohlhausen@jade-hs.de"
__copyright__ = "Copyright (C) 2022 Jule Pohlhausen"
__version__ = "0.0.1"
__info__ = '''VAD implementation of Marzinzik and Kollmeier (2002) 
            based on Matlab implementation of Thomas Rohdenburg (2009)
            for single channel audio'''


class MaKo_VAD:
    def __init__(self, samplerate, framesize = 0.008):
        """ init 
        """
        self.samplerate = samplerate
        self.framesize = framesize
        self.nr_of_output_features = 1
        self.blockSize = int(np.floor(self.framesize*self.samplerate))
        self.nfft = int(2**(2*self.blockSize + 1).bit_length())
        self.zeroLength = int(self.nfft/2-self.blockSize)
        self.window = np.hanning(self.blockSize*2)
        self.oldBlock = np.zeros(self.blockSize)
        self.oldOutBlock = np.zeros(self.nfft)
        self.frameRate = self.samplerate/self.blockSize

        # init noise estimation params (cf. noiseEst1.ini)
        self.cutOff      = 2000      # 2 kHz, due to IEEE Paper 2002 by Mark Marzinzik
        self.tau         = 1.5       # Marzinzik: 3s, Als guter Wert hat sich eine Release Time von tau = 1.5 Sekunden ergeben 25.09.2009
        self.tauPSD      = 32/1000   # Marzinzik: 32 ms Glättung
        self.tauNoise    = 200/1000  # Marzinzik: n.a.   TR: 200 ms Glättung des Rauschens (siehe alter Ephraim Malah Algo)
        self.eta         = 7         # Marzinzik: 5dB, Als guter Wert hat sich ein Dynamik-Wert eta von 6dB oder höher ergeben 25.09.2009
        self.fraction    = 0.1       # Marzinzik: 0.1, taken from IEEE Paper 2002
        self.noiseFactor = 1
        self.forceSpeechPauselfime = 0  # force Speechpause for x seconds for faster adaptation
        self.alpha       = np.exp(-1/(self.tau*self.frameRate))
        self.alphaPSD    = np.exp(-1/(self.tauPSD*self.frameRate))
        self.alphaNoise  = np.exp(-1/(self.tauNoise*self.frameRate))
        self.noiseEst    = np.zeros(int(self.nfft/2+1))

        # init variables
        self.max_E          = 0
        self.max_E_lp       = 0
        self.max_E_hp       = 0
        self.min_E          = 0
        self.min_E_lp       = 0
        self.min_E_hp       = 0
        self.delta          = 0
        self.delta_lp       = 0
        self.delta_hp       = 0
        self.delta_lp_min   = 0
        self.delta_hp_min   = 0
        self.cutOffIndex    = int(np.floor(self.cutOff*self.nfft/self.samplerate)+1)
        self.speechPause    = 0
        self.counter        = 0
        self.oldsumPSD      = 0
        self.oldsumPSDLp    = 0
        self.oldsumPSDHp    = 0
        self.speechPauseCount = np.floor(self.forceSpeechPauselfime*self.frameRate)


    def get_algoname(self):
        return "vad_MaKo"

    def need_features(self):
        return False

    def olatime2freq_Process(self, data_block):
        # concatenate 2 blocks (old+new)
        actBlock = np.concatenate((self.oldBlock, data_block))*self.window
        self.oldBlock = data_block
        spec = np.fft.fft(np.concatenate((np.zeros(self.zeroLength), actBlock, np.zeros(self.zeroLength))))
        return spec

    def noiseEst1_Process(self, spec):
        # Frequenzblock von 1 bis nfft/2 quadriert
        psdX = abs(spec[:int(self.nfft/2+1)])**2 

        # estimation part
        sumPSD = 10*np.log10(sum(psdX))
        sumPSD = self.attack_release_filter(sumPSD, self.oldsumPSD, 0, self.alphaPSD)
        self.oldsumPSD = sumPSD
        self.max_E = self.attack_release_filter(sumPSD, self.max_E, 0, self.alpha)
        self.min_E = self.attack_release_filter(sumPSD, self.min_E, self.alpha, 0)

        sumPSDLp  = 10*np.log10(sum(psdX[:self.cutOffIndex]))
        sumPSDLp = self.attack_release_filter(sumPSDLp, self.oldsumPSDLp, 0, self.alphaPSD)
        self.oldsumPSDLp = sumPSDLp
        self.max_E_lp = self.attack_release_filter(sumPSDLp, self.max_E_lp, 0, self.alpha)
        self.min_E_lp = self.attack_release_filter(sumPSDLp, self.min_E_lp, self.alpha, 0)

        sumPSDHp  = 10*np.log10(sum(psdX[self.cutOffIndex-1:]))
        sumPSDHp = self.attack_release_filter(sumPSDHp, self.oldsumPSDHp, 0, self.alphaPSD)
        self.oldsumPSDHp = sumPSDHp
        self.max_E_hp = self.attack_release_filter(sumPSDHp, self.max_E_hp, 0, self.alpha)
        self.min_E_hp = self.attack_release_filter(sumPSDHp, self.min_E_hp, self.alpha, 0)

        # initialization for 50 blocks
        if self.counter<50:
            self.max_E = sumPSD
            self.min_E = sumPSD
            self.max_E_lp = sumPSDLp
            self.min_E_lp = sumPSDLp
            self.max_E_hp = sumPSDHp
            self.min_E_hp = sumPSDHp
            self.noiseEst = psdX
        self.counter+=1

        # calc dynamics
        self.delta           = self.max_E-self.min_E
        self.delta_lp        = self.max_E_lp-self.min_E_lp
        self.delta_hp        = self.max_E_hp-self.min_E_hp
        self.delta_lp_min    = sumPSDLp-self.min_E_lp
        self.delta_hp_min    = sumPSDHp-self.min_E_hp

        # decision part
        self.decision()
        if self.counter<self.speechPauseCount:
            # force speech pause
            self.speechPause = 4

        # update noise with current signal spectrum
        if self.speechPause:
            self.noiseEst = self.alphaNoise * self.noiseEst + (1-self.alphaNoise)*psdX

        return

    def attack_release_filter(self, newVal, oldVal, attack, release):
        if newVal>oldVal:
            # attack
            out = attack * oldVal + (1-attack)*newVal
        else:
            # release
            out = release * oldVal + (1-release)*newVal
        return out

    def log10q(self, vec):
        # quiet log10 (without the warning "log of zero")
        mask = vec==0
        vec[mask] = 1
        out = np.log10(vec)
        out[mask] = -np.inf
        return

    def decision(self):
        # decision part
        condition01 = self.delta_lp < self.eta
        condition02 = self.delta_hp < self.eta

        condition03 = self.delta_lp > 2 * self.eta
        condition04 = self.delta_hp > 2 * self.eta

        condition05 = (self.oldsumPSD - self.min_E) < (0.5 * self.delta)

        condition06 = self.delta_lp_min <   self.fraction * self.delta_lp
        condition07 = self.delta_lp_min < 2*self.fraction * self.delta_lp
        condition08 = self.delta_lp_min <             0.5 * self.delta_lp

        condition09 = self.delta_hp_min <   self.fraction * self.delta_hp
        condition10 = self.delta_hp_min < 2*self.fraction * self.delta_hp
        condition11 = self.delta_hp_min <             0.5 * self.delta_hp 

        # default: No Speech Pause
        self.speechPause = 0
        if condition01 and condition02:
            # Dynamic Speech Pause
            self.speechPause = 1
        else:
            if condition01:
                if condition09 and condition05:
                    # HP Speech Pause
                    self.speechPause = 2
            else:
                if condition06:
                    if condition02:
                        if condition05:
                            # LP Speech Pause
                            self.speechPause = 3
                    else:
                        if condition04:
                            if condition10:
                                # LP Speech Pause
                                self.speechPause = 3
                        else:
                            if condition11:
                                # LP Speech Pause
                                self.speechPause = 3
                            else:
                                if condition09:
                                    if condition03:
                                        if condition07:
                                            # HP Speech Pause
                                            self.speechPause = 2
                                    else:
                                        if condition08:
                                            # HP Speech Pause
                                            self.speechPause = 2
                else:
                    if condition09:
                        if condition03:
                            if condition07:
                                # HP Speech Pause
                                self.speechPause = 2
                        else:
                            if condition08:
                                # HP Speech Pause
                                self.speechPause = 2
        return
        
    def get_vadvector_for_allframes(self, data):
        timevec = np.arange(int(len(data)/self.samplerate/self.framesize))*self.framesize
        speechPause_vec = np.zeros_like(timevec)
        nBlocks = timevec.size

        # rearrange data
        data_blocks = data[0:int(nBlocks*self.blockSize)].reshape(nBlocks,self.blockSize)

        for block in range(nBlocks):
            # Zeitsignalausschnitt fenstern und in den Frequenzbereich transformieren
            spec = self.olatime2freq_Process(data_blocks[block, :])

            self.noiseEst1_Process(spec)
            #outSpec = np.sqrt(self.noiseEst)*np.exp(1j*np.angle(spec))
            #outTime = self.olafreq2time_Process(outSpec)
            
            # get VAD decision
            speechPause_vec[block] = self.speechPause

        vad_vec = speechPause_vec == 0

        return timevec, vad_vec