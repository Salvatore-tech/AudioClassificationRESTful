import librosa as lr
import numpy as np

class FeatureBuilder:
    mfcc = []
    zcr = []
    chroma = []
    rms = []
    mel = []
    def __init__(self, track_name):
        self.audio, self.sample_rate = lr.load(track_name, res_type='kaiser_fast')

    def extract_mfcc(self):
        mfcc = np.mean(lr.feature.mfcc(y=self.audio, sr=self.sample_rate).T, axis=0)
        self.mfcc.append(mfcc)
        return self

    def extract_zcr(self):
        zcr = np.mean(lr.feature.zero_crossing_rate(y=self.audio).T, axis=0)
        self.zcr.append(zcr[0])
        return self

    def extract_chroma(self):
        stft = np.abs(lr.stft(self.audio))
        chroma_stft = np.mean(lr.feature.chroma_stft(S=stft, sr=self.sample_rate).T, axis=0)
        self.chroma.append(chroma_stft)
        return self

    def extract_rms(self):
        rms = np.mean(lr.feature.rms(y=self.audio).T, axis=0)
        self.rms.append(rms[0])
        return self

    def extract_mel(self):
        mel = np.mean(lr.feature.melspectrogram(y=self.audio, sr=self.sample_rate).T, axis=0)
        self.mel.append(mel)
        return self

    def get_mfcc(self):
        return self.mfcc

    def get_zcr(self):
        return self.zcr

    def get_chroma(self):
        return self.chroma

    def get_rms(self):
        return self.rms

    def get_mel(self):
        return self.mel


