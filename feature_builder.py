import librosa as lr
import numpy as np

class FeatureBuilder:
    mfcc = []
    zcr = []
    def __init__(self, track_name):
        self.audio, self.sample_rate = lr.load(track_name, res_type='kaiser_fast')

    def extract_mfcc(self):
        mfcc = lr.feature.mfcc(y=self.audio)
        mfccs_processed = np.mean(mfcc.T, axis=0)
        self.mfcc.append(mfccs_processed)
        return self

    def extract_zcr(self):
        zcr = lr.feature.zero_crossing_rate(self.audio)
        zcr_processed = np.mean(zcr.T, axis=0)
        self.zcr.append(float(zcr_processed[0]))
        return self

    def get_mfcc(self):
        return self.mfcc

    def get_zcr(self):
        return self.zcr


