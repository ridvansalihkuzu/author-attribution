from SCAP import SCAP
import numpy as np
class CNG(SCAP):
    def compare_profiles(self, profile1, profile2):
        ngrams = set(profile1.keys() | profile2.keys())
        d1 = np.array([profile1.get(ng, 0.) for ng in ngrams])
        d2 = np.array([profile2.get(ng, 0.) for ng in ngrams])
        return np.mean(4 * np.square((d1 - d2) / (d1 + d2 + 1e-16)))

