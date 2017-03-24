from scipy.spatial import distance
from collections import defaultdict
import numpy as np
from SCAP import SCAP
from Utils import Utils


class RLP(SCAP):
    def __init__(self, n, L):
        super(RLP, self).__init__(n, L)
        self.language_profile = None

    def compare_profiles(self, profile1, profile2):
        # All n-grams in the top L of either profile.
        ngrams = set(Utils.top_L(profile1,self.L).keys() | Utils.top_L(profile1,self.L).keys())
        # Profile vector for profile 1
        d1 = np.array([profile1.get(ng, 0.) for ng in ngrams])
        # Profile vector for profile 2
        d2 = np.array([profile2.get(ng, 0.) for ng in ngrams])
        try:
            return distance.cosine(d1, d2)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            print(template.format(type(ex).__name__, ex.args))
            return float('inf')

    def fit(self, documents, classes):
        self.language_profile = self.create_profile(documents)
        super(RLP, self).fit(documents, classes)

    def create_profile(self, documents):
        # Creates a profile of a document or list of documents.
        if isinstance(documents, str):
            # documents can be either a list of documents, or a single document.
            # if it's a single document, convert to a list
            documents = [documents, ]
        # profile each document independently
        profiles = (Utils.count_ngrams(document, self.n, normalise=False)
                    for document in documents)
        # Merge the profiles
        main_profile = defaultdict(float)
        for profile in profiles:
            for ngram in profile:
                main_profile[ngram] += profile[ngram]
        # Normalise the profile
        num_ngrams = float(sum(main_profile.values()))
        for ngram in main_profile:
            main_profile[ngram] /= num_ngrams
        if self.language_profile is not None:
            # Recentre profile.
            for key in main_profile:
                main_profile[key] = main_profile.get(key, 0) - self.language_profile.get(key, 0)
        # Note that the profile is returned in full, as exact frequencies are used
        # in comparing profiles (rather than chopped off)
        return main_profile
