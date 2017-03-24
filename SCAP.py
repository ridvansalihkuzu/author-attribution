from operator import itemgetter
from sklearn.base import BaseEstimator, ClassifierMixin  # For using scikit-learn's test framework. Ignore for now
from collections import defaultdict
import numpy as np
from Utils import Utils

class SCAP(BaseEstimator, ClassifierMixin):
    def __init__(self, n, L):
        self.n = n
        self.L = L
        self.author_profiles = None  # Will be trained by fit()
        self.classes=None

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
        num_ngrams = sum(main_profile.values())
        for ngram in main_profile:
            main_profile[ngram] /= num_ngrams
        # Return the profile with only the top L n-grams
        return Utils.top_L(main_profile,self.L)

    def compare_profiles(self, profile1, profile2):
        # Number of n-grams in both profiles, divided by L
        similarity = len(set(profile1.keys()) & set(profile2.keys())) / float(self.L)
        # Slight edge case here, similarity could be higher than 1.
        # Just make it equal to 1 if it is over.
        similarity = min(similarity, 1.0)
        distance = 1 - similarity
        return distance

    def fit(self, documents, classes):
        self.classes=classes
        author_documents = ((author, [documents[i] for i in range(len(documents)) if classes[i] == author]) for author in set(classes))
        self.author_profiles = {author: self.create_profile(cur_docs) for author, cur_docs in author_documents}
        print("Model fitting completed for n={}, L={}"
              .format(self.n,
                      self.L,))

    def predict(self, documents):
        # Predict which of the authors wrote each of the documents
        print("Prediction started for n={}, L={}"
              .format(self.n,
                      self.L, ))
        predictions = np.array([self.predict_single(document) for document in documents])
        return predictions

    def decision_function(self, documents):
        # Decisison Function for the Base Estimator (inherited class)
        predictions = np.array([self.predict_single(document) for document in documents])
        return predictions

    def predict_single(self, document):
        # Predicts the author of a single document
        # Profile of current document
        profile = self.create_profile(document)
        # Distance from document to each author
        distances = [(author, self.compare_profiles(profile, self.author_profiles[author]))
                     for author in self.author_profiles]
        # Get the nearest pair, and the author from that pair
        prelist = sorted(distances, key=itemgetter(1))
        prediction = prelist[0][0]

        return prediction

