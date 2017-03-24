from collections import defaultdict
import numpy as np
from Utils import Utils
from sklearn import decomposition
from ELM import ELMClassifier
from sklearn.feature_extraction.text import TfidfTransformer as TF_IDF
from sklearn.base import BaseEstimator, ClassifierMixin

class Author_Identifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n, L,layer,alpha,rbf):
        self.author_dictionary=None
        self.ELM=None
        self.scaler=None
        self.n = n
        self.L = L
        self.layer=layer
        self.alpha=alpha
        self.rbf=rbf
        self.LSA=None
        self.TF_IDF=None
        self.language_profile=None

    def fit(self, documents, classes):
        self.language_profile = self.create_profile(documents)
        document_profiles = [Utils.top_L(self.create_profile(cur_docs),self.L) for cur_docs in documents]

        vector_space, self.author_dictionary = Utils.vector_space_model(document_profiles,self.language_profile)

        self.TF_IDF=TF_IDF(norm=u'l2',sublinear_tf=False)
        vector_space=self.TF_IDF.fit_transform(vector_space)

        le=len(document_profiles)
        self.LSA = decomposition.TruncatedSVD(n_components=le)
        vector_space=self.LSA.fit_transform(vector_space)

        self.ELM = ELMClassifier(self.layer, self.alpha, self.rbf,activation_func='multiquadric')
        self.ELM.fit(vector_space, classes)

        return self


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
        return main_profile

    def predict(self, documents):
        # Predict which of the authors wrote each of the documents
        predictions = np.array([self.predict_single(document) for document in documents])
        return predictions

    def predict_single(self, document):
        # Predicts the author of a single document
        # Profile of current document
        profile = self.create_profile(document)

        #Vector Space Model projection of the profile
        vsm = (Utils.vector_space_represent(Utils.top_L(profile,self.L), self.author_dictionary))
        vsm = self.TF_IDF.transform(vsm)
        vsm = self.LSA.transform(vsm)
        prediction=self.ELM.predict(vsm)

        return prediction


