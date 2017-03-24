from scipy.spatial import distance
from Utils import Utils
from sklearn import decomposition
from sklearn.preprocessing import  StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from RLP import RLP


class RLP_PCA(RLP,BaseEstimator, ClassifierMixin):
    def __init__(self, n, L):
        super(RLP_PCA, self).__init__(n, L)
        self.author_dictionary=None
        self.vector_space=None
        self.pca=None
        self.scaler=None


    def compare_profiles(self, profile1, profile2):
        # All n-grams in the top L of either profile.
        v1=Utils.vector_space_represent(Utils.top_L(profile1),self.author_dictionary)
        v2=Utils.vector_space_represent(Utils.top_L(profile2), self.author_dictionary)
        v1=v1.reshape(1,-1)
        v2=v2.reshape(1,-1)

        d1=self.scaler.transform(v1)
        d2=self.scaler.transform(v2)

        d1=self.pca.transform(d1)
        d2 = self.pca.transform(d2)
        return distance.cosine(d1, d2)



    def fit(self, documents, classes):
        super(RLP_PCA, self).fit(documents, classes)

        self.vector_space, self.author_dictionary = Utils.vector_space_model(self.author_profiles,self.language_profile)

        self.pca = decomposition.PCA()
        self.pca.fit(self.vector_space)

        self.scaler = StandardScaler()
        self.scaler.fit(self.vector_space)









