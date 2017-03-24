from Utils import Utils
from SCAP import SCAP
from CNG import CNG
from RLP_PCA import RLP_PCA
from sklearn.model_selection import GridSearchCV
from Author_Identifier import Author_Identifier as AI
import numpy as np

train_folder = '/Users/ridvansalih/Desktop/Thesis/Data/Portekiz-Python/test/'
test_folder = '/Users/ridvansalih/Desktop/Thesis/Data/Portekiz-Python/train/'


train_documents, train_classes = Utils.get_corpus(train_folder,1)
train_documents = np.array(train_documents, dtype=object)

test_documents, test_classes = Utils.get_corpus(test_folder,1)
test_documents = np.array(test_documents, dtype=object)


print("\n".join("TRAINING: Author {} appears {} times".format(*r) for r in enumerate(np.bincount(train_classes))))
print("\n".join("TESTING: Author {} appears {} times".format(*r) for r in enumerate(np.bincount(test_classes))))



parameters = [{'n': [3,4,5],
               'L': [1000,3000,5000],
               'alpha': [0.5, 1],
               'layer':  [100,200,300],
               'rbf':  [0.5, 1]},]

clf = GridSearchCV(AI(n=1, L=1,alpha=1,layer=1,rbf=1), parameters, cv=5, scoring='accuracy')
clf.fit(train_documents, train_classes)
for params, mean_score, scores in clf.grid_scores_:
    print("{}: {:.3f} (+/-{:.3f})".format(params, mean_score, scores.std() / 2))
print("The BEST model for Author Identifier found has n={}, L={}, alpha={}, layer={},rbf={}, score={}"
      .format(clf.best_estimator_.n,clf.best_estimator_.L,clf.best_estimator_.alpha, clf.best_estimator_.layer,
              clf.best_estimator_.rbf,clf.best_score_))


parameters = [{'n': [3,4,5],
               'L': [1000,3000,5000]},]

clf = GridSearchCV(RLP_PCA(n=1, L=1), parameters, cv=5, scoring='accuracy')
clf.fit(train_documents, train_classes)
for params, mean_score, scores in clf.grid_scores_:
    print("{}: {:.3f} (+/-{:.3f})".format(params, mean_score, scores.std() / 2))
print("The BEST model for RLP PCA found has n={}, L={}, score={}"
      .format(clf.best_estimator_.n, clf.best_estimator_.L,clf.best_score_))



