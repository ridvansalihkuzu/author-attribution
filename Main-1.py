from Utils import Utils
from SCAP import SCAP
from CNG import CNG
from RLP_PCA import RLP_PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from Author_Identifier import Author_Identifier as AI
import numpy as np

#The Purpose of the main function is to compare author identification performance of profile-based
# and instance-based approach on Routers English News Corpus - C10 in this case

train_folder = '/Users/ridvansalih/Desktop/Thesis/Data/C10 yedek/train/'
test_folder = '/Users/ridvansalih/Desktop/Thesis/Data/C10 yedek/test/'

train_documents, train_classes = Utils.get_corpus(train_folder,1)
train_documents = np.array(train_documents, dtype=object)

test_documents, test_classes = Utils.get_corpus(test_folder,1)
test_documents = np.array(test_documents, dtype=object)


print("\n".join("TRAINING: Author {} appears {} times".format(*r) for r in enumerate(np.bincount(train_classes))))
print("\n".join("TESTING: Author {} appears {} times".format(*r) for r in enumerate(np.bincount(test_classes))))


#Validating and testing instance-based authorship attribution approach which is based on TF-IDF weighting of
#author vector space model followed by latent semantic analysis and extreme learning machine.

parameters = [{'n': [5],
               'L': [1000],
               'alpha': [0.5],
               'layer':  [200],
               'rbf':  [1]},]

clf = GridSearchCV(AI(n=1, L=1,alpha=1,layer=1,rbf=1), parameters, cv=5, scoring='accuracy')
clf.fit(train_documents, train_classes)
for params, mean_score, scores in clf.grid_scores_:
    print("{}: {:.3f} (+/-{:.3f})".format(params, mean_score, scores.std() / 2))
print("The BEST model for Author Identifier found has n={}, L={}, alpha={}, layer={},rbf={}, score={}"
      .format(clf.best_estimator_.n,clf.best_estimator_.L,clf.best_estimator_.alpha, clf.best_estimator_.layer,
              clf.best_estimator_.rbf,clf.best_score_))


model = AI(n=clf.best_estimator_.n, L=clf.best_estimator_.L,alpha=clf.best_estimator_.alpha,
           rbf=clf.best_estimator_.rbf,layer=clf.best_estimator_.layer)
model.fit(train_documents, train_classes)
y_true, y_pred = test_classes, model.predict(test_documents)
print(classification_report(y_true, y_pred, digits=3))




#Validating and testing profile-based authorship attribution approach which is based on cosine similarity of
# PCA transformed author vector space model created by recentered local profiles of each author.

parameters = [{'n': [4,5],
               'L': [1000,5000]},]

clf = GridSearchCV(RLP_PCA(n=1, L=1), parameters, cv=5, scoring='accuracy')
clf.fit(train_documents, train_classes)
for params, mean_score, scores in clf.grid_scores_:
    print("{}: {:.3f} (+/-{:.3f})".format(params, mean_score, scores.std() / 2))
print("The BEST model for RLP PCA found has n={}, L={}, score={}"
      .format(clf.best_estimator_.n, clf.best_estimator_.L,clf.best_score_))

model = RLP_PCA(n=clf.best_estimator_.n, L=clf.best_estimator_.L)
model.fit(train_documents, train_classes)
y_true, y_pred = test_classes, model.predict(test_documents)
print(classification_report(y_true, y_pred, digits=3))



