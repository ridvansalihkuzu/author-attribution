from Utils import Utils
from RLP_PCA import RLP_PCA
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt


train_folder = '/Users/ridvansalih/Desktop/Thesis/Data/Portuguese-Authors/train/'
test_folder = '/Users/ridvansalih/Desktop/Thesis/Data/PPortuguese-Authors/test/'

train_documents, train_classes = Utils.get_corpus(train_folder,1)
train_documents = np.array(train_documents, dtype=object)

test_documents, test_classes = Utils.get_corpus(test_folder,1)
test_documents = np.array(test_documents, dtype=object)


print("\n".join("TRAINING: Author {} appears {} times".format(*r) for r in enumerate(np.bincount(train_classes))))
print("\n".join("TESTING: Author {} appears {} times".format(*r) for r in enumerate(np.bincount(test_classes))))

n_classes=100

train_classes = label_binarize(train_classes, classes=np.arange(n_classes))
test_classes = label_binarize(test_classes, classes=np.arange(n_classes))


classifier = OneVsRestClassifier(RLP_PCA(5,2000))
y_score = classifier.fit(train_documents, train_classes).fit(test_documents)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_classes[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_classes.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (AUC = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (AUC = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for C10 Dataset' )
plt.legend(loc="lower right")
plt.show()