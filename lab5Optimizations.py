import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_digits()
x,y = data.data, data.target

randomState = 1
testSplit = 0.3

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSplit, random_state=randomState)

seed = 2
kfold_splits = 5

kf = KFold(n_splits=kfold_splits, random_state=seed, shuffle=True)

meanScores = []

kk = np.arange(1, 50, 2)
for k in kk:
    classifier = KNeighborsClassifier(n_neighbors=k)
    
    f = cross_val_score(classifier, X_train, y_train, cv=kf)
    
    meanScores.append(np.mean(f))
    
plt.plot(kk, meanScores)
# plt.show()
print(best := kk[np.argmax(meanScores)])
bestclassifier = KNeighborsClassifier(n_neighbors=best)

pred = bestclassifier.fit(X_train, y_train).predict(X_test)

print(accuracy_score(y_test, pred))

    