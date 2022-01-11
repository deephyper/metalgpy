# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# * original code
clf = KNeighborsClassifier(3)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test) # outputs 0.925

# * MetalgPy
import metalgpy as mpy

clf = mpy.meta(KNeighborsClassifier)(mpy.Int(3,5))
clf = clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

score.freeze([3]) # we choose "clf = KNeighborsClassifier(3)"
score = score.evaluate() # outputs 0.925