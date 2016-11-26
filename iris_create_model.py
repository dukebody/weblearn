# taken from a sklearn official example

from sklearn import svm, datasets
from sklearn.externals import joblib

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

# we create an instance of SVM and fit out data.
C = 1.0  # SVM regularization parameter
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C, probability=True)\
          .fit(X, y)
joblib.dump(rbf_svc, 'models/iris.pickle', compress=3)
