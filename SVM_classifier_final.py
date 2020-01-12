import _pickle as cPickle
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

input = open('discr_features.pkl', 'rb')
discr_features = cPickle.load(input)
input.close()

input = open('discr_labels.pkl', 'rb')
discr_labels = cPickle.load(input)
input.close()

#splitting data to train and test
X_train, X_test, y_train, y_test = train_test_split(discr_features, discr_labels, test_size = 0.5, random_state=0)


clf = svm.SVC(kernel='rbf')
params = {"C":[0.1, 1, 10], "gamma": [0.1, 0.01, 0.001]}
grid_search = GridSearchCV(clf, params)
grid_search.fit(X_train, y_train)


clf_predictions = grid_search.predict(X_test)

#printing results - confusion matrix and classification report
print(confusion_matrix(y_test, clf_predictions, labels=[0,1,2]))
print(classification_report(y_test, clf_predictions))



