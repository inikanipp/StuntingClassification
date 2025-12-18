from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def modeling(X_train, y_train, X_test, y_test, n, metric, weight ) :
    knn = KNeighborsClassifier(n_neighbors=n, metric=metric, weights=weight)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return confusion_matrix(y_test,y_pred), accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)