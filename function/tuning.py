from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def tuning(X_train, y_train) :
    
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = dict(grid_search.best_params_)
    best_score = grid_search.best_score_

    return best_params,best_score