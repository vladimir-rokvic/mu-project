from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

KNN_PARAMS = {
        "n_neighbors": [1, 2, 3, 5, 7, 6, 9, 10, 15, 25],
        "weights": ['uniform', 'distance'],
        "metric": ['euclidean', 'manhattan', 'cosine', 'minkowski', 'chebyshev'],
        "p": [1, 2, 3]
        }

test_speakers = [4, 8, 14, 26, 28]
df = pd.read_csv('agregated_features_2.csv')

X_train = df.loc[~df["spk"].isin(test_speakers)]
y_train = X_train["y"]
X_train = X_train.drop(["spk", "y"], axis=1)

X_test = df.loc[df["spk"].isin(test_speakers)]
y_test = X_test["y"]
X_test = X_test.drop(["spk", "y"], axis=1)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("Broj obelezja:", X_train.shape[1])

lda = LinearDiscriminantAnalysis(n_components=29)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

knn = KNeighborsClassifier()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
        knn,
        KNN_PARAMS,
        scoring = 'accuracy',
        n_jobs=-1,
        verbose=1,
        cv=cv
        )
search.fit(X_train, y_train)
best = search.best_estimator_
y_test_predict = best.predict(X_test)
y_train_predict = best.predict(X_train)

print(f"Best params:    {search.best_params_}")
print(f"Best CV acc:    {search.best_score_:.4f}")
print(f"Train accuracy: {accuracy_score(y_train, y_train_predict):.2f}")
print(f"Test accuracy:  {accuracy_score(y_test, y_test_predict):.2f}")
