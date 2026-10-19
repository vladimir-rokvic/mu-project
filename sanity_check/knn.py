from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, precision_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

words = ["nula", "jedan", "dva", "tri", "četiri", "pet",
         "šest", "sedam", "osam", "devet",
         "napred", "nazad", "levo", "desno", "gore", "dole",
         "potvrdi", "odustani", "obriši", "pošalji", "dalje", "početak", "kraj",
         "ponedeljak", "utorak", "sreda", "četvrtak", "petak", "subota", "nedelja"]

KNN_PARAMS = {
        "knn__n_neighbors": [5, 10, 15, 25],
        "knn__weights": ['uniform', 'distance'],
        "knn__metric": ['euclidean', 'manhattan', 'cosine'],
        }

def no_pca_lda(cv):
    pipe = Pipeline([
         ("scale", StandardScaler()),
         ("knn", KNeighborsClassifier())
    ])
    search = GridSearchCV(
            pipe,
            KNN_PARAMS,
            scoring = 'accuracy',
            n_jobs=-1,
            cv=cv
        )
    search.fit(X_train, y_train)
    
    print("\nSamo StandardScaler")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    
    best_rf = search.best_estimator_
    y_pred  = best_rf.predict(X_test)
    train_acc = accuracy_score(y_train, best_rf.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

def only_lda(cv):
    pipe = Pipeline([
         ("scale", StandardScaler()),
         ("lda", LinearDiscriminantAnalysis()),
         ("knn", KNeighborsClassifier())
    ])
    search = GridSearchCV(
            pipe,
            KNN_PARAMS,
            scoring = 'accuracy',
            n_jobs=-1,
            cv=cv
        )
    search.fit(X_train, y_train)
    
    print("\nSamo LDA")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    
    best_rf = search.best_estimator_
    y_pred  = best_rf.predict(X_test)
    train_acc = accuracy_score(y_train, best_rf.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

def only_pca(cv, n):
    pipe = Pipeline([
         ("scale", StandardScaler()),
         ("pca", PCA(n_components=n, whiten=True)),
         ("knn", KNeighborsClassifier())
    ])
    search = GridSearchCV(
            pipe,
            KNN_PARAMS,
            scoring = 'accuracy',
            n_jobs=-1,
            cv=cv
        )
    search.fit(X_train, y_train)
    
    print("\nPCA:", n)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    
    best_rf = search.best_estimator_
    y_pred  = best_rf.predict(X_test)
    train_acc = accuracy_score(y_train, best_rf.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

def pca_lda(cv, n):
    pipe = Pipeline([
         ("scale", StandardScaler()),
         ("pca", PCA(n_components=n, whiten=True)),
         ("lda", LinearDiscriminantAnalysis()),
         ("knn", KNeighborsClassifier())
    ])
    search = GridSearchCV(
            pipe,
            KNN_PARAMS,
            scoring = 'accuracy',
            n_jobs=-1,
            cv=cv
        )
    search.fit(X_train, y_train)
    
    print("\nPCA:", n)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    
    best_rf = search.best_estimator_
    y_pred  = best_rf.predict(X_test)
    train_acc = accuracy_score(y_train, best_rf.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

test_speakers = [4, 8, 14, 26, 28]

df_timed = pd.read_csv('geometric_timed.csv')
df_timed = df_timed.drop(["y", "spk"], axis=1)

pca_df = pd.read_csv('features_pca.csv')

df = pd.concat([df_timed, pca_df], axis=1)

X_train = df.loc[~df["spk"].isin(test_speakers)]
y_train = X_train["y"]
X_train = X_train.drop(["spk", "y"], axis=1)

X_test = df.loc[df["spk"].isin(test_speakers)]
y_test = X_test["y"]
X_test = X_test.drop(["spk", "y"], axis=1)

print("X train shape", X_train.shape)
print("X test shape", X_test.shape)

print("y train shape", y_train.shape)
print("y test shape", y_test.shape)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#no_pca_lda(cv)
#only_lda(cv)
#
#for n in [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]:
#    only_pca(cv, n)
#
#for n in [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]:
#    pca_lda(cv, n)

pipe = Pipeline([
         ("scale", StandardScaler()),
         ("pca", PCA(n_components=0.98, whiten=True)),
         ("lda", LinearDiscriminantAnalysis()),
         ("knn", KNeighborsClassifier())
    ])
search = GridSearchCV(
            pipe,
            KNN_PARAMS,
            scoring = 'accuracy',
            n_jobs=-1,
            cv=cv
        )
search.fit(X_train, y_train)
best_rf = search.best_estimator_
y_pred  = best_rf.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=words))
