import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, precision_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

words = ["nula", "jedan", "dva", "tri", "četiri", "pet",
         "šest", "sedam", "osam", "devet",
         "napred", "nazad", "levo", "desno", "gore", "dole",
         "potvrdi", "odustani", "obriši", "pošalji", "dalje", "početak", "kraj",
         "ponedeljak", "utorak", "sreda", "četvrtak", "petak", "subota", "nedelja"]

RF_PARAM_DIST = {
    "rf__n_estimators":      [300, 500],
    "rf__max_depth":         [5, 7, None],
    "rf__min_samples_split": [5, 7, 10],
    "rf__min_samples_leaf":  [4, 8],
    "rf__max_features":      ["sqrt", "log2"],
}

def print_feature_importance(rf_model, X):
    importances = rf_model.feature_importances_
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

def no_pca_lda(cv):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42))
        ])
    
    search = GridSearchCV(
        pipe,
        RF_PARAM_DIST,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    
    print("Samo StandardScaler")
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
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
        ("rf", RandomForestClassifier(random_state=42))
        ])
    
    search = GridSearchCV(
        pipe,
        RF_PARAM_DIST,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    
    print("Samo LDA")
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
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("rf", RandomForestClassifier(random_state=42))
        ])
    
    search = GridSearchCV(
        pipe,
        RF_PARAM_DIST,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
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
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("lda", LinearDiscriminantAnalysis()),
        ("rf", RandomForestClassifier(random_state=42))
        ])
    
    search = GridSearchCV(
        pipe,
        RF_PARAM_DIST,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
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

#print_feature_importance(best_rf.named_steps['rf'], X_train)
pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.98)),
        ("lda", LinearDiscriminantAnalysis()),
        ("rf", RandomForestClassifier(random_state=42))
        ])
    
search = GridSearchCV(
        pipe,
        RF_PARAM_DIST,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
search.fit(X_train, y_train)

best_rf = search.best_estimator_
y_pred  = best_rf.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=words))
