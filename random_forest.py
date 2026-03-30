import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

words = ["nula", "jedan", "dva", "tri", "četiri", "pet",
         "šest", "sedam", "osam", "devet",
         "napred", "nazad", "levo", "desno", "gore", "dole",
         "potvrdi", "odustani", "obriši", "pošalji", "dalje", "početak", "kraj",
         "ponedeljak", "utorak", "sreda", "četvrtak", "petak", "subota", "nedelja"]

RF_PARAM_DIST = {
    "rf__n_estimators":      [100, 200, 300, 500],
    "rf__max_depth":         [3, 5, 7, 10, 15, 20, None],
    "rf__min_samples_split": [2, 5, 10, 20],
    "rf__min_samples_leaf":  [1, 2, 4, 8],
    "rf__max_features":      ["sqrt", "log2", 0.3, 0.5, 0.8],
    "rf__class_weight":      [None, "balanced"],
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
    
    search = RandomizedSearchCV(
        pipe,
        RF_PARAM_DIST,
        n_iter=30,
        cv=cv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=1
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
    
    search = RandomizedSearchCV(
        pipe,
        RF_PARAM_DIST,
        n_iter=30,
        cv=cv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=1
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
    
    search = RandomizedSearchCV(
        pipe,
        RF_PARAM_DIST,
        n_iter=30,
        cv=cv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=1
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

#df_agregated = pd.read_csv('geometric_agregated.csv')
df_interp = pd.read_csv('geometric_interp.csv')
#df_timed = pd.read_csv('geometric_timed.csv')
#df_timed = df_timed.drop(["y", "spk"], axis=1)

df = df_interp

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

no_pca_lda(cv)
only_lda(cv)

for n in [0.8, 0.85, 0.9, 0.95, 0.99]:
    only_pca(cv, n)

#print_feature_importance(best_rf.named_steps['rf'], X_train)
