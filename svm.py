from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

SVM_PARAMS = {
        "svm__C": [0.01, 0.1, 1.0, 10, 100],
        "svm__kernel": ['linear', 'rbf'],
        "svm__gamma": ['scale', 'auto', 0.1],
        }

test_speakers = [4, 8, 14, 26, 28]

pca_df = pd.read_csv('pca_features_3.csv')
#of_df = pd.read_csv('optical_flow_2.csv')
#of_df = of_df.drop(["y","spk"], axis=1)
#
#df = pd.concat([pca_df, of_df], axis=1)

#mag_df = pd.read_csv('mag_features.csv')
#ang_df = pd.read_csv('ang_features.csv')
#
#mag_df = mag_df.drop(["y","spk"], axis=1)
#ang_df = ang_df.drop(["y","spk"], axis=1)

#pca_of_df = pd.read_csv('pca_of_1.csv')
#pca_of_df = pca_of_df.drop(["y","spk"], axis=1)

#pca_of_df = pd.concat([mag_df, ang_df], axis=1)
#
#df = pd.concat([pca_df, pca_of_df], axis=1)
#
#of_cols = [c for c in pca_of_df.columns if c not in ["y", "spk"]]
#
#mag_cols = [c for c in mag_df.columns if c not in ["y", "spk"]]
#ang_cols = [c for c in ang_df.columns if c not in ["y", "spk"]]
#
pca_cols = [c for c in pca_df.columns if c not in ["y", "spk"]]
#
#df = pca_df
#

dwt_df = pd.read_csv('dwt_features.csv')
dwt_df = dwt_df.drop(["y", "spk"], axis=1)

df = pd.concat([dwt_df, pca_df], axis=1)

dwt_cols = [c for c in dwt_df.columns if c not in ["y", "spk"]]

feat_cols = pca_cols + dwt_cols

'''
transformer = ColumnTransformer([
    ("pca", Pipeline([
        #("scale", StandardScaler()),
        ("pcaa", PCA(n_components=75)),
        #("ldaa", LinearDiscriminantAnalysis(n_components=29))
        ]), pca_cols),
    ("mag", Pipeline([
        #("scale2", StandardScaler()),
        ("pcaa2", PCA(n_components=75)),
        #("ldaa2", LinearDiscriminantAnalysis(n_components=29))
        ]), dwt_cols)
    ])
#'''

X_train = df.loc[~df["spk"].isin(test_speakers), feat_cols]
y_train = df.loc[~df["spk"].isin(test_speakers), "y"].values

X_test = df.loc[df["spk"].isin(test_speakers), feat_cols]
y_test = df.loc[df["spk"].isin(test_speakers), "y"].values

#pca - 0.98 lda 29
#of - 0.9 lda 29
#pca, of - 174 lda 29
#with 30 comps
pipe = Pipeline([
    #("transformer", transformer),
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.99, whiten=True)),
    ("lda", LinearDiscriminantAnalysis(n_components=29)),
    ("svm", SVC())
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
        pipe,
        SVM_PARAMS,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )
search.fit(X_train, y_train)

print(f"Best params:   {search.best_params_}")
print(f"Best CV score: {search.best_score_*100:.1f}%")
print(f"Test score:    {search.score(X_test, y_test)*100:.1f}%")
