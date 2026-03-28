from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

MLP_PARAMS = {
        "mlp__hidden_layer_sizes" : [(32,), (64,), (128,), (256, )],
        "mlp__activation": ['relu', 'tanh', 'logistic'],
        "mlp__solver": ['adam'],
        "mlp__max_iter": [200, 500, 1000],
        "mlp__alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001],
        #"learning_rate": ['constant', 'adaptive']
        }

warnings.simplefilter("ignore", category=ConvergenceWarning)

test_speakers = [4, 8, 14, 26, 28]
df = pd.read_csv('agregated_features_2.csv')

X_train = df.loc[~df["spk"].isin(test_speakers)]
y_train = X_train["y"]
X_train = X_train.drop(["spk", "y"], axis=1)

X_test = df.loc[df["spk"].isin(test_speakers)]
y_test = X_test["y"]
X_test = X_test.drop(["spk", "y"], axis=1)

print("Kolicina obelezja:", X_train.shape[1])

pipe = Pipeline([
        ("scaler", StandardScaler()),
        #("pca", PCA(n_components=0.95, random_state=42)),
        #("lda", LinearDiscriminantAnalysis(n_components=29)),
        ("mlp", MLPClassifier(random_state=42))
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
        pipe,
        MLP_PARAMS,
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
