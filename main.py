from modules import get_trend
from modules import prepare_data
from modules import split_data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import optuna


def objective(trial):

    n_estimators = trial.suggest_int("n_estimators", 2, 100)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 50)
    max_features = trial.suggest_categorical("max_features", ["log2", "sqrt"])
    bootstrap = trial.suggest_categorical("bootstrap", ["True", "False"])

    classifier_obj = RandomForestClassifier(max_depth=max_depth,
                                            criterion=criterion,
                                            n_estimators=n_estimators,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            bootstrap=bootstrap)

    score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=5, scoring='roc_auc')
    accuracy = score.mean()
    return accuracy


# Загружаем датасет из файла, форируем модифицированный датасет с установленным индексом по ключу Symbol
data = pd.read_csv("C:/Programs/Projects/IPO/kaggle/input/financial-ipo-data/IPODataFull.csv",
                   encoding="ISO-8859-1")
data_index = data.set_index('Symbol')

get_trend(data_index)
prepared_df = prepare_data(data_index)
X_train, X_test, y_train, y_test = split_data(prepared_df)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

best_rf_clf = RandomForestClassifier(**study.best_params)
best_rf_clf.fit(X_train, y_train)

rf_prob = best_rf_clf.predict_proba(X_test)[:, 1]
roc_value = roc_auc_score(y_test, rf_prob)
print(roc_value)
