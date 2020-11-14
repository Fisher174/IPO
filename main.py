import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import optuna

# Загружаем датасет из файла, форируем модифицированный датасет с установленным индексом по ключу Symbol
data = pd.read_csv("C:/Projects/IPO/kaggle/input/financial-ipo-data/IPODataFull.csv", encoding="ISO-8859-1")
data_index = data.set_index('Symbol')
price_start_position = 8
price_stop_position = 1318
price_step = 5
roll_window = 5
diff_step = 2
trend_const = 10
profitably_const = 0.3


# Функция, которая формирует список price с ценами закрытий акций за все время наблюдений (263 дня)
def get_price(tmp_data):
    return tmp_data[price_start_position:price_stop_position:price_step]


# Функция, которая формирует список day_of_max_price, содержащий номера дней, когда цена каждой акции была максимальна
def get_day_of_max_price():
    day_of_max_price = []
    for num in range(data_index.shape[0]):
        price = get_price(data_index.iloc[num])
        day_of_max_price.append(price.index(max(price)) + 1)
    return day_of_max_price


# Функция, которая:
# 1. Формирует новый df,состоящий только из цен закрытия за все 262 дня
# 2. Заменяет каждое значение цены средним за предыдущие 5 дней
# 3. Вычисляет разницу 2-х значений усредненных цен с шагом 2, затем сумируем все разницы
# 3. Заменяет каждое значение получившейся серии по условию:
# если сумма отрицательная - бычий тренд, -1, больше 10 - медвежий, 1.
# В ином случае - считаем, что тренд невыраженный, нулевой (то есть за год цена значительно не выросла и не упала).
# После этого функция добавляет к входному датасету новый столбец Trend со значениями тренда
# Также функция добавляет столбец financialResult в котором отражено на сколько процентов изменилась цена
# и столбец is_profit как показатель прибыльности акции
def get_trend(input_df):
    df = input_df.iloc[:, price_start_position:price_stop_position:price_step].copy()
    df = df.rolling(roll_window, axis=1).mean()
    df = df.diff(diff_step, axis=1).sum(axis=1)
    df[df < 0] = -1
    df[(df > 0) & (df <= trend_const)] = 0
    df[df > trend_const] = 1
    df = df.astype(int)
    input_df['trend'] = df
    input_df['financialResult'] = (input_df['closeDay261'] - input_df['closeDay0']) / input_df['closeDay0']
    input_df['is_profit'] = input_df['financialResult'] >= profitably_const
    input_df.is_profit[input_df['is_profit']] = 1
    return 0


# Функция предобработки данных. Вычищает спецсимволы, пробелы, приводит численные колонки к типу float,
# убирает пустые значения
def prepare_data(input_df):
    prepared_data = input_df[['MarketCap', 'Industry', 'Revenue', 'netIncome',
                              ' lastFiscalYearGrowth ', 'employees',
                              'YearFounded', 'is_profit']].copy()

    prepared_data = prepared_data.replace(r'\$', '', regex=True)
    prepared_data[['Revenue']] = prepared_data[['Revenue']].replace(' ', '', regex=True)
    prepared_data[['Revenue']] = prepared_data[['Revenue']].replace(',', '.', regex=True)
    prepared_data[['Revenue']] = prepared_data[['Revenue']].replace(r'\(', '', regex=True)
    prepared_data[['Revenue']] = prepared_data[['Revenue']].replace(r'\)', '', regex=True)
    prepared_data[['Revenue']] = prepared_data[['Revenue']].replace('B', '0000000', regex=True)
    prepared_data[['Revenue']] = prepared_data[['Revenue']].replace('M', '0000', regex=True)
    prepared_data[['Revenue']] = prepared_data[['Revenue']].replace(r'\.', '', regex=True).astype(float)
    prepared_data[['Revenue']] = np.log(prepared_data[['Revenue']])

    prepared_data[['netIncome']] = prepared_data[['netIncome']].replace(' ', '', regex=True)
    prepared_data[['netIncome']] = prepared_data[['netIncome']].replace(',', '.', regex=True)
    prepared_data[['netIncome']] = prepared_data[['netIncome']].replace('B', '0000000', regex=True)
    prepared_data[['netIncome']] = prepared_data[['netIncome']].replace('M', '0000', regex=True)
    prepared_data[['netIncome']] = prepared_data[['netIncome']].replace(r'\.', '', regex=True).astype(float)
    prepared_data[['netIncome']] = np.log(prepared_data[['netIncome']])

    prepared_data.rename(columns={" lastFiscalYearGrowth ": "lastFiscalYearGrowth"}, inplace=True)
    prepared_data[['lastFiscalYearGrowth']] = prepared_data[['lastFiscalYearGrowth']].replace(' ', '', regex=True)
    prepared_data[['lastFiscalYearGrowth']] = prepared_data[['lastFiscalYearGrowth']].replace('-', '0', regex=True)
    prepared_data[['lastFiscalYearGrowth']] = prepared_data.lastFiscalYearGrowth.astype(float)
    prepared_data[['lastFiscalYearGrowth']] = np.log(prepared_data[['lastFiscalYearGrowth']])

    prepared_data[['Industry']] = prepared_data[['Industry']].astype(str)

    prepared_data[['employees']] = prepared_data[['employees']].replace('-', '0', regex=True)
    prepared_data[['employees']] = prepared_data[['employees']].astype(float)

    prepared_data = prepared_data.replace([np.inf, -np.inf], np.nan)
    prepared_data.fillna(0, inplace=True)
    return prepared_data


def split_data(input_df):
    x = input_df.iloc[:, :7]
    y = input_df.is_profit
    label_encoder = LabelEncoder()
    x.iloc[:, 1] = label_encoder.fit_transform(x.iloc[:, 1])
    y.astype(int)
    x_train, x_test, y_tr, y_tes = train_test_split(x, y, random_state=0)
    return x_train, x_test, y_tr, y_tes


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


get_trend(data_index)
prepared_df = prepare_data(data_index)
X_train, X_test, y_train, y_test = split_data(prepared_df)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_rf_clf = RandomForestClassifier(**study.best_params)
best_rf_clf.fit(X_train, y_train)

rf_prob = best_rf_clf.predict_proba(X_test)[:, 1]
roc_value = roc_auc_score(y_test, rf_prob)
print(roc_value)
