import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import warnings

# Ignoruj ostrzeżenia
warnings.filterwarnings("ignore")

# Wczytanie i przygotowanie danych
print("Wczytywanie danych...")
n_rows = 30000  # Zmniejszona liczba wierszy do 30 000
df = pd.read_csv("train.csv", nrows=n_rows)
print(df.head(5))
Y = df['click'].values

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
print(X.shape)

# Podział na zbiory treningowy i testowy (90% / 10%)
n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# Kodowanie cech kategorycznych
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)
print(X_train_enc[0])

# 1. Strojenie hiperparametrów dla drzewa decyzyjnego
print("\n1. Strojenie hiperparametrów dla drzewa decyzyjnego...")
# Zmniejszona liczba kombinacji parametrów
cart_parameters = {
    'max_depth': [5, 10],
    'min_samples_split': [10, 30,],
    'class_weight': [None, 'balanced']
}

decision_tree = DecisionTreeClassifier(criterion='gini')
cart_grid = GridSearchCV(
    decision_tree,
    cart_parameters,
    n_jobs=1,  # Używamy tylko jednego rdzenia procesora
    cv=3,
    scoring='roc_auc'
)
cart_grid.fit(X_train_enc, Y_train)
print(f"Najlepsze parametry dla drzewa decyzyjnego: {cart_grid.best_params_}")

decision_tree_best = cart_grid.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]
best_cart_auc = roc_auc_score(Y_test, pos_prob)
print(f"Najwyższy wynik AUC dla drzewa decyzyjnego: {best_cart_auc:.3f}")

# 2. Strojenie hiperparametrów dla Random Forest
print("\n2. Strojenie hiperparametrów dla Random Forest...")
# Zmniejszona liczba kombinacji parametrów
rf_parameters = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [10, 30],
    'max_features': ['sqrt', 'log2']
}

random_forest = RandomForestClassifier(criterion='gini', n_jobs=1)  # Używamy tylko jednego rdzenia procesora
rf_grid = GridSearchCV(
    random_forest,
    rf_parameters,
    n_jobs=1,  # Używamy tylko jednego rdzenia procesora
    cv=3,
    scoring='roc_auc'
)
rf_grid.fit(X_train_enc, Y_train)
print(f"Najlepsze parametry dla Random Forest: {rf_grid.best_params_}")

random_forest_best = rf_grid.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
best_rf_auc = roc_auc_score(Y_test, pos_prob)
print(f"Najwyższy wynik AUC dla Random Forest: {best_rf_auc:.3f}")

# 3. Strojenie hiperparametrów dla XGBoost (GBT)
print("\n3. Strojenie hiperparametrów dla XGBoost (GBT)...")
# Zmniejszona liczba kombinacji parametrów
xgb_parameters = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(n_jobs=1)  # Używamy tylko jednego rdzenia procesora
xgb_grid = GridSearchCV(
    xgb_model,
    xgb_parameters,
    n_jobs=1,  # Używamy tylko jednego rdzenia procesora
    cv=3,
    scoring='roc_auc'
)
xgb_grid.fit(X_train_enc, Y_train)
print(f"Najlepsze parametry dla XGBoost: {xgb_grid.best_params_}")

xgb_best = xgb_grid.best_estimator_
pos_prob = xgb_best.predict_proba(X_test_enc)[:, 1]
best_xgb_auc = roc_auc_score(Y_test, pos_prob)
print(f"Najwyższy wynik AUC dla XGBoost: {best_xgb_auc:.3f}")

# Podsumowanie wyników
print("\nPodsumowanie wyników:")
print(f"1. Najwyższy wynik AUC dla drzewa decyzyjnego: {best_cart_auc:.3f}")
print(f"   Najlepsze parametry: {cart_grid.best_params_}")
print(f"2. Najwyższy wynik AUC dla Random Forest: {best_rf_auc:.3f}")
print(f"   Najlepsze parametry: {rf_grid.best_params_}")
print(f"3. Najwyższy wynik AUC dla XGBoost (GBT): {best_xgb_auc:.3f}")
print(f"   Najlepsze parametry: {xgb_grid.best_params_}")