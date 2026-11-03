import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Funkcje do generowania cech z poprzednich slajdów
def add_original_feature(df, df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)

def add_avg_price(df, df_new):
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']

def add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']

def add_std_price(df, df_new):
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']

def add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']

def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)

# Funkcja do dodawania cech z innych indeksów
def add_index_features(df, df_orig, index_df, index_name):
    # Synchronizuj indeksy według daty
    index_df = index_df.reindex(df.index, method='ffill')
    
    # Dodaj cechy z innych indeksów
    df[f'{index_name}_close_1'] = index_df['Close'].shift(1)
    
    # Dodaj zwroty z indeksu
    df[f'{index_name}_return_1'] = ((index_df['Close'] - index_df['Close'].shift(1)) / index_df['Close'].shift(1)).shift(1)
    df[f'{index_name}_return_5'] = ((index_df['Close'] - index_df['Close'].shift(5)) / index_df['Close'].shift(5)).shift(1)
    df[f'{index_name}_return_30'] = ((index_df['Close'] - index_df['Close'].shift(21)) / index_df['Close'].shift(21)).shift(1)
    
    # Dodaj średnie ruchome
    df[f'{index_name}_avg_price_5'] = index_df['Close'].rolling(5).mean().shift(1)
    df[f'{index_name}_avg_price_30'] = index_df['Close'].rolling(21).mean().shift(1)
    
    # Dodaj korelację pomiędzy głównym indeksem a indeksem zewnętrznym
    df[f'corr_{index_name}_5'] = df_orig['Close'].rolling(5).corr(index_df['Close']).shift(1)
    df[f'corr_{index_name}_30'] = df_orig['Close'].rolling(21).corr(index_df['Close']).shift(1)
    
    return df

# Funkcja do generowania podstawowych cech (bez indeksów)
def generate_basic_features(df):
    df_new = pd.DataFrame(index=df.index)
    
    # Dodaj oryginalne cechy
    add_original_feature(df, df_new)
    add_avg_price(df, df_new)
    add_avg_volume(df, df_new)
    add_std_price(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    
    # Cel
    df_new['close'] = df['Close']
    
    # Usuń wiersze z wartościami NaN
    df_new = df_new.dropna(axis=0)
    
    return df_new

# Funkcja do generowania rozszerzonych cech (z indeksami)
def generate_extended_features(df, russell_df=None, sp500_df=None):
    df_new = pd.DataFrame(index=df.index)
    
    # Dodaj oryginalne cechy
    add_original_feature(df, df_new)
    add_avg_price(df, df_new)
    add_avg_volume(df, df_new)
    add_std_price(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    
    # Dodaj cechy z indeksów Russell i S&P 500, jeśli dostępne
    if russell_df is not None:
        df_new = add_index_features(df_new, df, russell_df, 'russell')
    
    if sp500_df is not None:
        df_new = add_index_features(df_new, df, sp500_df, 'sp500')
    
    # Cel
    df_new['close'] = df['Close']
    
    # Usuń wiersze z wartościami NaN
    df_new = df_new.dropna(axis=0)
    
    return df_new

# Wczytanie danych
data_raw = pd.read_csv('19900101_20230630.csv', index_col='Date', parse_dates=True)
russell_raw = pd.read_csv('russell.csv', index_col='Date', parse_dates=True)
sp500_raw = pd.read_csv('sp500.csv', index_col='Date', parse_dates=True)

# Generowanie cech podstawowych i rozszerzonych
data_basic = generate_basic_features(data_raw)
data_extended = generate_extended_features(data_raw, russell_raw, sp500_raw)

# Definiowanie okresów treningowych i testowych
start_train = '1990-01-01'
end_train = '2022-12-31'
start_test = '2023-01-01'
end_test = '2023-06-30'

# Przygotowanie danych treningowych dla modelu podstawowego
data_train_basic = data_basic.loc[start_train:end_train]
X_train_basic = data_train_basic.drop('close', axis=1).values
y_train_basic = data_train_basic['close'].values

# Przygotowanie danych testowych dla modelu podstawowego
data_test_basic = data_basic.loc[start_test:end_test]
X_test_basic = data_test_basic.drop('close', axis=1).values
y_test_basic = data_test_basic['close'].values

# Przygotowanie danych treningowych dla modelu rozszerzonego
data_train_extended = data_extended.loc[start_train:end_train]
X_train_extended = data_train_extended.drop('close', axis=1).values
y_train_extended = data_train_extended['close'].values

# Przygotowanie danych testowych dla modelu rozszerzonego
data_test_extended = data_extended.loc[start_test:end_test]
X_test_extended = data_test_extended.drop('close', axis=1).values
y_test_extended = data_test_extended['close'].values

# Normalizacja danych dla modelu podstawowego
scaler_basic = StandardScaler()
X_scaled_train_basic = scaler_basic.fit_transform(X_train_basic)
X_scaled_test_basic = scaler_basic.transform(X_test_basic)

# Normalizacja danych dla modelu rozszerzonego
scaler_extended = StandardScaler()
X_scaled_train_extended = scaler_extended.fit_transform(X_train_extended)
X_scaled_test_extended = scaler_extended.transform(X_test_extended)

# Konfiguracja walidacji krzyżowej dla szeregów czasowych
tscv = TimeSeriesSplit(n_splits=3)

# Model 1: Regresja liniowa z SGD - podstawowy
param_grid_lr = {
    "alpha": [1e-4, 3e-4, 1e-3],
    "eta0": [0.01, 0.03, 0.1],
}
lr_basic = SGDRegressor(penalty='l2', max_iter=5000, random_state=42)
grid_search_lr_basic = GridSearchCV(lr_basic, param_grid_lr, cv=tscv, scoring='r2')
grid_search_lr_basic.fit(X_scaled_train_basic, y_train_basic)
lr_basic_best = grid_search_lr_basic.best_estimator_
predictions_lr_basic = lr_basic_best.predict(X_scaled_test_basic)

# Model 1: Regresja liniowa z SGD - rozszerzony
lr_extended = SGDRegressor(penalty='l2', max_iter=5000, random_state=42)
grid_search_lr_extended = GridSearchCV(lr_extended, param_grid_lr, cv=tscv, scoring='r2')
grid_search_lr_extended.fit(X_scaled_train_extended, y_train_extended)
lr_extended_best = grid_search_lr_extended.best_estimator_
predictions_lr_extended = lr_extended_best.predict(X_scaled_test_extended)

# Model 2: Drzewo decyzyjne - podstawowy
param_grid_dt = {
    'max_depth': [20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}
dt_basic = DecisionTreeRegressor(random_state=42)
grid_search_dt_basic = GridSearchCV(dt_basic, param_grid_dt, cv=tscv, scoring='r2', n_jobs=-1)
grid_search_dt_basic.fit(X_train_basic, y_train_basic)
dt_basic_best = grid_search_dt_basic.best_estimator_
predictions_dt_basic = dt_basic_best.predict(X_test_basic)

# Model 2: Drzewo decyzyjne - rozszerzony
dt_extended = DecisionTreeRegressor(random_state=42)
grid_search_dt_extended = GridSearchCV(dt_extended, param_grid_dt, cv=tscv, scoring='r2', n_jobs=-1)
grid_search_dt_extended.fit(X_train_extended, y_train_extended)
dt_extended_best = grid_search_dt_extended.best_estimator_
predictions_dt_extended = dt_extended_best.predict(X_test_extended)

# Model 3: Las losowy - podstawowy
param_grid_rf = {
    'max_depth': [20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}
rf_basic = RandomForestRegressor(n_estimators=30, n_jobs=-1, random_state=42)
grid_search_rf_basic = GridSearchCV(rf_basic, param_grid_rf, cv=tscv, scoring='r2', n_jobs=-1)
grid_search_rf_basic.fit(X_train_basic, y_train_basic)
rf_basic_best = grid_search_rf_basic.best_estimator_
predictions_rf_basic = rf_basic_best.predict(X_test_basic)

# Model 3: Las losowy - rozszerzony
rf_extended = RandomForestRegressor(n_estimators=30, n_jobs=-1, random_state=42)
grid_search_rf_extended = GridSearchCV(rf_extended, param_grid_rf, cv=tscv, scoring='r2', n_jobs=-1)
grid_search_rf_extended.fit(X_train_extended, y_train_extended)
rf_extended_best = grid_search_rf_extended.best_estimator_
predictions_rf_extended = rf_extended_best.predict(X_test_extended)

# Ocena modeli - obliczanie metryk
results = pd.DataFrame(columns=['Model', 'R^2', 'MSE', 'MAE'])

# Regresja liniowa - podstawowy
lr_basic_r2 = r2_score(y_test_basic, predictions_lr_basic)
lr_basic_mse = mean_squared_error(y_test_basic, predictions_lr_basic)
lr_basic_mae = mean_absolute_error(y_test_basic, predictions_lr_basic)
results.loc[0] = ['LR - Basic', lr_basic_r2, lr_basic_mse, lr_basic_mae]

# Regresja liniowa - rozszerzony
lr_extended_r2 = r2_score(y_test_extended, predictions_lr_extended)
lr_extended_mse = mean_squared_error(y_test_extended, predictions_lr_extended)
lr_extended_mae = mean_absolute_error(y_test_extended, predictions_lr_extended)
results.loc[1] = ['LR - Extended', lr_extended_r2, lr_extended_mse, lr_extended_mae]

# Drzewo decyzyjne - podstawowy
dt_basic_r2 = r2_score(y_test_basic, predictions_dt_basic)
dt_basic_mse = mean_squared_error(y_test_basic, predictions_dt_basic)
dt_basic_mae = mean_absolute_error(y_test_basic, predictions_dt_basic)
results.loc[2] = ['DT - Basic', dt_basic_r2, dt_basic_mse, dt_basic_mae]

# Drzewo decyzyjne - rozszerzony
dt_extended_r2 = r2_score(y_test_extended, predictions_dt_extended)
dt_extended_mse = mean_squared_error(y_test_extended, predictions_dt_extended)
dt_extended_mae = mean_absolute_error(y_test_extended, predictions_dt_extended)
results.loc[3] = ['DT - Extended', dt_extended_r2, dt_extended_mse, dt_extended_mae]

# Las losowy - podstawowy
rf_basic_r2 = r2_score(y_test_basic, predictions_rf_basic)
rf_basic_mse = mean_squared_error(y_test_basic, predictions_rf_basic)
rf_basic_mae = mean_absolute_error(y_test_basic, predictions_rf_basic)
results.loc[4] = ['RF - Basic', rf_basic_r2, rf_basic_mse, rf_basic_mae]

# Las losowy - rozszerzony
rf_extended_r2 = r2_score(y_test_extended, predictions_rf_extended)
rf_extended_mse = mean_squared_error(y_test_extended, predictions_rf_extended)
rf_extended_mae = mean_absolute_error(y_test_extended, predictions_rf_extended)
results.loc[5] = ['RF - Extended', rf_extended_r2, rf_extended_mse, rf_extended_mae]

# Wyświetlanie wyników
print("Porównanie wydajności modeli:")
print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Obliczanie różnic w metrykach między modelami rozszerzonymi a podstawowymi
comparison = pd.DataFrame(columns=['Model', 'R^2 Change', 'MSE Change', 'MAE Change'])

# Regresja liniowa
lr_r2_change = lr_extended_r2 - lr_basic_r2
lr_mse_change = lr_extended_mse - lr_basic_mse
lr_mae_change = lr_extended_mae - lr_basic_mae
comparison.loc[0] = ['Linear Regression', lr_r2_change, lr_mse_change, lr_mae_change]

# Drzewo decyzyjne
dt_r2_change = dt_extended_r2 - dt_basic_r2
dt_mse_change = dt_extended_mse - dt_basic_mse
dt_mae_change = dt_extended_mae - dt_basic_mae
comparison.loc[1] = ['Decision Tree', dt_r2_change, dt_mse_change, dt_mae_change]

# Las losowy
rf_r2_change = rf_extended_r2 - rf_basic_r2
rf_mse_change = rf_extended_mse - rf_basic_mse
rf_mae_change = rf_extended_mae - rf_basic_mae
comparison.loc[2] = ['Random Forest', rf_r2_change, rf_mse_change, rf_mae_change]

# Analiza poprawy/pogorszenia
print("\nAnaliza wpływu dodania indeksów zewnętrznych:")
for i, model in enumerate(['Linear Regression', 'Decision Tree', 'Random Forest']):
    r2_change = comparison.loc[i, 'R^2 Change']
    mse_change = comparison.loc[i, 'MSE Change']
    mae_change = comparison.loc[i, 'MAE Change']
    
    print(f"\n{model}:")
    # R^2 - wyższy jest lepszy
    if r2_change > 0:
        print(f"  R^2: Poprawa o {r2_change:.3f}")
    else:
        print(f"  R^2: Pogorszenie o {abs(r2_change):.3f}")
    
    # MSE - niższy jest lepszy
    if mse_change < 0:
        print(f"  MSE: Poprawa o {abs(mse_change):.3f}")
    else:
        print(f"  MSE: Pogorszenie o {mse_change:.3f}")
    
    # MAE - niższy jest lepszy
    if mae_change < 0:
        print(f"  MAE: Poprawa o {abs(mae_change):.3f}")
    else:
        print(f"  MAE: Pogorszenie o {mae_change:.3f}")
    
    # Ogólna ocena
    if r2_change > 0 and mse_change < 0 and mae_change < 0:
        print("  Wniosek: Dodanie indeksów zewnętrznych znacząco poprawiło wydajność modelu.")
    elif r2_change > 0 or mse_change < 0 or mae_change < 0:
        print("  Wniosek: Dodanie indeksów zewnętrznych częściowo poprawiło wydajność modelu.")
    else:
        print("  Wniosek: Dodanie indeksów zewnętrznych pogorszyło wydajność modelu.")

# Wizualizacja wyników - dwa oddzielne wykresy
plt.figure(figsize=(12, 10))

# Wykres 1: Modele rozszerzone (Extended)
plt.subplot(2, 1, 1)
plt.plot(data_test_extended.index, y_test_extended, 'k-', label='Rzeczywista wartość', linewidth=2)
plt.plot(data_test_extended.index, predictions_lr_extended, 'b-', label='Regresja liniowa (Extended)', alpha=0.7)
plt.plot(data_test_extended.index, predictions_dt_extended, 'g-', label='Drzewo decyzyjne (Extended)', alpha=0.7)
plt.plot(data_test_extended.index, predictions_rf_extended, 'r-', label='Las losowy (Extended)', alpha=0.7)
plt.xticks(rotation=45)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Cena zamknięcia', fontsize=12)
plt.legend(fontsize=10)
plt.title('Modele z dodatkowymi indeksami (Extended)', fontsize=14)
plt.grid(True, alpha=0.3)

# Wykres 2: Modele podstawowe (Basic)
plt.subplot(2, 1, 2)
plt.plot(data_test_basic.index, y_test_basic, 'k-', label='Rzeczywista wartość', linewidth=2)
plt.plot(data_test_basic.index, predictions_lr_basic, 'b-', label='Regresja liniowa (Basic)', alpha=0.7)
plt.plot(data_test_basic.index, predictions_dt_basic, 'g-', label='Drzewo decyzyjne (Basic)', alpha=0.7)
plt.plot(data_test_basic.index, predictions_rf_basic, 'r-', label='Las losowy (Basic)', alpha=0.7)
plt.xticks(rotation=45)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Cena zamknięcia', fontsize=12)
plt.legend(fontsize=10)
plt.title('Modele podstawowe (Basic)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Zadanie 2: Implementacja modeli ensemble z różnymi strategiami ważenia

# 1. Model ensemble z równymi wagami
def equal_weights_ensemble(predictions_list):
    """Tworzy predykcje na podstawie średniej arytmetycznej z predykcji poszczególnych modeli"""
    return np.mean(predictions_list, axis=0)

# 2. Model ensemble z wagami opartymi na R^2
def r2_weighted_ensemble(predictions_list, y_true):
    """Tworzy predykcje ważone wartościami R^2 poszczególnych modeli"""
    r2_scores = []
    for predictions in predictions_list:
        r2 = r2_score(y_true, predictions)
        r2_scores.append(max(r2, 0))  # Używamy tylko nieujemnych wartości R^2
    
    # Jeśli wszystkie R^2 są ujemne, użyj równych wag
    if sum(r2_scores) == 0:
        weights = np.ones(len(predictions_list)) / len(predictions_list)
    else:
        weights = np.array(r2_scores) / sum(r2_scores)
    
    # Ważona suma predykcji
    weighted_predictions = np.zeros_like(predictions_list[0])
    for i, predictions in enumerate(predictions_list):
        weighted_predictions += weights[i] * predictions
        
    return weighted_predictions

# 3. Model ensemble z wagami opartymi na MSE (im mniejszy MSE, tym większa waga)
def mse_weighted_ensemble(predictions_list, y_true):
    """Tworzy predykcje ważone odwrotnościami MSE poszczególnych modeli"""
    mse_scores = []
    for predictions in predictions_list:
        mse = mean_squared_error(y_true, predictions)
        mse_scores.append(mse)
    
    # Obliczamy odwrotności MSE (im mniejszy MSE, tym większa waga)
    inverse_mse = 1.0 / np.array(mse_scores)
    weights = inverse_mse / sum(inverse_mse)
    
    # Ważona suma predykcji
    weighted_predictions = np.zeros_like(predictions_list[0])
    for i, predictions in enumerate(predictions_list):
        weighted_predictions += weights[i] * predictions
        
    return weighted_predictions

# 4. Model ensemble z ręcznie dobranymi wagami
def manual_weights_ensemble(predictions_list, manual_weights=None):
    """Tworzy predykcje na podstawie ręcznie dobranych wag"""
    if manual_weights is None:
        # Domyślne wagi: większa waga dla lasu losowego, mniejsze dla pozostałych
        manual_weights = [0.2, 0.3, 0.5]  # LR, DT, RF
    
    # Normalizacja wag (aby sumowały się do 1)
    weights = np.array(manual_weights) / sum(manual_weights)
    
    # Ważona suma predykcji
    weighted_predictions = np.zeros_like(predictions_list[0])
    for i, predictions in enumerate(predictions_list):
        weighted_predictions += weights[i] * predictions
        
    return weighted_predictions

# Przygotowanie listy predykcji z modeli podstawowych
predictions_basic_list = [predictions_lr_basic, predictions_dt_basic, predictions_rf_basic]

# Tworzenie predykcji ensemble dla modeli podstawowych
ensemble_equal_basic = equal_weights_ensemble(predictions_basic_list)
ensemble_r2_basic = r2_weighted_ensemble(predictions_basic_list, y_test_basic)
ensemble_mse_basic = mse_weighted_ensemble(predictions_basic_list, y_test_basic)
ensemble_manual_basic = manual_weights_ensemble(predictions_basic_list)

# Obliczanie metryk dla modeli ensemble
# Ensemble z równymi wagami
equal_basic_r2 = r2_score(y_test_basic, ensemble_equal_basic)
equal_basic_mse = mean_squared_error(y_test_basic, ensemble_equal_basic)
equal_basic_mae = mean_absolute_error(y_test_basic, ensemble_equal_basic)

# Ensemble z wagami R^2
r2_basic_r2 = r2_score(y_test_basic, ensemble_r2_basic)
r2_basic_mse = mean_squared_error(y_test_basic, ensemble_r2_basic)
r2_basic_mae = mean_absolute_error(y_test_basic, ensemble_r2_basic)

# Ensemble z wagami MSE
mse_basic_r2 = r2_score(y_test_basic, ensemble_mse_basic)
mse_basic_mse = mean_squared_error(y_test_basic, ensemble_mse_basic)
mse_basic_mae = mean_absolute_error(y_test_basic, ensemble_mse_basic)

# Ensemble z ręcznymi wagami
manual_basic_r2 = r2_score(y_test_basic, ensemble_manual_basic)
manual_basic_mse = mean_squared_error(y_test_basic, ensemble_manual_basic)
manual_basic_mae = mean_absolute_error(y_test_basic, ensemble_manual_basic)

# Tworzenie tabeli wyników dla modeli ensemble
ensemble_results = pd.DataFrame({
    'Model': ['Ensemble (równe wagi)', 'Ensemble (wagi R^2)', 'Ensemble (wagi MSE)', 'Ensemble (ręczne wagi)'],
    'R^2': [equal_basic_r2, r2_basic_r2, mse_basic_r2, manual_basic_r2],
    'MSE': [equal_basic_mse, r2_basic_mse, mse_basic_mse, manual_basic_mse],
    'MAE': [equal_basic_mae, r2_basic_mae, mse_basic_mae, manual_basic_mae]
})

# Wyrównanie do prawej strony
pd.set_option('display.colheader_justify', 'right')
pd.set_option('display.width', None)

# Wyświetlanie wyników
print("\nWyniki modeli ensemble:")
print(ensemble_results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Wizualizacja wyników - porównanie tylko modeli ensemble
plt.figure(figsize=(12, 6))
plt.plot(data_test_basic.index, y_test_basic, 'k-', label='Rzeczywista wartość', linewidth=2)
plt.plot(data_test_basic.index, ensemble_equal_basic, 'b-', label='Ensemble (równe wagi)', alpha=0.7)
plt.plot(data_test_basic.index, ensemble_r2_basic, 'g-', label='Ensemble (wagi R^2)', alpha=0.7)
plt.plot(data_test_basic.index, ensemble_mse_basic, 'r-', label='Ensemble (wagi MSE)', alpha=0.7)
plt.plot(data_test_basic.index, ensemble_manual_basic, 'm-', label='Ensemble (ręczne wagi)', alpha=0.7)
plt.xticks(rotation=45)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Cena zamknięcia', fontsize=12)
plt.legend(fontsize=10)
plt.title('Porównanie modeli ensemble', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()