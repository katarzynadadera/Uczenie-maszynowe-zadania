import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import time

# Wczytanie danych
# Zakładamy, że plik train.csv znajduje się w bieżącym katalogu
n_rows = 1500000  # Używamy 1 500 000 wierszy
df = pd.read_csv("train.csv", nrows=n_rows)

# Przygotowanie cech i zmiennej docelowej
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

# Podział na zbiory treningowe i testowe (90% trening, 10% test, jak w prezentacji)
n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# Kodowanie cech kategorycznych metodą one-hot
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

# Definicja siatki hiperparametrów do dostrojenia
penalties = ['l1', 'l2', 'elasticnet']
wspolczynniki_uczenia = [0.001, 0.01, 0.1]
wspolczynniki_regularyzacji = [0.0001, 0.001, 0.01]

# Słownik do przechowywania wyników
wyniki = {}

# Przeszukiwanie siatki hiperparametrów
najlepszy_auc = 0
najlepsze_parametry = {}

for penalty in penalties:
    for eta0 in wspolczynniki_uczenia:
        for alpha in wspolczynniki_regularyzacji:
            # Tworzenie i trenowanie modelu
            print(f"Trenowanie z parametrami: penalty={penalty}, eta0={eta0}, alpha={alpha}")
            czas_poczatek = time.time()
            
            sgd_lr = SGDClassifier(
                loss='log_loss',
                penalty=penalty,
                alpha=alpha,
                fit_intercept=True,
                max_iter=20,
                learning_rate='constant',
                eta0=eta0,
                random_state=42
            )
            
            sgd_lr.fit(X_train_enc, Y_train)
            
            # Generowanie prognoz
            prog = sgd_lr.predict_proba(X_test_enc)[:, 1]
            
            # Obliczanie AUC
            auc = roc_auc_score(Y_test, prog)
            
            # Zapisywanie wyniku
            klucz = f"penalty={penalty}, eta0={eta0}, alpha={alpha}"
            wyniki[klucz] = auc
            
            # Aktualizacja najlepszych parametrów, jeśli bieżący jest lepszy
            if auc > najlepszy_auc:
                najlepszy_auc = auc
                najlepsze_parametry = {'penalty': penalty, 'eta0': eta0, 'alpha': alpha}
            
            print(f"AUC: {auc:.4f}, Czas: {time.time() - czas_poczatek:.2f}s")


# Wyświetlenie najlepszych parametrów
print(f"\nNajlepsze parametry: {najlepsze_parametry}")
print(f"Najlepszy AUC: {najlepszy_auc:.4f}")

# Analiza statystyk wag dla różnych metod regularyzacji
print("\nAnaliza statystyk wag dla różnych metod regularyzacji:")

# Trenowanie modeli z najlepszymi wartościami eta0 i alpha dla każdego typu penalty
najlepsze_eta0 = najlepsze_parametry['eta0']
najlepsze_alpha = najlepsze_parametry['alpha']

statystyki_wag = {}

for penalty in penalties:
    sgd_lr = SGDClassifier(
        loss='log_loss',
        penalty=penalty,
        alpha=najlepsze_alpha,
        fit_intercept=True,
        max_iter=20,
        learning_rate='constant',
        eta0=najlepsze_eta0,
        random_state=42
    )
    
    sgd_lr.fit(X_train_enc, Y_train)
    
    # Pobranie wag modelu
    wagi = sgd_lr.coef_[0]
    
    # Obliczenie statystyk wag
    statystyki_wag[penalty] = {
        'niezerowe_wagi': np.count_nonzero(wagi),
        'wszystkie_wagi': len(wagi),
        'srednia': np.mean(wagi),
        'odch_std': np.std(wagi),
        'minimum': np.min(wagi),
        'maksimum': np.max(wagi)
    }

# Wyświetlenie statystyk wag
for penalty, stats in statystyki_wag.items():
    print(f"\nTyp penalty: {penalty}")
    print(f"Niezerowe wagi: {stats['niezerowe_wagi']} z {stats['wszystkie_wagi']} ({stats['niezerowe_wagi']/stats['wszystkie_wagi']*100:.2f}%)")
    print(f"Średnia: {stats['srednia']:.6f}")
    print(f"Odchylenie standardowe: {stats['odch_std']:.6f}")
    print(f"Minimum: {stats['minimum']:.6f}")
    print(f"Maksimum: {stats['maksimum']:.6f}")

# Wnioski
print("\nWnioski:")
print("1. Regularyzacja L1 prowadzi do tworzenia rzadkich modeli (wiele wag jest dokładnie równych 0)")
print("2. Regularyzacja L2 zmniejsza wszystkie wagi w kierunku zera, ale rzadko czyni je dokładnie równymi 0")
print("3. Elasticnet łączy właściwości L1 i L2")
print(f"4. Najlepiej działający model używał regularyzacji {najlepsze_parametry['penalty']} z eta0={najlepsze_parametry['eta0']} i alpha={najlepsze_parametry['alpha']}")
print(f"5. Najwyższy osiągnięty AUC wynosił {najlepszy_auc:.4f}")