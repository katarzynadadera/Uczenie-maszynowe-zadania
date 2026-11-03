import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import time
import warnings

# Ignoruj ostrzeżenia
warnings.filterwarnings("ignore")

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Indeks cechy do podziału
        self.threshold = threshold      # Wartość progowa dla podziału
        self.left = left                # Lewe poddrzewo
        self.right = right              # Prawe poddrzewo
        self.value = value              # Przewidywana wartość dla liścia

class CART:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Kryteria zatrzymania rekurencji
        if (self.max_depth is not None and depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Znajdź najlepszy podział
        best_feature_idx, best_threshold = self._best_split(X, y)
        
        # Jeśli nie znaleziono podziału poprawiającego kryterium, utwórz liść
        if best_feature_idx is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Podziel dane
        left_idxs = X[:, best_feature_idx] < best_threshold
        right_idxs = ~left_idxs
        
        # Rekurencyjnie buduj poddrzewa
        left_subtree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature_idx=best_feature_idx, threshold=best_threshold,
                   left=left_subtree, right=right_subtree)
    
    def _gini_impurity(self, y):
        """Oblicz czystość Giniego dla danego zbioru etykiet."""
        m = len(y)
        if m == 0:
            return 0
        
        counts = np.bincount(y)
        fractions = counts / float(m)
        return 1 - np.sum(fractions ** 2)
    
    def _entropy(self, y):
        """Oblicz entropię dla danego zbioru etykiet."""
        m = len(y)
        if m == 0:
            return 0
        
        counts = np.bincount(y)
        fractions = counts / float(m)
        return -np.sum(fractions * np.log2(fractions + 1e-10))  # Dodajemy małą wartość, aby uniknąć log(0)
    
    def _best_split(self, X, y):
        """Znajdź najlepszy podział dla danych."""
        m, n = X.shape
        if m <= 1:
            return None, None
        
        # Wybierz funkcję kryterium
        if self.criterion == 'gini':
            criterion_func = self._gini_impurity
        else:  # 'entropy'
            criterion_func = self._entropy
        
        # Wartość kryterium dla bieżącego węzła
        current_impurity = criterion_func(y)
        best_gain = 0.0  # Inicjalizacja najlepszego zysku informacyjnego
        best_feature_idx, best_threshold = None, None
        
        # Przeszukaj wszystkie cechy
        for feature_idx in range(n):
            # Pobierz unikalne wartości cechy - dla wydajności ograniczamy liczbę wartości progowych
            feature_values = X[:, feature_idx]
            if len(np.unique(feature_values)) > 10:
                percentiles = np.percentile(feature_values, [25, 50, 75])
                thresholds = percentiles
            else:
                thresholds = np.unique(feature_values)
            
            # Przeszukaj wszystkie możliwe wartości progowe
            for threshold in thresholds:
                # Podziel dane
                left_idxs = X[:, feature_idx] < threshold
                right_idxs = ~left_idxs
                
                # Pomiń, jeśli jeden z podziałów jest pusty
                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue
                
                # Oblicz ważoną wartość kryterium po podziale
                left_impurity = criterion_func(y[left_idxs])
                right_impurity = criterion_func(y[right_idxs])
                
                # Wagi to proporcje próbek w każdym podziale
                n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
                weighted_impurity = (n_left / m) * left_impurity + (n_right / m) * right_impurity
                
                # Oblicz zysk informacyjny
                information_gain = current_impurity - weighted_impurity
                
                # Aktualizuj, jeśli to najlepszy podział dotychczas
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold
    
    def _most_common_label(self, y):
        """Zwróć najczęściej występującą etykietę."""
        return np.bincount(y).argmax()
    
    def predict(self, X):
        """Przewiduj klasy dla próbek w X."""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Przeglądaj drzewo, aby znaleźć przewidywanie dla próbki x."""
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict_proba(self, X):
        """Przewiduj prawdopodobieństwa klas dla X."""
        y_pred = self.predict(X)
        # Konwertuj na one-hot encoding dla klasyfikacji binarnej
        proba = np.zeros((len(y_pred), 2))
        for i, val in enumerate(y_pred):
            proba[i, val] = 1.0
        return proba


def plot_roc_curves(y_true, y_score_custom, y_score_sklearn):
    """Narysuj krzywe ROC dla obu implementacji."""
    plt.figure(figsize=(10, 8))
    
    # Krzywa ROC dla własnej implementacji
    fpr_custom, tpr_custom, _ = roc_curve(y_true, y_score_custom)
    roc_auc_custom = auc(fpr_custom, tpr_custom)
    plt.plot(fpr_custom, tpr_custom, color='blue', lw=2,
             label=f'Własna implementacja ROC (AUC = {roc_auc_custom:.3f})')
    
    # Krzywa ROC dla scikit-learn
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_true, y_score_sklearn)
    roc_auc_sklearn = auc(fpr_sklearn, tpr_sklearn)
    plt.plot(fpr_sklearn, tpr_sklearn, color='red', lw=2,
             label=f'Scikit-learn ROC (AUC = {roc_auc_sklearn:.3f})')
    
    # Linia odniesienia (losowy klasyfikator)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Losowy klasyfikator')
    
    # Formatowanie wykresu
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Krzywe ROC - porównanie implementacji CART')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig('roc_curves.png')
    plt.close()
    print("Wykres krzywych ROC został zapisany do pliku 'roc_curves.png'")


def main():
    # Wczytaj dane
    print("Wczytywanie danych...")
    n_rows = 200000  # Zwiększona liczba wierszy
    df = pd.read_csv("train.csv", nrows=n_rows)
    print(df.head(5))
    Y = df['click'].values
    
    # Przygotuj cechy
    X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
    print(f"Wymiar danych: {X.shape}")
    
    # Podział na zbiory treningowy i testowy
    n_train = int(n_rows * 0.8)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    
    # Kodowanie one-hot
    print("Kodowanie cech...")
    enc = OneHotEncoder(handle_unknown='ignore')
    X_train_enc = enc.fit_transform(X_train)
    X_test_enc = enc.transform(X_test)
    
    # Używamy większej próbki dla treningu
    sample_size = 50000  # Zwiększona próbka dla treningu
    sample_indices = np.random.choice(X_train_enc.shape[0], sample_size, replace=False)
    X_train_sample = X_train_enc[sample_indices].toarray()
    Y_train_sample = Y_train[sample_indices]
    
    test_sample_size = 10000  # Zwiększona próbka dla testu
    test_indices = np.random.choice(X_test_enc.shape[0], test_sample_size, replace=False)
    X_test_sample = X_test_enc[test_indices].toarray()
    Y_test_sample = Y_test[test_indices]
    
    # Otwórz plik do zapisu wyników
    with open("wyniki.txt", "w") as results_file:
        # Funkcja do zapisu zarówno do konsoli jak i do pliku
        def log(message):
            print(message)
            results_file.write(message + "\n")
        
        # Trenowanie własnego CART
        log("\nTrenowanie wlasnej implementacji CART...")
        start_time = time.time()
        max_depth = 5  # Zwiększona głębokość
        custom_cart = CART(max_depth=max_depth, min_samples_split=10, criterion='gini')
        custom_cart.fit(X_train_sample, Y_train_sample)
        custom_train_time = time.time() - start_time
        log(f"Czas trenowania wlasnej implementacji: {custom_train_time:.2f} sekund")
        
        # Trenowanie CART z scikit-learn
        log("\nTrenowanie CART z scikit-learn...")
        start_time = time.time()
        sklearn_cart = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_split=10)
        sklearn_cart.fit(X_train_sample, Y_train_sample)
        sklearn_train_time = time.time() - start_time
        log(f"Czas trenowania scikit-learn: {sklearn_train_time:.2f} sekund")
        
        # Ewaluacja - czas predykcji
        log("\nEwaluacja czasu predykcji...")
        start_time = time.time()
        custom_preds = custom_cart.predict(X_test_sample)
        custom_predict_time = time.time() - start_time
        log(f"Czas predykcji wlasnej implementacji: {custom_predict_time:.2f} sekund")
        
        start_time = time.time()
        sklearn_preds = sklearn_cart.predict(X_test_sample)
        sklearn_predict_time = time.time() - start_time
        log(f"Czas predykcji scikit-learn: {sklearn_predict_time:.2f} sekund")
        
        # Dokładność klasyfikacji
        custom_accuracy = accuracy_score(Y_test_sample, custom_preds)
        sklearn_accuracy = accuracy_score(Y_test_sample, sklearn_preds)
        
        log(f"\nDokładność wlasnej implementacji CART: {custom_accuracy:.3f}")
        log(f"Dokładność implementacji CART z scikit-learn: {sklearn_accuracy:.3f}")
        
        # Obliczanie AUC (Area Under the ROC Curve)
        custom_proba = custom_cart.predict_proba(X_test_sample)[:, 1]
        sklearn_proba = sklearn_cart.predict_proba(X_test_sample)[:, 1]
        
        custom_auc = roc_auc_score(Y_test_sample, custom_proba)
        sklearn_auc = roc_auc_score(Y_test_sample, sklearn_proba)
        
        log(f"\nAUC wlasnej implementacji CART: {custom_auc:.3f}")
        log(f"AUC implementacji CART z scikit-learn: {sklearn_auc:.3f}")
        
        # Analiza różnic w predykcjach
        diff_count = np.sum(custom_preds != sklearn_preds)
        log(f"\nLiczba roznych predykcji: {diff_count} z {len(Y_test_sample)} ({diff_count/len(Y_test_sample)*100:.1f}%)")
        
        # Generowanie wykresu krzywych ROC
        try:
            plot_roc_curves(Y_test_sample, custom_proba, sklearn_proba)
            log("Wykres krzywych ROC zostal zapisany do pliku 'roc_curves.png'")
        except Exception as e:
            log(f"Nie udalo sie wygenerowac wykresu krzywych ROC: {e}")
        
        # Podsumowanie
        log("\nPodsumowanie porownania:")
        log("1. Wlasna implementacja CART:")
        log(f"   - Dokladnosc: {custom_accuracy:.3f}")
        log(f"   - AUC: {custom_auc:.3f}")
        log(f"   - Czas trenowania: {custom_train_time:.2f} sekund")
        log(f"   - Czas predykcji: {custom_predict_time:.2f} sekund")
        
        log("2. Implementacja CART z scikit-learn:")
        log(f"   - Dokladnosc: {sklearn_accuracy:.3f}")
        log(f"   - AUC: {sklearn_auc:.3f}")
        log(f"   - Czas trenowania: {sklearn_train_time:.2f} sekund")
        log(f"   - Czas predykcji: {sklearn_predict_time:.2f} sekund")
        
        # Porównanie wydajności
        log("\nPorownanie wydajnosci:")
        train_ratio = sklearn_train_time / custom_train_time
        log(f"Wlasna implementacja jest {train_ratio:.1f}x szybsza/wolniejsza w trenowaniu.")
        
        predict_ratio = sklearn_predict_time / custom_predict_time
        log(f"Wlasna implementacja jest {predict_ratio:.1f}x szybsza/wolniejsza w predykcji.")
        
        # Wnioski
        log("\nWnioski:")
        if abs(custom_accuracy - sklearn_accuracy) < 0.01:
            log("1. Obie implementacje osiągają podobną dokladnosc.")
        elif custom_accuracy > sklearn_accuracy:
            log(f"1. Wlasna implementacja osiąga lepszą dokladnosc o {(custom_accuracy - sklearn_accuracy) * 100:.1f}%.")
        else:
            log(f"1. Implementacja scikit-learn osiąga lepsza dokladnosc o {(sklearn_accuracy - custom_accuracy) * 100:.1f}%.")
        
        if custom_train_time < sklearn_train_time:
            log("2. Wlasna implementacja jest szybsza w trenowaniu.")
        else:
            log("2. Implementacja scikit-learn jest szybsza w trenowaniu.")
        
        if custom_predict_time < sklearn_predict_time:
            log("3. Wlasna implementacja jest szybsza w predykcji.")
        else:
            log("3. Implementacja scikit-learn jest szybsza w predykcji.")

if __name__ == "__main__":
    main()