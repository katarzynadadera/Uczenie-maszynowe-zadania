# Import niezbędnych bibliotek
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings

# Ignoruj ostrzeżenia o zbieżności, bo wiemy już o tym problemie
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Wczytanie danych twarzy bez określania ścieżki do plików pairsDevTrain.txt
try:
    face_data = fetch_lfw_people(
        min_faces_per_person=80, 
        download_if_missing=True,
        resize=0.5  # Zmniejszenie rozmiaru obrazów dla przyspieszenia obliczeń
    )
except Exception as e:
    print(f"Błąd wczytywania danych: {e}")
    # Alternatywne wczytanie z domyślnej ścieżki
    face_data = fetch_lfw_people(min_faces_per_person=80, download_if_missing=True)

# Wyodrębnienie cech i etykiet
X = face_data.data
Y = face_data.target

# Wyświetlenie informacji o zbiorze danych
print('Rozmiar danych wejściowych:', X.shape)
print('Rozmiar danych wyjściowych:', Y.shape)
print('Nazwy etykiet:', face_data.target_names)

# Sprawdzenie rozkładu klas
for i in range(len(face_data.target_names)):
    print(f'Klasa {i} ({face_data.target_names[i]}) ma {(Y == i).sum()} próbek.')

# Podział danych na zbiór treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

# Utworzenie potoku z PCA do redukcji wymiarowości i LinearSVC do klasyfikacji
pca = PCA(n_components=100, whiten=True, random_state=42)
linear_svc = LinearSVC(
    random_state=42,
    max_iter=10000,  # Znacznie zwiększona liczba iteracji dla zapewnienia zbieżności
    class_weight='balanced',  # Obsługa niezbalansowanych klas
    dual=True  # Zawsze używaj dual=True dla funkcji straty hinge
)
model = Pipeline([('pca', pca), ('linear_svc', linear_svc)])

# Zdefiniowanie POPRAWNYCH hiperparametrów do dostrojenia dla LinearSVC
# Unikamy kombinacji dual=False i loss='hinge', które powodują błędy
parameters_pipeline = {
    # Szeroki zakres wartości C - skala logarytmiczna
    'linear_svc__C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
    
    # Szerszy zakres komponentów PCA
    'pca__n_components': [50, 75, 100, 125, 150, 200, 250, 300],
    
    # Opcjonalnie możesz również eksperymentować z tymi parametrami
    'linear_svc__loss': ['squared_hinge'],
    'linear_svc__dual': [True, False]  # False dla większej wydajności gdy n_samples > n_features
}

# Wykonanie przeszukiwania siatki z walidacją krzyżową
print("Rozpoczynam przeszukiwanie siatki...")
grid_search = GridSearchCV(
    model, 
    parameters_pipeline, 
    n_jobs=-1,  # Wykorzystanie wszystkich dostępnych rdzeni
    cv=5,  # 5-krotna walidacja krzyżowa
    verbose=1  # Pokazuj postęp
)
grid_search.fit(X_train, Y_train)

# Wyświetlenie najlepszych parametrów i wyników
print('\nNajlepsze parametry modelu:\n', grid_search.best_params_)
print('Najlepsza uśredniona skuteczność:', grid_search.best_score_)

# Ocena modelu na zbiorze testowym
model_best = grid_search.best_estimator_
accuracy = model_best.score(X_test, Y_test)
print(f'\nDokładność na zbiorze testowym: {accuracy*100:.1f}%')

# Wygenerowanie szczegółowego raportu klasyfikacji
pred = model_best.predict(X_test)
print('\nRaport klasyfikacji:')
print(classification_report(Y_test, pred, target_names=face_data.target_names))

# Dodatkowa analiza: Zbadanie znaczenia komponentów PCA
pca_fitted = model_best.named_steps['pca']
n_components = pca_fitted.n_components_
explained_variance_ratio = pca_fitted.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

print(f"\nAnaliza PCA:")
print(f"Liczba użytych komponentów: {n_components}")
print(f"Całkowita wyjaśniona wariancja: {cumulative_variance[-1]*100:.2f}%")
print(f"Pierwsze 10 komponentów wyjaśnia: {cumulative_variance[9]*100:.2f}% wariancji")

# Analiza parametrów LinearSVC
svc_fitted = model_best.named_steps['linear_svc']
print(f"\nParametry LinearSVC:")
print(f"Wartość C: {svc_fitted.C}")
print(f"Funkcja straty: {svc_fitted.loss}")
print(f"Kara: {svc_fitted.penalty}")
print(f"Formulacja dualna: {svc_fitted.dual}")

print("\nPodsumowanie:")
print(f"Najlepsza dokładność rozpoznawania twarzy: {accuracy*100:.1f}%")
print("Hiperparametry, które należy dostosować:")
print("1. C - parametr regularyzacji (kompromis między dopasowaniem a generalizacją)")
print("2. n_components w PCA - liczba głównych składowych")
print("3. dual - czy używać sformułowania dualnego czy prymalnego")
print("4. loss - funkcja straty (squared_hinge jest bezpieczniejsza niż hinge)")