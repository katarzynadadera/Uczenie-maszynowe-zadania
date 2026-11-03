import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 1. Wczytanie danych twarzy z min_faces_per_person = 50
face_data = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
X = face_data.data
Y = face_data.target
target_names = face_data.target_names
h, w = face_data.images[0].shape

# 2. Podział danych na treningowe i testowe (w tym obrazy)
X_train, X_test, Y_train, Y_test, img_train, img_test = train_test_split(
    X, Y, face_data.images, random_state=42
)

# 3. Pipeline: PCA + SVC
pca = PCA(n_components=100, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', random_state=42)
model = Pipeline([('pca', pca), ('svc', svc)])

# 4. GridSearchCV do znalezienia najlepszych hiperparametrów
param_grid = {
    'svc__C': [0.1, 1, 10, 30, 100, 300],
    'svc__gamma': [0.0005, 0.001, 0.003, 0.005, 0.01, 0.03]
}
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)

# 5. Najlepszy model i predykcja
model_best = grid_search.best_estimator_
pred = model_best.predict(X_test)

print("Dokładność walidacji krzyżowej:", grid_search.best_score_)
print('Najlepsze parametry:', grid_search.best_params_)
print('Dokładność na zbiorze testowym: {:.1f}%'.format(model_best.score(X_test, Y_test)*100))
print(classification_report(Y_test, pred, target_names=target_names))

# 6. Funkcja: pokazanie błędnie sklasyfikowanych twarzy
def plot_misclassified_faces(images, true_labels, pred_labels, names, max_images=12):
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes = axes.flat
    errors = np.where(true_labels != pred_labels)[0]
    for i, ax in enumerate(axes):
        if i >= len(errors):
            ax.axis('off')
            continue
        idx = errors[i]
        ax.imshow(images[idx], cmap='bone')
        ax.set_title(f"True: {names[true_labels[idx]]}\nPred: {names[pred_labels[idx]]}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.suptitle("Źle sklasyfikowane twarze", fontsize=16)
    plt.subplots_adjust(top=0.88)
    plt.show()

# 7. Pokazanie błędnych predykcji
plot_misclassified_faces(img_test, Y_test, pred, target_names)

# 8. PCA 2D: wizualizacja danych
pca_vis = PCA(n_components=2, whiten=True, random_state=42)
X_vis = pca_vis.fit_transform(X)

plt.figure(figsize=(10, 8))
for i, name in enumerate(target_names):
    plt.scatter(X_vis[Y == i, 0], X_vis[Y == i, 1], label=name, alpha=0.5)
plt.legend()
plt.title("Rzut PCA do 2D (wizualizacja klas)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
