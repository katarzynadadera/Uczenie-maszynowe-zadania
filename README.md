# Uczenie Maszynowe - Laboratoria i Zadania

Repozytorium zawiera zbiór skryptów i projektów realizowanych w ramach zajęć z przedmiotu **Uczenie Maszynowe**. Zbiór obejmuje różnorodne zagadnienia i techniki użycia sztucznej inteligencji, analizy danych oraz modelowania predykcyjnego.

## Tematyka zadań

W repozytorium znajdują się rozwiązania zadań obejmujących między innymi:

* **Przetwarzanie Języka Naturalnego (NLP):** Tokenizacja, lematyzacja, stemming, usuwanie stop words, reprezentacje tekstu (Bag of Words, Word2Vec) oraz wizualizacja t-SNE.
* **Rozpoznawanie obrazów:** Redukcja wymiarowości z wykorzystaniem PCA połączona z klasyfikacją maszyn wektorów nośnych (SVM) na zbiorze LFW (Labeled Faces in the Wild).
* **Klasyfikacja i ocena modeli (CTR, Systemy rekomendacyjne):** Przewidywanie kliknięć, analiza filmów (MovieLens) z wykorzystaniem algorytmów takich jak naiwny klasyfikator Bayesa (Naive Bayes), Drzewa Decyzyjne, Random Forest, XGBoost czy SGDClassifier z regularyzacjami (L1, L2, ElasticNet). Obliczanie metryk jakości (np. AUC ROC).
* **Regresja i Szeregi Czasowe:** Przewidywanie cen akcji na podstawie wskaźników giełdowych, inżynieria cech (feature engineering), testowanie modeli regresyjnych oraz tworzenie modeli zespołowych (Ensemble Learning).
* **Proste symulacje:** np. symulacja rzutu z trebusza z wizualizacją trajektorii.

## Wykorzystywane technologie i biblioteki

Do realizacji zadań wykorzystano język **Python** oraz popularne biblioteki do analizy danych i uczenia maszynowego:
* `scikit-learn`
* `pandas`, `numpy`
* `matplotlib`
* `nltk`, `gensim`
* `xgboost`

## 📊 Wyniki i wizualizacje

Większość skryptów generuje po uruchomieniu odpowiednie wyniki w konsoli (metryki takie jak *Accuracy*, *AUC*, *MSE*, *R^2*) oraz zapisuje wizualizacje w postaci wykresów (pliki `.png` lub `.pdf`), które pozwalają na łatwiejszą analizę i porównanie różnych podejść oraz hiperparametrów modeli.