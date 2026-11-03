import numpy as np

def naive_bayes_prediction(matrix):
    matrix = np.array(matrix)
    n, m = matrix.shape
    
    # Oddzielamy dane treningowe od testowego wiersza
    training_data = matrix[:-1, :]
    test_row = matrix[-1, :-1]
    
    # Dzielimy dane na klasy Y=1 i Y=0
    liked = training_data[training_data[:, -1] == 1]
    not_liked = training_data[training_data[:, -1] == 0]
    
    # P(Y=1) i P(Y=0)
    p_y1 = len(liked) / len(training_data)
    p_y0 = len(not_liked) / len(training_data)
    
    # Wygładzanie Laplace’a (dodajemy 1 do licznika, zwiększamy mianownik o 2)
    alpha = 1  # Parametr Laplace'a
    k = 2  # Liczba możliwych wartości cechy (0 lub 1)
    
    p_x_given_y1 = np.prod([
        ((np.sum(liked[:, i]) + alpha) / (len(liked) + k)) if test_row[i] == 1 else 
        ((len(liked) - np.sum(liked[:, i]) + alpha) / (len(liked) + k))
        for i in range(m-1)
    ])
    
    p_x_given_y0 = np.prod([
        ((np.sum(not_liked[:, i]) + alpha) / (len(not_liked) + k)) if test_row[i] == 1 else 
        ((len(not_liked) - np.sum(not_liked[:, i]) + alpha) / (len(not_liked) + k))
        for i in range(m-1)
    ])
    
    # Prawdopodobieństwa łączone
    p_x_and_y1 = p_x_given_y1 * p_y1
    p_x_and_y0 = p_x_given_y0 * p_y0
    
    # Normalizacja do otrzymania P(Y=1 | X)
    if (p_x_and_y1 + p_x_and_y0) == 0:
        return 0.5  # Neutralne prawdopodobieństwo w przypadku zerowego mianownika
    
    p_y1_given_x = p_x_and_y1 / (p_x_and_y1 + p_x_and_y0)
    
    return p_y1_given_x

# Macierz danych zgodnie z podanym przykładem
matrix = [
    [0, 1, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 1],
    [1, 1, 0, '?']  # Testowy użytkownik
]

# Konwersja '?' na brakującą wartość (nie wpływa na obliczenia)
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i][j] == '?':
            matrix[i][j] = 0  # Można zastąpić dowolną wartością, nie jest używana

# Obliczenie prawdopodobieństwa
prob = naive_bayes_prediction(matrix)
print(f"Prawdopodobieństwo, że użytkownik polubi film: {prob:.2%}")
