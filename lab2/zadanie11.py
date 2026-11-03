from collections import Counter

def calculate_probabilities(training_data, test_cases):
    counts = Counter(row[-1] for row in training_data)  # Liczenie Y i N
    total = sum(counts.values())
    
    prior_Y = counts['Y'] / total
    prior_N = counts['N'] / total
    
    results = []
    
    for test_case in test_cases:
        conditional_Y = 1
        conditional_N = 1
        
        for i in range(len(test_case)):
            count_Y = sum(1 for row in training_data if row[i] == test_case[i] and row[-1] == 'Y')
            count_N = sum(1 for row in training_data if row[i] == test_case[i] and row[-1] == 'N')
            
            conditional_Y *= (count_Y + 1) / (counts['Y'] + 2)  # wygladzanie (dodaje 1 do licznika i 2 do mianownika)
            conditional_N *= (count_N + 1) / (counts['N'] + 2)
        
        prob_Y = prior_Y * conditional_Y
        prob_N = prior_N * conditional_N
        
        normalized_Y = prob_Y / (prob_Y + prob_N)
        normalized_N = prob_N / (prob_Y + prob_N)
        
        prediction = 'Y' if normalized_Y > normalized_N else 'N'
        results.append((test_case, normalized_Y, normalized_N, prediction))
    
    return results

# Dane treningowe
training_data = [
    [0, 1, 1, 'Y'],
    [0, 0, 1, 'N'],
    [0, 0, 0, 'Y'],
    [1, 1, 0, 'Y']
]

# Przypadek testowy
test_cases = [
    [1, 1, 0]
]

results = calculate_probabilities(training_data, test_cases)

# Wyświetlanie wyników
for test_case, prob_Y, prob_N, prediction in results:
    print(f"Test case: {test_case}")
    print(f"P(Y|x) = {prob_Y:.2%}")
    print(f"P(N|x) = {prob_N:.2%}")
