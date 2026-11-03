import numpy as np
import matplotlib.pyplot as plt
import random

def simulate_trebuchet():
    # Stałe
    v0 = 50  # Początkowa prędkość w m/s
    h = 100  # Wysokość trebuchetu w metrach
    g = 9.81  # Przyspieszenie ziemskie w m/s^2
    
    # Losowanie celu
    target = random.randint(50, 340)
    print(f"Cel znajduje się w odległości: {target} metrów")
    
    while True:
        # Pobranie kąta od użytkownika
        angle = float(input("Podaj kąt strzału w stopniach: "))
        alpha = np.radians(angle)  # Konwersja do radianów
        
        # Obliczenie zasięgu według wzoru
        range_dist = (v0 * np.cos(alpha) / g) * (v0 * np.sin(alpha) + np.sqrt(v0**2 * np.sin(alpha)**2 + 2 * g * h))
        
        # Wynik strzału
        if abs(range_dist - target) <= 5:
            print("Cel trafiony!")
            break
        else:
            print(f"Chybiony! Pocisk doleciał na odległość {range_dist:.2f} metrów. Spróbuj ponownie.")
    
    # Rysowanie trajektorii
    t_flight = (v0 * np.sin(alpha) + np.sqrt(v0**2 * np.sin(alpha)**2 + 2 * g * h)) / g
    t = np.linspace(0, t_flight, num=500)
    x = v0 * np.cos(alpha) * t
    y = h + v0 * np.sin(alpha) * t - 0.5 * g * t**2
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Trajektoria pocisku")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(target, color='red', linestyle='--', label="Cel")
    plt.xlabel("Odległość (m)")
    plt.ylabel("Wysokość (m)")
    plt.title("Symulacja strzału z Trebusza")
    plt.legend()
    plt.grid()
    
    # Zapis wykresu do pliku
    plt.savefig("trajektoria.png")
    plt.show()

simulate_trebuchet()
