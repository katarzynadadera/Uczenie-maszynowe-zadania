import math
from datetime import datetime, timedelta

# Funkcja do obliczenia biorytmów
def oblicz_biorytmy(dni_zycia):
    fizyczny = math.sin(2 * math.pi / 23 * dni_zycia)
    emocjonalny = math.sin(2 * math.pi / 28 * dni_zycia)
    intelektualny = math.sin(2 * math.pi / 33 * dni_zycia)
    return fizyczny, emocjonalny, intelektualny

# Funkcja do sprawdzenia, czy biorytmy są wysokie lub niskie i oceny następnego dnia
def ocen_biorytmy(fizyczny, emocjonalny, intelektualny, dni_zycia):
    biorytmy = {"fizyczny": fizyczny, "emocjonalny": emocjonalny, "intelektualny": intelektualny}
    wyniki = []
    
    for typ, wynik in biorytmy.items():
        nastepny_dzien = dni_zycia + 1
        nastepny_wynik = math.sin(2 * math.pi / (23 if typ == "fizyczny" else 28 if typ == "emocjonalny" else 33) * nastepny_dzien)

        if wynik > 0.5:
            wyniki.append(f"{typ.capitalize()} biorytm jest wysoki! Gratulacje!")
        elif wynik < -0.5:
            wyniki.append(f"{typ.capitalize()} biorytm jest niski")

            # Sprawdzamy, czy jutro będzie lepiej czy gorzej
            if nastepny_wynik > wynik:
                wyniki.append(f"Nie martw się, jutro będzie wyzszy wynik dla biometru {typ}!")
            elif nastepny_wynik < wynik:
                wyniki.append(f"Niestety, jutro będzie niższy wynik biometru {typ}")
        else:
            wyniki.append(f"{typ.capitalize()} biorytm jest na średnim poziomie")

    return wyniki


def main():
    imie = input("Podaj swoje imie: ")
    rok_urodzenia = int(input("Podaj rok urodzenia: "))
    miesiac_urodzenia = int(input("Podaj miesiąc urodzenia (numer): "))
    dzien_urodzenia = int(input("Podaj dzień urodzenia: "))

    data_urodzenia = datetime(rok_urodzenia, miesiac_urodzenia, dzien_urodzenia)
    dzisiaj = datetime.now()

    #obliczanie liczby dni od urodzin
    dni_zycia = (dzisiaj - data_urodzenia).days

    print(f"\nWitaj, {imie}!")

    fizyczny, emocjonalny, intelektualny = oblicz_biorytmy(dni_zycia)

    print(f"Dziś masz {dni_zycia} dni życia.")
    print(f"Twoje biorytmy na dziś:")
    print(f"Biorytm fizyczny: {fizyczny:.4f}")
    print(f"Biorytm emocjonalny: {emocjonalny:.4f}")
    print(f"Biorytm intelektualny: {intelektualny:.4f}")

    wyniki = ocen_biorytmy(fizyczny, emocjonalny, intelektualny, dni_zycia)

    for wynik in wyniki:
        print(wynik)

if __name__ == "__main__":
    main()
