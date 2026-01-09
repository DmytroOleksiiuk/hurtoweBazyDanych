# 🍷 Wine Analytics & Food Pairings

Interaktywna aplikacja analityczna zbudowana w **Streamlit**, służąca do:
- eksploracji jakości czerwonych win,
- analizy zależności między cechami fizykochemicznymi a oceną jakości,
- wizualizacji danych (2D i 3D),
- rekomendowania najlepiej dopasowanych win do potraw i kuchni świata.

Projekt łączy **eksplorację danych (EDA)**, **wizualizację**, **filtrowanie**, **statystyki opisowe** oraz **prostą predykcję jakości wina**.

---

## 📂 Wykorzystywane datasety

1. **winequality-red.csv**  
   Dataset zawierający parametry fizykochemiczne czerwonych win oraz ocenę jakości (`quality`).

2. **wine_food_pairings.csv**  
   Dataset opisujący parowania win z jedzeniem (typ wina, kuchnia, danie, jakość dopasowania).

Pliki CSV muszą znajdować się w tym samym katalogu co plik aplikacji (`.py`).

---

## 🚀 Funkcjonalności aplikacji

### 1️⃣ Podstawowa eksploracja danych (EDA)
Dla **obu datasetów** aplikacja prezentuje:
- podgląd danych (`head()`),
- liczbę wierszy i kolumn,
- typy danych,
- brakujące wartości (ile i w których kolumnach),
- liczbę duplikatów.

Pozwala to szybko ocenić jakość danych wejściowych.

---

### 2️⃣ Analiza jakości wina (winequality-red.csv)

#### 🔎 Filtrowanie danych
Użytkownik może filtrować dane:
- po **ocenie jakości (`quality`)** – suwak,
- po **wybranej cesze fizykochemicznej** (np. alcohol, acidity) – zakres suwakami.

Po filtrach aplikacja pokazuje:
- liczbę pozostałych rekordów,
- tabelę wyników,
- szybkie statystyki (średnia, mediana, min, max).

---

#### 📈 Rozkłady i porównania
Panel umożliwia:
- wybór cechy do analizy,
- wyświetlenie:
  - histogramu,
  - boxplotu,
- porównanie rozkładów tej cechy dla dwóch grup jakości:
  - `quality ≤ X` vs `quality > X`,
  - `quality = A` vs `quality = B`.

Dla porównań prezentowane są także statystyki opisowe obu grup.

---

#### 🧊 Interaktywne wykresy 3D
Aplikacja oferuje **interaktywne wykresy 3D (Plotly)**:
- wybór osi X, Y, Z (dowolne cechy),
- kolorowanie punktów (np. jakość wina),
- możliwość obracania, przybliżania i eksplorowania punktów myszką.

Pozwala to analizować wielowymiarowe zależności między cechami.

---

#### 🤖 Predykcja jakości wina
Zastosowany jest prosty model **RandomForestRegressor**, który:
- przewiduje jakość wina na podstawie cech fizykochemicznych,
- prezentuje metryki jakości modelu (R², MAE),
- pokazuje ważność cech (feature importance),
- umożliwia interaktywną predykcję jakości dla danych podanych przez użytkownika.

---

### 3️⃣ Parowanie wina z jedzeniem

Moduł umożliwia:
- filtrowanie parowań po:
  - typie wina,
  - kategorii jedzenia,
  - kuchni,
  - minimalnej jakości parowania,
- prezentację wyników w tabeli,
- szybkie statystyki (średnia, mediana, min, max `pairing_quality`),
- analizę jakości parowań według typu wina,
- wyszukiwanie rekomendacji dla konkretnego dania.

---

### 4️⃣ Najlepsze dopasowania (kraj + jedzenie)

Moduł rekomendacyjny typu **user-centric**, w którym użytkownik:
- wybiera kraj / kuchnię,
- wybiera kategorię jedzenia lub wpisuje nazwę dania,
- ustawia minimalną jakość parowania,
- otrzymuje **TOP najlepiej dopasowane wina**,
- widzi statystyki i podsumowania rekomendacji.

---

## 🛠️ Technologie i biblioteki

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Plotly (interaktywne wykresy 3D)**
- **Scikit-learn**

---

## ▶️ Jak uruchomić aplikację

1. Zainstaluj wymagane biblioteki:
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
