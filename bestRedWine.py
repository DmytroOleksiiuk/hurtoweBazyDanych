import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics & Food Pairings",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍷 Wine Analytics & Food Pairings")
st.markdown(
    "Aplikacja do eksploracji jakości czerwonych win oraz "
    "parowania win z jedzeniem."
)

# ---------------------------------------------------------
# Stabilne ścieżki (Twoja struktura: CSV w root repo)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINE_QUALITY_PATH = os.path.join(BASE_DIR, "winequality-red.csv")
PAIRINGS_PATH = os.path.join(BASE_DIR, "wine_food_pairings.csv")

# ---------------------------------------------------------
# Funkcje wczytywania danych (Twoje + bezpieczniejsze)
# ---------------------------------------------------------
@st.cache_data
def load_wine_quality(path: str = WINE_QUALITY_PATH) -> pd.DataFrame:
    """
    winequality-red.csv w wielu wersjach ma separator ';'.
    Wczytujemy domyślnie, a jeśli wyjdzie 1 kolumna / brak 'quality' -> próbujemy sep=';'
    """
    df = pd.read_csv(path)
    if df.shape[1] == 1 or "quality" not in df.columns:
        df = pd.read_csv(path, sep=";")
    return df

@st.cache_data
def load_wine_food_pairings(path: str = PAIRINGS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # lekkie czyszczenie
    for col in ["wine_type", "wine_category", "food_item", "food_category", "cuisine", "quality_label", "description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "pairing_quality" in df.columns:
        df["pairing_quality"] = pd.to_numeric(df["pairing_quality"], errors="coerce")
    return df

# ---------------------------------------------------------
# Podstawowa eksploracja danych (EDA) - wymagania z zadania
# ---------------------------------------------------------
def basic_eda(df: pd.DataFrame, title: str):
    st.subheader(title)

    # head()
    st.markdown("### Podgląd danych (head)")
    st.dataframe(df.head(), use_container_width=True)

    # liczba wierszy/kolumn
    st.markdown("### Rozmiar danych")
    st.write(f"**Wiersze:** {df.shape[0]}  |  **Kolumny:** {df.shape[1]}")

    # typy danych
    st.markdown("### Typy danych")
    dtypes_df = pd.DataFrame({
        "kolumna": df.columns,
        "typ": [str(t) for t in df.dtypes.values]
    })
    st.dataframe(dtypes_df, hide_index=True, use_container_width=True)

    # brakujące wartości (ile i gdzie)
    st.markdown("### Brakujące wartości")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_df = pd.DataFrame({
        "kolumna": missing.index,
        "brakujące": missing.values,
        "procent": (missing.values / len(df) * 100).round(2)
    })

    if int(missing.sum()) == 0:
        st.success("✅ Brak brakujących wartości w datasetcie.")
        st.dataframe(missing_df, hide_index=True, use_container_width=True)
    else:
        st.warning(f"⚠️ Wykryto brakujące wartości: **{int(missing.sum())}**")
        st.dataframe(
            missing_df[missing_df["brakujące"] > 0],
            hide_index=True,
            use_container_width=True
        )

    # duplikaty (ile)
    st.markdown("### Duplikaty")
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        st.success("✅ Brak duplikatów.")
    else:
        st.warning(f"⚠️ Liczba duplikatów: **{dup_count}**")
        with st.expander("Pokaż przykładowe duplikaty"):
            st.dataframe(df[df.duplicated(keep=False)].head(50), use_container_width=True)

# ---------------------------------------------------------
# Próba wczytania danych + komunikaty błędów
# ---------------------------------------------------------
wine_quality_df, pairings_df = None, None
wine_quality_error, pairings_error = None, None

try:
    wine_quality_df = load_wine_quality()
except Exception as e:
    wine_quality_error = str(e)

try:
    pairings_df = load_wine_food_pairings()
except Exception as e:
    pairings_error = str(e)

# ---------------------------------------------------------
# Sidebar – wybór modułu (Opcja A dodana)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ustawienia")
module = st.sidebar.radio(
    "Wybierz moduł:",
    options=[
        "Eksploracja danych",
        "Analiza jakości wina",
        "Parowanie wina z jedzeniem"
    ]
)

# =========================================================
# 0. EKSPLORACJA DANYCH (EDA) - dla obu datasetów
# =========================================================
if module == "Eksploracja danych":
    st.header("🔎 Podstawowa eksploracja danych (EDA)")

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`\n\n"
            "Upewnij się, że plik jest w tym samym katalogu co plik aplikacji."
        )
    else:
        basic_eda(wine_quality_df, "🧪 winequality-red.csv")

    st.divider()

    if pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`\n\n"
            "Upewnij się, że plik jest w tym samym katalogu co plik aplikacji."
        )
    else:
        basic_eda(pairings_df, "🍽️ wine_food_pairings.csv")

# =========================================================
# 1. ANALIZA JAKOŚCI WINA (winequality-red.csv)
# =========================================================
elif module == "Analiza jakości wina":
    st.subheader("📊 Analiza jakości czerwonych win")

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co `app.py` / `bestRedWine.py`."
        )
        st.stop()

    df = wine_quality_df.copy()

    # -------------------------
    # Podstawowe informacje
    # -------------------------
    st.markdown("### Podgląd danych")
    st.write("Pierwsze wiersze datasetu:")
    st.dataframe(df.head())

    with st.expander("Informacje o datasetcie"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Kształt (liczba rekordów, liczba kolumn):**")
            st.write(df.shape)
            st.write("**Typy danych:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Podstawowe statystyki opisowe:**")
            st.write(df.describe().T)

    # -------------------------
    # Filtrowanie po jakości
    # -------------------------
    st.markdown("### Filtrowanie po ocenie jakości")
    min_q = int(df["quality"].min())
    max_q = int(df["quality"].max())

    quality_range = st.slider(
        "Zakres jakości (kolumna `quality`):",
        min_value=min_q,
        max_value=max_q,
        value=(min_q, max_q),
        step=1
    )

    filtered = df[(df["quality"] >= quality_range[0]) & (df["quality"] <= quality_range[1])]

    st.write(f"Liczba rekordów po filtrze: **{filtered.shape[0]}**")
    st.dataframe(filtered.head())

    # -------------------------
    # Rozkład jakości
    # -------------------------
    st.markdown("### Rozkład jakości wina")

    fig, ax = plt.subplots()
    ax.hist(df["quality"], bins=range(min_q, max_q + 2), edgecolor="black")
    ax.set_xlabel("Jakość (quality)")
    ax.set_ylabel("Liczba próbek")
    ax.set_title("Histogram ocen jakości wina")
    st.pyplot(fig)

    # -------------------------
    # Korelacja cech
    # -------------------------
    st.markdown("### Korelacje między cechami")

    corr = df.corr(numeric_only=True)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Macierz korelacji")
    st.pyplot(fig_corr)

    # -------------------------
    # Scatter: wybrana cecha vs jakość
    # -------------------------
    st.markdown("### Zależność cechy od jakości")

    feature_cols = [c for c in df.columns if c != "quality"]
    default_feature = "alcohol" if "alcohol" in feature_cols else feature_cols[0]
    x_feature = st.selectbox("Wybierz cechę (oś X):", feature_cols, index=feature_cols.index(default_feature))

    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(df[x_feature], df["quality"], alpha=0.6)
    ax_scatter.set_xlabel(x_feature)
    ax_scatter.set_ylabel("quality")
    ax_scatter.set_title(f"{x_feature} vs quality")
    st.pyplot(fig_scatter)

    # -------------------------
    # Prosty model predykcji jakości
    # -------------------------
    st.markdown("### 🤖 Prosty model predykcji jakości (RandomForest)")

    with st.expander("Parametry modelu i metryki"):
        test_size = st.slider(
            "Udział danych testowych",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
        n_estimators = st.slider(
            "Liczba drzew (n_estimators)",
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )
        random_state = st.number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1
        )

        # Przygotowanie danych
        X = df.drop("quality", axis=1)
        y = df["quality"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("R² na zbiorze testowym", f"{r2:.3f}")
        with col_m2:
            st.metric("MAE na zbiorze testowym", f"{mae:.3f}")

        # Ważność cech
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.markdown("**Ważność cech (feature importance):**")
        st.bar_chart(importances)

    # -------------------------
    # Interaktywna predykcja dla użytkownika
    # -------------------------
    st.markdown("### 🔮 Predykcja jakości dla podanych parametrów")

    with st.form("prediction_form"):
        cols = st.columns(3)
        user_input = {}

        for i, col_name in enumerate(X.columns):
            col = cols[i % 3]
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())
            mean_val = float(df[col_name].mean())
            step = (max_val - min_val) / 100 if max_val > min_val else 0.01

            user_input[col_name] = col.slider(
                col_name,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=step
            )

        submitted = st.form_submit_button("Oblicz przewidywaną jakość")

    if "model" in locals() and submitted:
        input_df = pd.DataFrame([user_input])
        pred_quality = model.predict(input_df)[0]
        st.success(f"Przewidywana jakość wina: **{pred_quality:.2f}** (w skali jak w kolumnie `quality`)")

# =========================================================
# 2. PAROWANIE WINA Z JEDZENIEM (wine_food_pairings.csv)
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader("🍽️ Parowanie wina z jedzeniem")

    if pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co `app.py` / `bestRedWine.py`."
        )
        st.stop()

    dfp = pairings_df.copy()

    # -------------------------
    # Podgląd danych
    # -------------------------
    st.markdown("### Podgląd danych")
    st.dataframe(dfp.head())

    with st.expander("Informacje o datasetcie"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Kształt:**", dfp.shape)
            st.write("**Kolumny:**")
            st.write(list(dfp.columns))
        with col2:
            if "wine_type" in dfp.columns:
                st.write("wine_type:", dfp["wine_type"].unique()[:10])
            if "food_category" in dfp.columns:
                st.write("food_category:", dfp["food_category"].unique()[:10])
            if "cuisine" in dfp.columns:
                st.write("cuisine:", dfp["cuisine"].unique()[:10])
            if "quality_label" in dfp.columns:
                st.write("quality_label:", dfp["quality_label"].unique())

    # -------------------------
    # Filtrowanie parowań
    # -------------------------
    st.markdown("### Filtrowanie rekomendacji")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        wine_type_sel = st.multiselect(
            "Typ wina (`wine_type`):",
            options=sorted(dfp["wine_type"].dropna().unique()),
            default=None
        )

    with col_f2:
        food_cat_sel = st.multiselect(
            "Kategoria jedzenia (`food_category`):",
            options=sorted(dfp["food_category"].dropna().unique()),
            default=None
        )

    with col_f3:
        cuisine_sel = st.multiselect(
            "Kuchnia (`cuisine`):",
            options=sorted(dfp["cuisine"].dropna().unique()),
            default=None
        )

    with col_f4:
        min_pair_quality = int(np.nanmin(dfp["pairing_quality"])) if "pairing_quality" in dfp.columns else 1
        max_pair_quality = int(np.nanmax(dfp["pairing_quality"])) if "pairing_quality" in dfp.columns else 5
        pairing_quality_sel = st.slider(
            "Minimalna ocena parowania (`pairing_quality`):",
            min_value=min_pair_quality,
            max_value=max_pair_quality,
            value=min_pair_quality,
            step=1
        )

    filt = dfp.copy()
    if wine_type_sel:
        filt = filt[filt["wine_type"].isin(wine_type_sel)]
    if food_cat_sel:
        filt = filt[filt["food_category"].isin(food_cat_sel)]
    if cuisine_sel:
        filt = filt[filt["cuisine"].isin(cuisine_sel)]

    if "pairing_quality" in filt.columns:
        filt = filt[filt["pairing_quality"].fillna(-999) >= pairing_quality_sel]

    st.markdown(f"Znaleziono **{filt.shape[0]}** dopasowań.")
    st.dataframe(
        filt[
            [c for c in [
                "wine_type",
                "wine_category",
                "food_item",
                "food_category",
                "cuisine",
                "pairing_quality",
                "quality_label",
                "description"
            ] if c in filt.columns]
        ].sort_values(by="pairing_quality", ascending=False) if "pairing_quality" in filt.columns else filt,
        use_container_width=True
    )

    # -------------------------
    # Statystyki jakości parowań
    # -------------------------
    st.markdown("### Podsumowanie jakości parowań")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        if "quality_label" in dfp.columns:
            st.write("**Liczba parowań per etykieta jakości (`quality_label`):**")
            st.bar_chart(dfp["quality_label"].value_counts())

    with col_s2:
        if "pairing_quality" in dfp.columns and "wine_type" in dfp.columns:
            st.write("**Średnia ocena parowania per typ wina (`wine_type`):**")
            mean_quality_by_wine = (
                dfp.groupby("wine_type")["pairing_quality"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(mean_quality_by_wine)

    # -------------------------
    # Prosta „rekomendacja” na podstawie wyboru użytkownika
    # -------------------------
    st.markdown("### 🔍 Znajdź rekomendacje dla konkretnego dania")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        chosen_food = st.text_input(
            "Podaj nazwę dania (część nazwy z kolumny `food_item`):",
            value=""
        )
    with col_r2:
        cuisine_options = ["(dowolna)"] + sorted(dfp["cuisine"].dropna().unique().tolist()) if "cuisine" in dfp.columns else ["(dowolna)"]
        chosen_cuisine = st.selectbox(
            "Wybierz kuchnię (opcjonalnie):",
            options=cuisine_options
        )

    if chosen_food.strip():
        rec = dfp[dfp["food_item"].str.contains(chosen_food, case=False, na=False)]
        if chosen_cuisine != "(dowolna)" and "cuisine" in rec.columns:
            rec = rec[rec["cuisine"] == chosen_cuisine]

        if "pairing_quality" in rec.columns:
            rec = rec.sort_values(by="pairing_quality", ascending=False)

        if rec.empty:
            st.warning("Brak rekomendacji spełniających kryteria.")
        else:
            st.success(f"Znaleziono **{rec.shape[0]}** rekomendacji.")
            st.dataframe(
                rec[
                    [c for c in [
                        "food_item",
                        "cuisine",
                        "wine_type",
                        "wine_category",
                        "pairing_quality",
                        "quality_label",
                        "description"
                    ] if c in rec.columns]
                ].head(20),
                use_container_width=True
            )
    else:
        st.info("Wpisz fragment nazwy dania, aby zobaczyć rekomendacje.")
