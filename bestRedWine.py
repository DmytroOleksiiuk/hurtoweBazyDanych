import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
# Funkcje wczytywania danych
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

    for col in ["wine_type", "wine_category", "food_item", "food_category", "cuisine", "quality_label", "description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "pairing_quality" in df.columns:
        df["pairing_quality"] = pd.to_numeric(df["pairing_quality"], errors="coerce")

    return df


# ---------------------------------------------------------
# Podstawowa eksploracja danych (EDA)
# ---------------------------------------------------------
def basic_eda(df: pd.DataFrame, title: str):
    st.subheader(title)

    st.markdown("### Podgląd danych (head)")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Rozmiar danych")
    st.write(f"**Wiersze:** {df.shape[0]}  |  **Kolumny:** {df.shape[1]}")

    st.markdown("### Typy danych")
    dtypes_df = pd.DataFrame({
        "kolumna": df.columns,
        "typ": [str(t) for t in df.dtypes.values]
    })
    st.dataframe(dtypes_df, hide_index=True, use_container_width=True)

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

    st.markdown("### Duplikaty")
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        st.success("✅ Brak duplikatów.")
    else:
        st.warning(f"⚠️ Liczba duplikatów: **{dup_count}**")
        with st.expander("Pokaż przykładowe duplikaty"):
            st.dataframe(df[df.duplicated(keep=False)].head(50), use_container_width=True)


# ---------------------------------------------------------
# Szybkie statystyki (po filtrach)
# ---------------------------------------------------------
def quick_stats(df: pd.DataFrame, numeric_cols: list[str], title: str = "Szybkie statystyki"):
    st.markdown(f"### {title}")
    if df.empty:
        st.info("Brak danych po filtrach – brak statystyk.")
        return

    cols_to_use = [c for c in numeric_cols if c in df.columns]
    if not cols_to_use:
        st.info("Brak kolumn numerycznych do statystyk.")
        return

    stats = pd.DataFrame({
        "kolumna": cols_to_use,
        "średnia": [df[c].mean() for c in cols_to_use],
        "mediana": [df[c].median() for c in cols_to_use],
        "min": [df[c].min() for c in cols_to_use],
        "max": [df[c].max() for c in cols_to_use],
    })
    st.dataframe(stats, hide_index=True, use_container_width=True)


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
# Sidebar – wybór modułu
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ustawienia")
module = st.sidebar.radio(
    "Wybierz moduł:",
    options=[
        "Eksploracja danych",
        "Analiza jakości wina",
        "Parowanie wina z jedzeniem",
        "Najlepsze dopasowania (kraj + jedzenie)"
    ]
)

# =========================================================
# 0. EKSPLORACJA DANYCH (EDA)
# =========================================================
if module == "Eksploracja danych":
    st.header("🔎 Podstawowa eksploracja danych (EDA)")

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`"
        )
    else:
        basic_eda(wine_quality_df, "🧪 winequality-red.csv")

    st.divider()

    if pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`"
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
            f"Komunikat błędu:\n`{wine_quality_error}`"
        )
        st.stop()

    df = wine_quality_df.copy()

    # -------------------------
    # Podgląd danych
    # -------------------------
    st.markdown("### Podgląd danych")
    st.dataframe(df.head(), use_container_width=True)

    with st.expander("Informacje o datasetcie"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Kształt (rekordy, kolumny):**")
            st.write(df.shape)
            st.write("**Typy danych:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Statystyki opisowe:**")
            st.write(df.describe().T)

    # -------------------------
    # Filtrowanie: quality + wybrana cecha (dwa suwaki)
    # -------------------------
    st.markdown("### Filtrowanie: quality + wybrana cecha")

    min_q = int(df["quality"].min())
    max_q = int(df["quality"].max())

    quality_range = st.slider(
        "Zakres jakości (quality):",
        min_value=min_q,
        max_value=max_q,
        value=(min_q, max_q),
        step=1
    )

    feature_cols = [c for c in df.columns if c != "quality"]
    default_feature = "alcohol" if "alcohol" in feature_cols else feature_cols[0]

    chosen_feature = st.selectbox(
        "Wybierz cechę do filtrowania:",
        feature_cols,
        index=feature_cols.index(default_feature),
        key="filter_feature"
    )

    f_min = float(df[chosen_feature].min())
    f_max = float(df[chosen_feature].max())

    feature_range = st.slider(
        f"Zakres dla cechy: {chosen_feature}",
        min_value=f_min,
        max_value=f_max,
        value=(f_min, f_max),
        key="filter_feature_range"
    )

    filtered = df[
        (df["quality"] >= quality_range[0]) & (df["quality"] <= quality_range[1]) &
        (df[chosen_feature] >= feature_range[0]) & (df[chosen_feature] <= feature_range[1])
    ]

    st.write(f"✅ Rekordy po filtrach: **{filtered.shape[0]}** / {df.shape[0]}")

    st.markdown("### Wyniki po filtrach")
    st.dataframe(filtered, use_container_width=True)

    quick_stats(
        filtered,
        numeric_cols=["quality", chosen_feature],
        title="Szybkie wnioski: quality + wybrana cecha"
    )

    # =====================================================
    # Rozkłady i porównania (panel)
    # =====================================================
    st.markdown("## 📈 Rozkłady i porównania (winequality-red)")
    st.caption("Panel działa na danych **po filtrach** (powyżej).")

    if filtered.empty:
        st.warning("Brak danych po filtrach – nie mogę narysować rozkładów.")
    else:
        c1, c2 = st.columns([1, 1])

        with c1:
            dist_feature = st.selectbox(
                "Wybierz cechę do analizy rozkładu:",
                feature_cols,
                index=feature_cols.index(default_feature),
                key="dist_feature"
            )

        with c2:
            compare_mode = st.radio(
                "Tryb porównania (2 grupy jakości):",
                options=["quality ≤ X vs quality > X", "quality = A vs quality = B"],
                horizontal=True,
                key="compare_mode"
            )

        st.markdown("### Histogram cechy")
        fig_h, ax_h = plt.subplots()
        ax_h.hist(filtered[dist_feature].dropna(), bins=30, edgecolor="black")
        ax_h.set_xlabel(dist_feature)
        ax_h.set_ylabel("Liczba próbek")
        ax_h.set_title(f"Histogram: {dist_feature}")
        st.pyplot(fig_h)

        st.markdown("### Boxplot cechy")
        fig_b, ax_b = plt.subplots()
        ax_b.boxplot(filtered[dist_feature].dropna(), vert=True, labels=[dist_feature])
        ax_b.set_title(f"Boxplot: {dist_feature}")
        st.pyplot(fig_b)

        st.markdown("### Porównanie rozkładów dla 2 grup jakości")

        if compare_mode == "quality ≤ X vs quality > X":
            q_min = int(filtered["quality"].min())
            q_max = int(filtered["quality"].max())

            if q_min == q_max:
                st.info("Po filtrach masz tylko jedną wartość quality – porównanie progowe nie ma sensu.")
            else:
                x = st.slider(
                    "Wybierz próg X:",
                    min_value=q_min,
                    max_value=q_max - 1,
                    value=min(q_min + 1, q_max - 1),
                    step=1,
                    key="threshold_x"
                )

                g1 = filtered[filtered["quality"] <= x][dist_feature].dropna()
                g2 = filtered[filtered["quality"] > x][dist_feature].dropna()

                st.write(f"Grupa 1 (quality ≤ {x}): **{len(g1)}** rekordów")
                st.write(f"Grupa 2 (quality > {x}): **{len(g2)}** rekordów")

                fig_ch, ax_ch = plt.subplots()
                ax_ch.hist(g1, bins=30, alpha=0.6, label=f"quality ≤ {x}", edgecolor="black")
                ax_ch.hist(g2, bins=30, alpha=0.6, label=f"quality > {x}", edgecolor="black")
                ax_ch.set_xlabel(dist_feature)
                ax_ch.set_ylabel("Liczba próbek")
                ax_ch.set_title(f"Porównanie histogramów: {dist_feature}")
                ax_ch.legend()
                st.pyplot(fig_ch)

                fig_cb, ax_cb = plt.subplots()
                ax_cb.boxplot([g1, g2], labels=[f"≤ {x}", f"> {x}"])
                ax_cb.set_title(f"Porównanie boxplotów: {dist_feature}")
                ax_cb.set_ylabel(dist_feature)
                st.pyplot(fig_cb)

                comp_stats = pd.DataFrame({
                    "grupa": [f"quality ≤ {x}", f"quality > {x}"],
                    "średnia": [g1.mean() if len(g1) else np.nan, g2.mean() if len(g2) else np.nan],
                    "mediana": [g1.median() if len(g1) else np.nan, g2.median() if len(g2) else np.nan],
                    "min": [g1.min() if len(g1) else np.nan, g2.min() if len(g2) else np.nan],
                    "max": [g1.max() if len(g1) else np.nan, g2.max() if len(g2) else np.nan],
                })
                st.markdown("#### Szybkie statystyki (porównanie)")
                st.dataframe(comp_stats, hide_index=True, use_container_width=True)

        else:
            qualities = sorted(filtered["quality"].dropna().unique().tolist())

            if len(qualities) < 2:
                st.info("Po filtrach masz mniej niż 2 różne wartości quality – wybierz szersze filtry.")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    q_a = st.selectbox("Wybierz jakość A:", qualities, index=0, key="qa")
                with col_b:
                    default_idx = 1 if len(qualities) > 1 else 0
                    q_b = st.selectbox("Wybierz jakość B:", qualities, index=default_idx, key="qb")

                if q_a == q_b:
                    st.warning("Wybierz dwie różne wartości jakości (A != B).")
                else:
                    g1 = filtered[filtered["quality"] == q_a][dist_feature].dropna()
                    g2 = filtered[filtered["quality"] == q_b][dist_feature].dropna()

                    st.write(f"Grupa 1 (quality = {q_a}): **{len(g1)}** rekordów")
                    st.write(f"Grupa 2 (quality = {q_b}): **{len(g2)}** rekordów")

                    fig_ch, ax_ch = plt.subplots()
                    ax_ch.hist(g1, bins=30, alpha=0.6, label=f"quality = {q_a}", edgecolor="black")
                    ax_ch.hist(g2, bins=30, alpha=0.6, label=f"quality = {q_b}", edgecolor="black")
                    ax_ch.set_xlabel(dist_feature)
                    ax_ch.set_ylabel("Liczba próbek")
                    ax_ch.set_title(f"Porównanie histogramów: {dist_feature}")
                    ax_ch.legend()
                    st.pyplot(fig_ch)

                    fig_cb, ax_cb = plt.subplots()
                    ax_cb.boxplot([g1, g2], labels=[f"{q_a}", f"{q_b}"])
                    ax_cb.set_title(f"Porównanie boxplotów: {dist_feature}")
                    ax_cb.set_ylabel(dist_feature)
                    st.pyplot(fig_cb)

                    comp_stats = pd.DataFrame({
                        "grupa": [f"quality = {q_a}", f"quality = {q_b}"],
                        "średnia": [g1.mean() if len(g1) else np.nan, g2.mean() if len(g2) else np.nan],
                        "mediana": [g1.median() if len(g1) else np.nan, g2.median() if len(g2) else np.nan],
                        "min": [g1.min() if len(g1) else np.nan, g2.min() if len(g2) else np.nan],
                        "max": [g1.max() if len(g1) else np.nan, g2.max() if len(g2) else np.nan],
                    })
                    st.markdown("#### Szybkie statystyki (porównanie)")
                    st.dataframe(comp_stats, hide_index=True, use_container_width=True)

    # =====================================================
    # INTERAKTYWNE WYKRESY 3D (PLOTLY)
    # =====================================================
    st.markdown("## 🧊 Interaktywne wykresy 3D – zależności między cechami")
    st.caption("Możesz obracać, przybliżać i eksplorować punkty myszką.")

    if filtered.empty:
        st.warning("Brak danych po filtrach – nie mogę narysować wykresu 3D.")
    else:
        numeric_features = [c for c in df.columns if c != "quality"]

        c3d_1, c3d_2, c3d_3, c3d_4 = st.columns(4)

        with c3d_1:
            x_3d = st.selectbox(
                "Oś X",
                numeric_features,
                index=numeric_features.index("alcohol") if "alcohol" in numeric_features else 0,
                key="px3d_x"
            )
        with c3d_2:
            y_3d = st.selectbox(
                "Oś Y",
                numeric_features,
                index=numeric_features.index("volatile acidity") if "volatile acidity" in numeric_features else 1,
                key="px3d_y"
            )
        with c3d_3:
            z_3d = st.selectbox(
                "Oś Z",
                numeric_features,
                index=numeric_features.index("sulphates") if "sulphates" in numeric_features else 2,
                key="px3d_z"
            )
        with c3d_4:
            color_3d = st.selectbox(
                "Kolorowanie punktów",
                ["quality"] + numeric_features,
                key="px3d_color"
            )

        plot_df = filtered[[x_3d, y_3d, z_3d, color_3d]].dropna()
        if plot_df.empty:
            st.warning("Brak danych do wizualizacji 3D.")
        else:
            fig_3d = px.scatter_3d(
                plot_df,
                x=x_3d,
                y=y_3d,
                z=z_3d,
                color=color_3d,
                opacity=0.7,
                title=f"Interaktywny wykres 3D: {x_3d} vs {y_3d} vs {z_3d}",
            )
            fig_3d.update_layout(height=650, margin=dict(l=0, r=0, b=0, t=50))
            st.plotly_chart(fig_3d, use_container_width=True)

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

    x_feature = st.selectbox(
        "Wybierz cechę (oś X):",
        feature_cols,
        index=feature_cols.index(default_feature),
        key="scatter_feature"
    )

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

    if submitted:
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
            f"Komunikat błędu:\n`{pairings_error}`"
        )
        st.stop()

    dfp = pairings_df.copy()

    st.markdown("### Podgląd danych")
    st.dataframe(dfp.head(), use_container_width=True)

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

    st.markdown("### Filtrowanie rekomendacji")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        wine_type_sel = st.multiselect(
            "Typ wina (`wine_type`):",
            options=sorted(dfp["wine_type"].dropna().unique()),
            default=[]
        )
    with col_f2:
        food_cat_sel = st.multiselect(
            "Kategoria jedzenia (`food_category`):",
            options=sorted(dfp["food_category"].dropna().unique()),
            default=[]
        )
    with col_f3:
        cuisine_sel = st.multiselect(
            "Kuchnia (`cuisine`):",
            options=sorted(dfp["cuisine"].dropna().unique()),
            default=[]
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
    st.write(f"✅ Rekordy po filtrach: **{filt.shape[0]}** / {dfp.shape[0]}")

    if "pairing_quality" in filt.columns:
        stats_df = pd.DataFrame({
            "metryka": ["średnia", "mediana", "min", "max"],
            "pairing_quality": [
                filt["pairing_quality"].mean(),
                filt["pairing_quality"].median(),
                filt["pairing_quality"].min(),
                filt["pairing_quality"].max(),
            ]
        })
        st.markdown("### 📊 Szybkie statystyki (pairing_quality)")
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

        if "wine_type" in filt.columns and not filt.empty:
            st.markdown("### 🍷 Top wine_type wg średniej pairing_quality (po filtrach)")
            mean_by_wine = (
                filt.groupby("wine_type")["pairing_quality"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
                .rename(columns={"pairing_quality": "średnia_pairing_quality"})
            )
            st.dataframe(mean_by_wine, hide_index=True, use_container_width=True)

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

# =========================================================
# 3. NAJLEPSZE DOPASOWANIA (kraj + jedzenie)
# =========================================================
elif module == "Najlepsze dopasowania (kraj + jedzenie)":
    st.subheader("🌍🍽️ Najlepsze dopasowania: kraj + jedzenie")

    if pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`"
        )
        st.stop()

    dfp = pairings_df.copy()

    st.markdown(
        "Wybierz **kraj (kuchnię)** oraz **rodzaj jedzenia** lub **konkretne danie**, "
        "aby zobaczyć najlepiej dopasowane wina."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        cuisine_sel = st.selectbox(
            "🌍 Wybierz kuchnię / kraj:",
            options=["(dowolna)"] + sorted(dfp["cuisine"].dropna().unique().tolist()),
            key="best_cuisine"
        )

    with col_b:
        food_cat_sel = st.selectbox(
            "🍽️ Wybierz kategorię jedzenia:",
            options=["(dowolna)"] + sorted(dfp["food_category"].dropna().unique().tolist()),
            key="best_food_cat"
        )

    with col_c:
        if "pairing_quality" in dfp.columns:
            pq_min = int(np.nanmin(dfp["pairing_quality"]))
            pq_max = int(np.nanmax(dfp["pairing_quality"]))
        else:
            pq_min, pq_max = 1, 5

        min_pair_quality = st.slider(
            "⭐ Minimalna jakość parowania:",
            min_value=pq_min,
            max_value=pq_max,
            value=pq_min,
            step=1,
            key="best_min_pq"
        )

    st.divider()

    food_text = st.text_input(
        "🔎 (Opcjonalnie) Wpisz nazwę dania:",
        placeholder="np. pizza, steak, pasta, salmon...",
        key="best_food_text"
    )

    filt = dfp.copy()

    if cuisine_sel != "(dowolna)" and "cuisine" in filt.columns:
        filt = filt[filt["cuisine"] == cuisine_sel]

    if food_cat_sel != "(dowolna)" and "food_category" in filt.columns:
        filt = filt[filt["food_category"] == food_cat_sel]

    if food_text.strip() and "food_item" in filt.columns:
        filt = filt[filt["food_item"].str.contains(food_text, case=False, na=False)]

    if "pairing_quality" in filt.columns:
        filt = filt[filt["pairing_quality"].fillna(-999) >= min_pair_quality]

    st.markdown(f"### ✅ Znaleziono **{filt.shape[0]}** dopasowań")

    if filt.empty:
        st.warning("Brak dopasowań dla wybranych kryteriów.")
        st.stop()

    top_results = filt.sort_values(by="pairing_quality", ascending=False).head(20) if "pairing_quality" in filt.columns else filt.head(20)

    cols_to_show = [c for c in [
        "food_item",
        "food_category",
        "cuisine",
        "wine_type",
        "wine_category",
        "pairing_quality",
        "quality_label",
        "description"
    ] if c in top_results.columns]

    st.dataframe(
        top_results[cols_to_show],
        use_container_width=True
    )

    st.markdown("### 📊 Szybkie wnioski")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        if "wine_type" in top_results.columns:
            st.write("**Top typy win (liczba rekomendacji):**")
            st.bar_chart(top_results["wine_type"].value_counts())

    with col_s2:
        if "wine_type" in top_results.columns and "pairing_quality" in top_results.columns:
            st.write("**Średnia jakość parowania wg typu wina:**")
            mean_by_wine = (
                top_results.groupby("wine_type")["pairing_quality"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(mean_by_wine)
