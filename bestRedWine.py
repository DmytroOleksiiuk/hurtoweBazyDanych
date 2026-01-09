import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Wine Data App", page_icon="🍷", layout="wide")

PAIRINGS_PATH = "wine_food_pairings.csv"
QUALITY_PATH = "winequality-red.csv"


@st.cache_data(show_spinner=False)
def load_pairings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # lekkie porządki
    for col in ["wine_type", "wine_category", "food_item", "food_category", "cuisine", "quality_label", "description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "pairing_quality" in df.columns:
        df["pairing_quality"] = pd.to_numeric(df["pairing_quality"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_wine_quality(path: str) -> pd.DataFrame:
    # ten dataset zwykle ma separator ';'
    df = pd.read_csv(path, sep=";")
    return df


@st.cache_resource(show_spinner=False)
def train_quality_model(df: pd.DataFrame):
    target = "quality"
    X = df.drop(columns=[target])
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=350,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
        "y_test": y_test,
        "preds": preds,
        "X_cols": list(X.columns),
    }

    # Feature importance z RF (po pipeline: model.named_steps["rf"])
    importances = model.named_steps["rf"].feature_importances_
    fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    return model, metrics, fi


def pairing_page(df: pd.DataFrame):
    st.title("🍽️🍷 Wine–Food Pairings")
    st.caption("Wyszukuj i filtruj parowania (wino–jedzenie) oraz sortuj po jakości dopasowania.")

    # Sidebar filtry
    st.sidebar.header("Filtry (pairings)")

    wine_category = st.sidebar.multiselect(
        "Kategoria wina",
        options=sorted(df["wine_category"].dropna().unique().tolist()),
        default=[],
    )
    wine_type = st.sidebar.multiselect(
        "Typ wina",
        options=sorted(df["wine_type"].dropna().unique().tolist()),
        default=[],
    )
    food_category = st.sidebar.multiselect(
        "Kategoria jedzenia",
        options=sorted(df["food_category"].dropna().unique().tolist()),
        default=[],
    )
    cuisine = st.sidebar.multiselect(
        "Kuchnia",
        options=sorted(df["cuisine"].dropna().unique().tolist()),
        default=[],
    )
    min_quality = st.sidebar.slider(
        "Minimalna jakość parowania (pairing_quality)",
        min_value=int(np.nanmin(df["pairing_quality"])) if "pairing_quality" in df else 1,
        max_value=int(np.nanmax(df["pairing_quality"])) if "pairing_quality" in df else 5,
        value=3,
    )

    query_food = st.sidebar.text_input("Szukaj po nazwie jedzenia (food_item)", value="").strip()
    query_wine = st.sidebar.text_input("Szukaj po nazwie wina (wine_type)", value="").strip()

    filt = df.copy()

    if wine_category:
        filt = filt[filt["wine_category"].isin(wine_category)]
    if wine_type:
        filt = filt[filt["wine_type"].isin(wine_type)]
    if food_category:
        filt = filt[filt["food_category"].isin(food_category)]
    if cuisine:
        filt = filt[filt["cuisine"].isin(cuisine)]
    if "pairing_quality" in filt.columns:
        filt = filt[filt["pairing_quality"].fillna(-999) >= min_quality]

    if query_food:
        filt = filt[filt["food_item"].str.contains(query_food, case=False, na=False)]
    if query_wine:
        filt = filt[filt["wine_type"].str.contains(query_wine, case=False, na=False)]

    # sort
    sort_by = st.selectbox(
        "Sortuj po",
        options=["pairing_quality (desc)", "pairing_quality (asc)", "wine_type", "food_item", "cuisine"],
        index=0,
    )
    if "pairing_quality" in filt.columns:
        if sort_by == "pairing_quality (desc)":
            filt = filt.sort_values("pairing_quality", ascending=False)
        elif sort_by == "pairing_quality (asc)":
            filt = filt.sort_values("pairing_quality", ascending=True)
    if sort_by == "wine_type":
        filt = filt.sort_values("wine_type")
    if sort_by == "food_item":
        filt = filt.sort_values("food_item")
    if sort_by == "cuisine":
        filt = filt.sort_values("cuisine")

    c1, c2, c3 = st.columns(3)
    c1.metric("Wszystkie rekordy", len(df))
    c2.metric("Po filtrach", len(filt))
    if "pairing_quality" in df.columns:
        c3.metric("Śr. pairing_quality (po filtrach)", f"{filt['pairing_quality'].mean():.2f}" if len(filt) else "—")

    st.subheader("Wyniki")
    st.dataframe(
        filt,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Szybka rekomendacja: wybierz jedzenie → pokaż top wina")
    food_pick = st.selectbox(
        "Wybierz food_item",
        options=sorted(df["food_item"].dropna().unique().tolist()),
        index=0,
    )
    top_n = st.slider("Ile rekomendacji", 3, 20, 8)

    rec = df[df["food_item"] == food_pick].copy()
    if "pairing_quality" in rec.columns:
        rec = rec.sort_values("pairing_quality", ascending=False)
    rec = rec.head(top_n)

    st.dataframe(rec, use_container_width=True, hide_index=True)


def quality_page(df: pd.DataFrame):
    st.title("🧪🍷 Wine Quality (ML)")
    st.caption("Model ML (RandomForestRegressor) trenuje się automatycznie i przewiduje jakość na podstawie cech.")

    model, metrics, fi = train_quality_model(df)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Metryki (test split 80/20)")
        st.write(f"**MAE:** {metrics['MAE']:.3f}")
        st.write(f"**R²:** {metrics['R2']:.3f}")

    with c2:
        st.subheader("Najważniejsze cechy (feature importance)")
        st.dataframe(fi.head(12), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Predykcja jakości dla własnych parametrów")

    X_cols = metrics["X_cols"]
    defaults = df[X_cols].median(numeric_only=True)

    # UI do wprowadzania
    input_values = {}
    cols = st.columns(3)
    for i, col in enumerate(X_cols):
        # zakresy na podstawie danych
        mn = float(df[col].min())
        mx = float(df[col].max())
        dv = float(defaults[col]) if col in defaults else float(df[col].mean())

        with cols[i % 3]:
            input_values[col] = st.slider(
                col,
                min_value=mn,
                max_value=mx,
                value=float(np.clip(dv, mn, mx)),
            )

    X_in = pd.DataFrame([input_values], columns=X_cols)
    pred = model.predict(X_in)[0]

    # quality jest całkowite w datasetach (3..8). Pokaż też zaokrąglenie.
    st.success(f"Prognozowana jakość (float): **{pred:.2f}**  |  zaokrąglone: **{int(round(pred))}**")

    st.caption("Uwaga: to baseline model. Jeśli chcesz klasyfikację (np. low/medium/high), da się łatwo przerobić.")


def data_page(pairings: pd.DataFrame, quality: pd.DataFrame):
    st.title("📦 Podgląd danych")
    st.subheader("wine_food_pairings.csv")
    st.write("Kolumny:", list(pairings.columns))
    st.dataframe(pairings.head(25), hide_index=True, use_container_width=True)

    st.subheader("winequality-red.csv")
    st.write("Kolumny:", list(quality.columns))
    st.dataframe(quality.head(25), hide_index=True, use_container_width=True)


def main():
    st.sidebar.title("Nawigacja")
    page = st.sidebar.radio("Wybierz widok", ["Pairings", "Wine Quality (ML)", "Data Preview"])

    # ładowanie danych
    try:
        pairings = load_pairings(PAIRINGS_PATH)
    except FileNotFoundError:
        st.error(f"Brak pliku: {PAIRINGS_PATH}. Umieść CSV w folderze data/.")
        st.stop()

    try:
        quality = load_wine_quality(QUALITY_PATH)
    except FileNotFoundError:
        st.error(f"Brak pliku: {QUALITY_PATH}. Umieść CSV w folderze data/.")
        st.stop()

    if page == "Pairings":
        pairing_page(pairings)
    elif page == "Wine Quality (ML)":
        quality_page(quality)
    else:
        data_page(pairings, quality)


if __name__ == "__main__":
    main()
