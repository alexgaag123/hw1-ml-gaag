from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "name", "year", "km_driven", "fuel", "seller_type", "transmission",
    "owner", "mileage", "engine", "max_power", "torque", "seats",
]
PREDICTION = "predicted_selling_price"


@st.cache_resource
def load_model():
    # Загружаем только доверенный локальный pickle, не пользовательский файл.
    with (ROOT / "model.pickle").open("rb") as source:
        return cloudpickle.load(source)


def show_eda():
    st.header("EDA")
    st.image(str(ROOT / "assets/numeric_histograms.png"), caption="Распределения числовых признаков до обработки выбросов и логарифмирования")
    st.write("У цены и пробега длинные правые хвосты: редкие большие значения сжимают основную часть распределения. В ноутбуке затем ограничили крайний пробег и добавили логарифмы признаков.")
    st.image(str(ROOT / "assets/phik_heatmap.png"), caption="Связи числовых признаков: PhiK")
    st.write("Сильная связь цены с мощностью: PhiK около 0,87; у объёма двигателя и мощности около 0,72. PhiK показывает силу связи, но не её направление и не причинность.")
    st.image(str(ROOT / "assets/pairplot_train.png"), caption="Попарные зависимости числовых признаков на train")



def show_prediction(model):
    st.header("Прогноз по CSV")
    st.write("CSV в UTF-8, разделитель запятая, одна машина в строке. Обязательные столбцы:")
    st.code(", ".join(REQUIRED), language=None)
    st.write("Сохраняйте исходные единицы: 23.4 kmpl, 1248 CC, 74 bhp, 190Nm@ 2000rpm.")
    st.download_button("Скачать пример CSV", (ROOT / "data/example_cars.csv").read_bytes(), "example_cars.csv", "text/csv")
    uploaded = st.file_uploader("CSV с автомобилями", type=["csv"])
    if uploaded is None:
        return

    try:
        frame = pd.read_csv(uploaded)
        if frame.empty:
            st.error("CSV пуст: добавьте хотя бы одну машину.")
            return
        missing = [column for column in REQUIRED if column not in frame.columns]
        if missing:
            st.error("Не хватает столбцов: " + ", ".join(missing))
            return
        if PREDICTION in frame.columns:
            st.error(f"Переименуйте столбец {PREDICTION}: он нужен для результата.")
            return

        predictions = model.predict(frame[REQUIRED])
        if not np.isfinite(predictions).all():
            st.error("Модель вернула неконечные прогнозы. Проверьте значения в CSV.")
            return
    except Exception as error:
        st.error(f"Не удалось обработать CSV: {error}")
        return

    st.subheader("Предпросмотр входных данных")
    st.dataframe(frame.head(20), hide_index=True)
    result = frame.copy()
    result[PREDICTION] = predictions
    st.subheader("Прогнозы")
    st.dataframe(result, hide_index=True)
    st.download_button(
        "Скачать CSV с прогнозами",
        result.to_csv(index=False).encode("utf-8-sig"),
        "cars_predictions.csv",
        "text/csv",
    )



def show_weights(model):
    st.header("Веса модели")
    st.write("20 наибольших по модулю коэффициентов Ridge после стандартизации. Знак показывает направление вклада в прогноз; это не изменение цены за единицу исходного признака и не причинный эффект.")
    inner = model.named_steps["model"]
    weights = pd.Series(
        inner.named_steps["ridge"].coef_,
        index=inner.named_steps["columntransformer"].get_feature_names_out(),
        name="Коэффициент",
    )
    top = weights.loc[weights.abs().nlargest(20).index]
    st.bar_chart(top, horizontal=True, sort=False, height=600)


st.set_page_config(page_title="Стоимость автомобиля", layout="wide")
st.title("Предсказание стоимости автомобиля")
model = load_model()
eda, prediction, weights_tab = st.tabs(["EDA", "Прогноз по CSV", "Веса модели"])

with eda:
    show_eda()

with prediction:
    show_prediction(model)

with weights_tab:
    show_weights(model)
