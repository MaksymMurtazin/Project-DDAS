"""Streamlit-інтерфейс для сервісу оцінки нерухомості.

Підтримує:
- Окрему оцінку з інтерактивною візуалізацією та історією
- What-If аналіз чутливості ціни
- Масове оцінювання через завантаження CSV
- Аналітику якості моделі
"""
from __future__ import annotations

import io
import os
import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:5000").rstrip("/")
CURRENCY = os.getenv("CURRENCY", "$")
REQUEST_TIMEOUT = 15

st.set_page_config(
    page_title="Аналітичний сервіс оцінки нерухомості",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Словники для красивого відображення англомовних ознак та значень
FEATURE_LABELS = {
    "area": "Площа (кв. фути)",
    "bedrooms": "Спальні",
    "bathrooms": "Ванні кімнати",
    "stories": "Поверховість",
    "parking": "Паркомісця",
    "mainroad": "Прилеглість до головної дороги",
    "guestroom": "Наявність гостьової кімнати",
    "basement": "Підвал / Цокольний поверх",
    "hotwaterheating": "Автономне підігрівання води",
    "airconditioning": "Система кондиціювання",
    "prefarea": "Престижний район (Pref Area)",
    "furnishingstatus": "Стан меблювання",
}

FEATURE_HELP = {
    "area": "Загальна площа об'єкта нерухомості в квадратних футах",
    "bedrooms": "Кількість спальних кімнат",
    "bathrooms": "Кількість ванних кімнат/санвузлів",
    "stories": "Кількість поверхів у будинку",
    "parking": "Кількість критих або підземних паркомісць",
    "mainroad": "Чи під'єднаний будинок безпосередньо до головної дороги",
    "guestroom": "Наявність окремої кімнати для гостей",
    "basement": "Наявність підвального або цокольного приміщення",
    "hotwaterheating": "Наявність централізованого або бойлерного гарячого водопостачання",
    "airconditioning": "Наявність встановлених кондиціонерів",
    "prefarea": "Розташування у пріоритетному/елітному районі міста",
    "furnishingstatus": "Рівень укомплектованості меблями",
}

VALUE_LABELS = {
    "yes": "Так",
    "no": "Ні",
    "furnished": "З меблями (Furnished)",
    "semi-furnished": "Частково (Semi-furnished)",
    "unfurnished": "Без меблів (Unfurnished)",
}

REVERSE_VALUE_LABELS = {v: k for k, v in VALUE_LABELS.items()}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_info(api_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_url}/model/info", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def money(value: float) -> str:
    return f"{CURRENCY}{value:,.0f}"


def translate_warning(warning_text: str) -> str:
    """Перекладає англомовні попередження від API та замінює технічні назви колонок на українські."""
    # Шаблон для виходу за межі діапазону: 'feature'=val is outside the training range [min, max]...
    pattern = r"'(\w+)'=([\d\.]+) is outside the training range \[([\d\.]+), ([\d\.]+)\]; the estimate may be unreliable"
    match = re.match(pattern, warning_text)
    
    if match:
        feature_raw, val, min_val, max_val = match.groups()
        # Беремо перекладену назву з нашого словника FEATURE_LABELS
        feature_ukr = FEATURE_LABELS.get(feature_raw, feature_raw)
        
        # Красиво форматуємо числа (цілі без дробової частини)
        val_str = f"{float(val):.0f}" if float(val).is_integer() else val
        min_str = f"{float(min_val):.0f}" if float(min_val).is_integer() else min_val
        max_str = f"{float(max_val):.0f}" if float(max_val).is_integer() else max_val
        
        return (
            f"Параметр **«{feature_ukr}»** ({val_str}) виходить за межі навчальної вибірки **[{min_str}, {max_str}]**."
        )
    
    return warning_text


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def render_price_range_chart(predicted: float, min_price: float, max_price: float) -> go.Figure:
    """Створює графічний діапазон оцінки вартості."""
    fig = go.Figure()

    # Інтервальний відрізок
    fig.add_trace(
        go.Scatter(
            x=[min_price, max_price],
            y=["Оцінка", "Оцінка"],
            mode="lines",
            line=dict(color="#1f77b4", width=8),
            name="90% Інтервал",
            hoverinfo="skip",
        )
    )

    # Точка очікуваної вартості
    fig.add_trace(
        go.Scatter(
            x=[predicted],
            y=["Оцінка"],
            mode="markers+text",
            marker=dict(color="#2ca02c", size=20, symbol="diamond"),
            text=[f"Прогноз: {money(predicted)}"],
            textposition="top center",
            name="Очікувана вартість",
            hovertemplate="Прогноз: %{x:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="Вартість", tickprefix=CURRENCY, showgrid=True),
        yaxis=dict(showticklabels=False),
        showlegend=False,
    )
    return fig


def render_input_form(schema: dict[str, Any], key_prefix: str = "default") -> dict[str, Any]:
    """Формує структуровану форму з унікальними ключами для кожної вкладки."""
    values: dict[str, Any] = {}

    numeric_keys = list(schema["numeric"].keys())
    categorical_keys = list(schema["categorical"].keys())

    st.markdown("##### 📐 Основні параметри будинку")
    col1, col2, col3 = st.columns(3)

    for idx, name in enumerate(numeric_keys):
        stats = schema["numeric"][name]
        label = FEATURE_LABELS.get(name, name)
        help_text = FEATURE_HELP.get(name, "")
        target_col = [col1, col2, col3][idx % 3]

        with target_col:
            if stats["is_integer"]:
                values[name] = st.number_input(
                    label,
                    min_value=int(stats["min"]),
                    max_value=int(stats["max"] * 2),
                    value=int(stats["median"]),
                    step=1,
                    help=help_text,
                    key=f"{key_prefix}_num_{name}",
                )
            else:
                values[name] = st.number_input(
                    label,
                    min_value=float(stats["min"]),
                    value=float(stats["median"]),
                    help=help_text,
                    key=f"{key_prefix}_num_{name}",
                )

    st.markdown("---")
    st.markdown("##### ✨ Зручності, інфраструктура та стан")
    col4, col5 = st.columns(2)

    for idx, name in enumerate(categorical_keys):
        options = schema["categorical"][name]
        label = FEATURE_LABELS.get(name, name)
        help_text = FEATURE_HELP.get(name, "")
        target_col = col4 if idx % 2 == 0 else col5

        formatted_options = [VALUE_LABELS.get(opt, opt) for opt in options]

        with target_col:
            selected_display = st.selectbox(
                label,
                options=formatted_options,
                index=0,
                help=help_text,
                key=f"{key_prefix}_cat_{name}",
            )
            values[name] = REVERSE_VALUE_LABELS.get(selected_display, selected_display)

    return values


def request_prediction(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(f"{API_URL}/predict/price", json=payload, timeout=REQUEST_TIMEOUT)
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Помилка з'єднання з API: {exc}")
        return None

    if not response.ok:
        st.error("API відхилив запит з помилкою:")
        for detail in body.get("details", [body.get("message", "Невідома помилка")]):
            st.write(f"• {detail}")
        return None

    return body


def render_single_prediction_tab(schema: dict[str, Any]) -> None:
    col_form, col_res = st.columns([1.1, 0.9], gap="large")

    with col_form:
        payload = render_input_form(schema, key_prefix="eval")
        btn = st.button("📊 Розрахувати вартість", type="primary", use_container_width=True)

    with col_res:
        st.markdown("##### 🎯 Результат оцінки")
        if btn:
            with st.spinner("Обчислення..."):
                res = request_prediction(payload)
                if res:
                    price = res["predicted_price"]
                    p_min = res["price_range"]["min"]
                    p_max = res["price_range"]["max"]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Очікувана вартість", money(price))
                    c2.metric("Мінімум (90%)", money(p_min))
                    c3.metric("Максимум (90%)", money(p_max))

                    st.plotly_chart(
                        render_price_range_chart(price, p_min, p_max),
                        use_container_width=True,
                    )

                    st.caption(
                        f"Версія моделі: `{res['model_version']}`. "
                        f"Інтервал обчислено на основі емпіричної похибки."
                    )

                    warnings = res.get("warnings", [])
                    if warnings:
                        translated_warnings = [translate_warning(w) for w in warnings]
                        with st.expander("⚠️ Увага: деякі параметри виходять за межі норми", expanded=True):
                            st.write("Введені значення перевищують діапазон даних, на яких навчалася модель. Це може знизити точність прогнозу:")
                            for tw in translated_warnings:
                                st.markdown(f"• {tw}")

                    # Додаємо запис в історію
                    record = {
                        "Площа": payload.get("area"),
                        "Спальні": payload.get("bedrooms"),
                        "Ванні": payload.get("bathrooms"),
                        "Прогноз": money(price),
                        "Мін": money(p_min),
                        "Макс": money(p_max),
                    }
                    st.session_state.history.append(record)

        elif st.session_state.history:
            st.info("Змініть параметри та натисніть «Розрахувати вартість».")
        else:
            st.info("Заповніть форму ліворуч для розрахунку.")

        if st.session_state.history:
            st.markdown("---")
            st.markdown("##### 📜 Історія останніх оцінок")
            hist_df = pd.DataFrame(st.session_state.history[::-1])
            st.dataframe(hist_df, use_container_width=True)
            if st.button("Очистити історію"):
                st.session_state.history = []
                st.rerun()


def render_what_if_tab(schema: dict[str, Any]) -> None:
    st.markdown("#### 📈 Аналіз чутливості (What-If Analysis)")
    st.write("Подивитися, як змінюється прогнозована ціна залежно від зміни **площі будинку** при збереженні інших параметрів.")

    base_payload = render_input_form(schema, key_prefix="whatif")

    area_stats = schema["numeric"]["area"]
    min_area = int(area_stats["min"])
    max_area = int(area_stats["max"])

    step_area = st.slider("Крок зміни площі", 100, 1000, 200, key="whatif_slider_step")

    if st.button("Створити симуляцію", type="secondary", key="whatif_btn_submit"):
        areas = list(range(min_area, max_area + 1, step_area))
        batch_payload = []
        for a in areas:
            p = base_payload.copy()
            p["area"] = a
            batch_payload.append(p)

        with st.spinner("Симуляція графіку..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict/price/batch",
                    json={"instances": batch_payload},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.ok:
                    results = resp.json()["predictions"]
                    prices = [r["predicted_price"] for r in results]

                    sim_df = pd.DataFrame({"Площа": areas, "Прогнозована ціна": prices})

                    fig = px.line(
                        sim_df,
                        x="Площа",
                        y="Прогнозована ціна",
                        title="Динаміка ціни залежно від площі",
                        markers=True,
                    )
                    fig.update_yaxes(tickprefix=CURRENCY)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Помилка під час пакетної симуляції.")
            except Exception as e:
                st.error(f"Не вдалося виконати What-If запит: {e}")


def render_batch_tab() -> None:
    st.markdown("#### 📁 Масове оцінювання з CSV-файлу")
    st.write("Завантажте CSV файл, який містить необхідні колонки, і отримайте прогнози для всіх об'єктів одразу.")

    uploaded_file = st.file_uploader("Оберіть CSV файл", type=["csv"], key="batch_uploader")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Попередній перегляд завантажених даних:", df.head(3))

            if st.button("Опрацювати всі записи", key="batch_btn_process"):
                instances = df.to_dict(orient="records")
                with st.spinner(f"Запитуємо API для {len(instances)} записів..."):
                    resp = requests.post(
                        f"{API_URL}/predict/price/batch",
                        json={"instances": instances},
                        timeout=REQUEST_TIMEOUT * 2,
                    )
                    if resp.ok:
                        preds = resp.json()["predictions"]
                        res_df = df.copy()
                        res_df["predicted_price"] = [p["predicted_price"] for p in preds]
                        res_df["price_min_90"] = [p["price_range"]["min"] for p in preds]
                        res_df["price_max_90"] = [p["price_range"]["max"] for p in preds]

                        st.success("Обчислення завершено!")
                        st.dataframe(res_df.head(10))

                        # Завантаження CSV
                        csv_buffer = io.StringIO()
                        res_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Завантажити результати в CSV",
                            data=csv_buffer.getvalue(),
                            file_name="predictions_result.csv",
                            mime="text/csv",
                            key="batch_btn_download",
                        )
                    else:
                        st.error(f"Помилка API: {resp.text}")
        except Exception as e:
            st.error(f"Не вдалося зчитати файл: {e}")


def render_model_quality(info: dict[str, Any]) -> None:
    test = info["metrics"]["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE (Похибка)", f"{test['mape']:.1%}")
    c2.metric("R² (Точність)", f"{test['r2']:.3f}")
    c3.metric("В межах ±5%", f"{test['within_5pct']:.0%}")
    c4.metric("В межах ±7%", f"{test['within_7pct']:.0%}")

    st.markdown("---")
    st.subheader("📊 Порівняльний аналіз моделей")
    comparison = pd.DataFrame(
        {name: {**res["test"], "cv_rmse": res["cv"]["rmse"]} for name, res in info["model_comparison"].items()}
    ).T[["cv_rmse", "rmse", "mae", "mape", "r2", "within_5pct", "within_7pct"]]

    st.dataframe(comparison.style.highlight_min(axis=0, subset=["mape", "rmse"]).format("{:.3f}"), use_container_width=True)

    importance = pd.DataFrame(info.get("feature_importance") or [])
    if not importance.empty:
        st.subheader("🎯 Важливість ознак (Permutation Importance)")
        importance["feature_translated"] = importance["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
        fig = px.bar(
            importance.iloc[::-1],
            x="importance",
            y="feature_translated",
            orientation="h",
            labels={"importance": "Важливість", "feature_translated": "Ознака"},
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    init_session_state()

    st.title("🏠 Система аналізу та оцінки нерухомості")

    try:
        info = fetch_model_info(API_URL)
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Не вдалося з'єднатися з API ({API_URL}): {exc}")
        st.info("Переконайтеся, що натреновано модель (`python -m src.train`) та запущено Flask API (`python -m api.app`).")
        st.stop()

    tab_eval, tab_whatif, tab_batch, tab_quality = st.tabs(
        ["🏷️ Оцінка об'єкта", "📈 What-If Аналіз", "📁 Пакетний розрахунок (CSV)", "📊 Якість моделі"]
    )

    with tab_eval:
        render_single_prediction_tab(info["schema"])

    with tab_whatif:
        render_what_if_tab(info["schema"])

    with tab_batch:
        render_batch_tab()

    with tab_quality:
        render_model_quality(info)

    st.sidebar.markdown("### ℹ️ Інформація про систему")
    st.sidebar.text(f"API: {API_URL}")
    st.sidebar.text(f"Модель: {info['model_name']}")
    st.sidebar.text(f"Версія: {info['model_version']}")


if __name__ == "__main__":
    main()
