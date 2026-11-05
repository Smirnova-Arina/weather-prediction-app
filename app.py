import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime

st.set_page_config(
    page_title="Прогноз осадков",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Небольшая стилизация ---
st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.5rem;}
    .result-card {
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }
    .card-dry  {background: linear-gradient(135deg, #43a047, #66bb6a);}
    .card-rain {background: linear-gradient(135deg, #1e88e5, #42a5f5);}
    .card-snow {background: linear-gradient(135deg, #546e7a, #90a4ae);}
    .result-card h2 {margin: 0; font-size: 2.5rem;}
    .result-card p {margin: 0.3rem 0 0 0; font-size: 1.1rem; opacity: 0.95;}
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #e3f2fd;
        color: #1565c0;
        font-weight: 600;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('automl_model_multiclass.pkl', 'rb') as f:
        return pickle.load(f)

def fetch_recent_data(lat, lon, today, past_days=3):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": 0,
        "daily": "precipitation_sum,rain_sum,snowfall_sum",
        "timezone": "Europe/Moscow"
    }
    response = requests.get(url, params=params, timeout=15)
    data = response.json()

    if data.get("error"):
        reason = data.get("reason", "неизвестная ошибка")
        raise RuntimeError(f"Сервис Open-Meteo временно недоступен: {reason}")

    if "daily" not in data:
        raise RuntimeError("Open-Meteo не вернул данные об осадках")

    df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'prcp': data['daily']['precipitation_sum'],
        'rain_sum': data['daily']['rain_sum'],
        'snow_sum': data['daily']['snowfall_sum']
    })
    
    df[['prcp', 'rain_sum', 'snow_sum']] = df[['prcp', 'rain_sum', 'snow_sum']].fillna(0)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Оставляем только даты строго меньше сегодняшней
    df = df[df['date'] < pd.Timestamp(today.date())].reset_index(drop=True)
    
    return df

def get_features_for_date(target_date, lat, lon):
    df = fetch_recent_data(lat, lon, today=target_date, past_days=3)

    # Восстанавливаем мультиклассовый таргет для создания лагов
    conditions_target = [
        (df['snow_sum'] > 0),
        (df['rain_sum'] > 0) & (df['snow_sum'] == 0)
    ]
    df['target'] = np.select(conditions_target, [2, 1], default=0)

    # Категории осадков
    df['prcp_category'] = np.select(
        [df['prcp'] == 0, (df['prcp'] > 0) & (df['prcp'] <= 2.5), (df['prcp'] > 2.5) & (df['prcp'] <= 10.0), df['prcp'] > 10.0],
        ['нет осадков', 'мало осадков', 'средние осадки', 'большое кол-во осадков'], default='нет осадков'
    )

    df['target_lag_1'] = df['target'].shift(1)
    df['target_lag_2'] = df['target'].shift(2)
    df['target_lag_3'] = df['target'].shift(3)
    
    df['prcp_lag_1'] = df['prcp'].shift(1)
    df['prcp_cat_lag_1'] = df['prcp_category'].shift(1)
    df['prcp_cat_lag_2'] = df['prcp_category'].shift(2)
    df['prcp_cat_lag_3'] = df['prcp_category'].shift(3)

    last_row = df.iloc[-1]
    input_data = pd.DataFrame({
        'date': [target_date],
        'month': [target_date.month],
        'day': [target_date.day],
        'dayofweek': [target_date.weekday()],
        'target_lag_1': [last_row['target_lag_1']],
        'target_lag_2': [last_row['target_lag_2']],
        'target_lag_3': [last_row['target_lag_3']],
        'prcp_lag_1': [last_row['prcp_lag_1']],
        'prcp_cat_lag_1': [last_row['prcp_cat_lag_1']],
        'prcp_cat_lag_2': [last_row['prcp_cat_lag_2']],
        'prcp_cat_lag_3': [last_row['prcp_cat_lag_3']]
    })
    
    for col in ['prcp_cat_lag_1', 'prcp_cat_lag_2', 'prcp_cat_lag_3']:
        input_data[col] = input_data[col].astype('category')
        
    return input_data, df

cities = {
    "Москва": (55.536757, 37.161013),
    "Санкт-Петербург": (59.934280, 30.335099),
    "Новосибирск": (55.008353, 82.935733),
    "Екатеринбург": (56.838011, 60.597465),
    "Казань": (55.796127, 49.106405),
    "Нижний Новгород": (56.326887, 44.005986),
    "Челябинск": (55.164442, 61.436843),
    "Самара": (53.195878, 50.100202),
    "Омск": (54.988480, 73.324236),
    "Ростов-на-Дону": (47.222078, 39.720358)
}

st.title("🌧️ Прогноз осадков на сегодня")
st.caption("Мультиклассовая модель машинного обучения (Сухо / Дождь / Снег)")

st.sidebar.header("Параметры прогноза")
mode = st.sidebar.radio("Способ выбора места", ["Город из списка", "Свои координаты"])

if mode == "Город из списка":
    selected_city = st.sidebar.selectbox("Город", list(cities.keys()))
    lat, lon = cities[selected_city]
else:
    lat = st.sidebar.number_input("Широта", value=55.536757, format="%.6f")
    lon = st.sidebar.number_input("Долгота", value=37.161013, format="%.6f")
    selected_city = f"({lat:.4f}, {lon:.4f})"

today = datetime.now()

st.markdown(
    f"<span class='badge'>📅 Дата прогноза: {today.strftime('%d.%m.%Y')}</span> "
    f"<span class='badge'>📍 {selected_city}</span>",
    unsafe_allow_html=True
)
st.write("")

predict_clicked = st.button("🔮 Сделать прогноз", width="stretch", type="primary")

if predict_clicked:
    with st.spinner("Загрузка данных и расчёт прогноза..."):
        try:
            automl = load_model()
            input_data, history_df = get_features_for_date(today, lat, lon)
            
            # Предсказание (получаем вероятности для 3 классов)
            pred = automl.predict(input_data)
            probs = pred.data[0] 
            
            # Суммарная вероятность любых осадков (дождь + снег)
            prob_any_precip = probs[1] + probs[2]
            
            # Порог чувствительности (0.4 = 40%)
            THRESHOLD = 0.4 
            
            if prob_any_precip >= THRESHOLD:
                # Если осадки вероятны, смотрим, чего именно больше
                if probs[2] > probs[1]:
                    card_class, icon, title = "card-snow", "❄️", "Ожидается снег"
                else:
                    card_class, icon, title = "card-rain", "🌧️", "Ожидается дождь"
            else:
                card_class, icon, title = "card-dry", "☀️", "Осадков не ожидается"

            # Карточка результата
            st.markdown(
                f"<div class='result-card {card_class}'>"
                f"<h2>{icon} {title}</h2>"
                f"<p>Вероятность любых осадков: <b>{prob_any_precip:.1%}</b></p>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Блок с детализацией вероятностей
            st.write("**Детализация вероятностей:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("Сухо")
                st.progress(float(probs[0]), text=f"{probs[0]:.1%}")
            with col2:
                st.caption("Дождь")
                st.progress(float(probs[1]), text=f"{probs[1]:.1%}")
            with col3:
                st.caption("Снег")
                st.progress(float(probs[2]), text=f"{probs[2]:.1%}")

            st.markdown("---")
            st.subheader("📊 Фактические данные за прошлые дни")
            st.caption("На основе этих значений модель строит лаговые признаки.")
            
            # Подготовка таблицы
            display_df = history_df[['date', 'prcp', 'rain_sum', 'snow_sum', 'target']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%d.%m.%Y')
            display_df['target'] = display_df['target'].map({0: 'Сухо', 1: 'Дождь', 2: 'Снег'})
            
            display_df = display_df.rename(columns={
                'date': 'Дата', 
                'prcp': 'Всего (мм)', 
                'rain_sum': 'Дождь (мм)',
                'snow_sum': 'Снег (см)',
                'target': 'Факт'
            })
            
            st.dataframe(display_df, width="stretch", hide_index=True)

        except FileNotFoundError:
            st.error("Файл модели 'automl_model_multiclass.pkl' не найден. Запустите ноутбук и сохраните модель.")
        except RuntimeError as e:
            st.warning(str(e))
            st.info("Попробуйте обновить страницу через 1–2 минуты.")
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")