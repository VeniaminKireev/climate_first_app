# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import aiohttp
import asyncio
from concurrent.futures import ProcessPoolExecutor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================================

st.set_page_config(
    page_title="Climate Analysis Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ФУНКЦИИ ДЛЯ АНАЛИЗА ИСТОРИЧЕСКИХ ДАННЫХ
# ============================================================================

class HistoricalAnalyzer:
    """Класс для анализа исторических температурных данных"""
    
    def __init__(self):
        self.data = None
        self.results = {}
        
    def load_data(self, df):
        """Загрузка исторических данных"""
        self.data = df.copy()
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        return self.data
    
    def analyze_city(self, city_data):
        """Анализ данных для одного города"""
        city_data = city_data.sort_values('timestamp').copy()
        
        # 1. Скользящее среднее за 30 дней
        city_data['rolling_avg_30d'] = city_data['temperature'].rolling(
            window=30, min_periods=1).mean()
        
        # 2. Статистика по сезонам
        seasonal_stats = city_data.groupby('season').agg({
            'temperature': ['mean', 'std', 'count', 'min', 'max', 'median']
        })
        
        seasonal_stats.columns = ['mean', 'std', 'count', 'min', 'max', 'median']
        seasonal_stats = seasonal_stats.reset_index()
        
        # 3. Выявление аномалий (среднее ± 2σ)
        anomalies = []
        season_limits = {}
        
        for _, row in seasonal_stats.iterrows():
            season = row['season']
            mean_temp = row['mean']
            std_temp = row['std']
            
            upper_limit = mean_temp + 2 * std_temp
            lower_limit = mean_temp - 2 * std_temp
            
            season_limits[season] = {
                'mean': mean_temp,
                'std': std_temp,
                'upper': upper_limit,
                'lower': lower_limit
            }
            
            # Находим аномалии для этого сезона
            season_data = city_data[city_data['season'] == season]
            season_anomalies = season_data[
                (season_data['temperature'] > upper_limit) | 
                (season_data['temperature'] < lower_limit)
            ]
            
            for _, anomaly in season_anomalies.iterrows():
                anomalies.append({
                    'timestamp': anomaly['timestamp'],
                    'temperature': anomaly['temperature'],
                    'season': season,
                    'rolling_avg': anomaly['rolling_avg_30d'],
                    'mean_temp': mean_temp,
                    'std_temp': std_temp,
                    'deviation': anomaly['temperature'] - mean_temp,
                    'z_score': (anomaly['temperature'] - mean_temp) / std_temp if std_temp > 0 else 0
                })
        
        anomalies_df = pd.DataFrame(anomalies) if anomalies else pd.DataFrame()
        
        return {
            'city_name': city_data['city'].iloc[0],
            'city_data': city_data,
            'seasonal_stats': seasonal_stats,
            'season_limits': season_limits,
            'anomalies': anomalies_df,
            'num_anomalies': len(anomalies),
            'total_observations': len(city_data)
        }
    
    def analyze_sequential(self):
        """Последовательный анализ всех городов"""
        results = {}
        for city in self.data['city'].unique():
            city_data = self.data[self.data['city'] == city].copy()
            results[city] = self.analyze_city(city_data)
        return results
    
    def analyze_parallel(self, max_workers=4):
        """Параллельный анализ всех городов"""
        cities = self.data['city'].unique()
        results = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for city in cities:
                city_data = self.data[self.data['city'] == city].copy()
                future = executor.submit(self.analyze_city, city_data)
                futures[future] = city
            
            for future in futures:
                result = future.result()
                results[result['city_name']] = result
        
        return results

# ============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С OpenWeatherMap API
# ============================================================================

class WeatherAPI:
    """Класс для работы с OpenWeatherMap API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_current_weather_sync(self, city):
        """Синхронное получение текущей погоды"""
        if not self.api_key:
            return {'success': False, 'error': 'API ключ не указан'}
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'city': data['name'],
                    'country': data['sys']['country'],
                    'temperature': data['main']['temp'],
                    'feels_like': data['main']['feels_like'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'],
                    'icon': data['weather'][0]['icon'],
                    'wind_speed': data['wind']['speed'],
                    'timestamp': datetime.now()
                }
            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': 'Invalid API key',
                    'message': 'Неверный API ключ. Пожалуйста, проверьте ключ.'
                }
            else:
                error_data = response.json()
                return {
                    'success': False,
                    'error': f"API Error {response.status_code}",
                    'message': error_data.get('message', 'Unknown error')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(type(e).__name__),
                'message': f'Ошибка: {str(e)}'
            }
    
    async def get_current_weather_async(self, city, session):
        """Асинхронное получение текущей погоды"""
        if not self.api_key:
            return {'success': False, 'error': 'API ключ не указан'}
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        
        try:
            async with session.get(self.base_url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'success': True,
                        'city': data['name'],
                        'country': data['sys']['country'],
                        'temperature': data['main']['temp'],
                        'feels_like': data['main']['feels_like'],
                        'humidity': data['main']['humidity'],
                        'pressure': data['main']['pressure'],
                        'description': data['weather'][0]['description'],
                        'icon': data['weather'][0]['icon'],
                        'wind_speed': data['wind']['speed'],
                        'timestamp': datetime.now()
                    }
                elif response.status == 401:
                    return {
                        'success': False,
                        'error': 'Invalid API key',
                        'message': 'Неверный API ключ'
                    }
                else:
                    error_data = await response.json()
                    return {
                        'success': False,
                        'error': f"API Error {response.status}",
                        'message': error_data.get('message', 'Unknown error')
                    }
                        
        except Exception as e:
            return {
                'success': False,
                'error': str(type(e).__name__),
                'message': f'Ошибка: {str(e)}'
            }

# ============================================================================
# ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ
# ============================================================================

def create_temperature_time_series(city_data, anomalies_df):
    """Создание временного ряда температур с выделением аномалий"""
    fig = go.Figure()
    
    # Основной временной ряд
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data['temperature'],
        mode='lines',
        name='Температура',
        line=dict(color='blue', width=1),
        opacity=0.7
    ))
    
    # Скользящее среднее
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data['rolling_avg_30d'],
        mode='lines',
        name='Скользящее среднее (30 дней)',
        line=dict(color='green', width=2)
    ))
    
    # Аномалии
    if not anomalies_df.empty:
        fig.add_trace(go.Scatter(
            x=anomalies_df['timestamp'],
            y=anomalies_df['temperature'],
            mode='markers',
            name='Аномалии',
            marker=dict(
                color='red',
                size=8,
                symbol='x',
                line=dict(width=1, color='darkred')
            )
        ))
    
    fig.update_layout(
        title='Временной ряд температур',
        xaxis_title='Дата',
        yaxis_title='Температура (°C)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_seasonal_profile(seasonal_stats):
    """Создание сезонных профилей"""
    fig = go.Figure()
    
    # Порядок сезонов
    season_order = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    seasonal_stats = seasonal_stats.copy()
    seasonal_stats['order'] = seasonal_stats['season'].map(season_order)
    seasonal_stats = seasonal_stats.sort_values('order')
    
    # Средние температуры
    fig.add_trace(go.Bar(
        x=seasonal_stats['season'],
        y=seasonal_stats['mean'],
        name='Средняя температура',
        marker_color='lightblue',
        error_y=dict(
            type='data',
            array=seasonal_stats['std'] * 2,
            color='gray',
            thickness=1.5,
            width=3
        )
    ))
    
    fig.update_layout(
        title='Сезонные профили температур',
        xaxis_title='Сезон',
        yaxis_title='Температура (°C)',
        height=400
    )
    
    return fig

def create_distribution_plot(city_data, current_temp=None):
    """Создание графика распределения температур"""
    fig = go.Figure()
    
    # Гистограмма
    fig.add_trace(go.Histogram(
        x=city_data['temperature'],
        name='Распределение температур',
        nbinsx=50,
        marker_color='skyblue',
        opacity=0.7
    ))
    
    # Линия плотности
    fig.add_trace(go.Scatter(
        x=np.sort(city_data['temperature']),
        y=np.linspace(0, 1, len(city_data)),
        mode='lines',
        name='Функция распределения',
        yaxis='y2',
        line=dict(color='darkblue', width=2)
    ))
    
    # Текущая температура (если указана)
    if current_temp is not None:
        fig.add_vline(
            x=current_temp,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Текущая: {current_temp:.1f}°C",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title='Распределение температур',
        xaxis_title='Температура (°C)',
        yaxis_title='Частота',
        yaxis2=dict(
            title='Вероятность',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        height=400
    )
    
    return fig

def create_box_plot_by_season(city_data):
    """Создание боксплота по сезонам"""
    fig = px.box(
        city_data,
        x='season',
        y='temperature',
        color='season',
        title='Распределение температур по сезонам',
        labels={'season': 'Сезон', 'temperature': 'Температура (°C)'}
    )
    
    fig.update_layout(height=400)
    return fig

# ============================================================================
# ГЕНЕРАЦИЯ ДЕМО-ДАННЫХ (если файл не загружен)
# ============================================================================

def generate_demo_data():
    """Генерация демонстрационных данных"""
    seasonal_temperatures = {
        "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
        "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
        "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
        "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
        "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
        "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
        "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
        "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
        "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
        "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
        "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
        "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
        "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
        "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
        "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
    }
    
    month_to_season = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    }
    
    np.random.seed(42)
    cities = list(seasonal_temperatures.keys())
    dates = pd.date_range(start="2015-01-01", end="2020-12-31", freq="D")
    data = []
    
    for city in cities:
        for date in dates:
            season = month_to_season[date.month]
            mean_temp = seasonal_temperatures[city][season]
            temperature = np.random.normal(loc=mean_temp, scale=5)
            data.append({
                "city": city,
                "timestamp": date,
                "temperature": temperature,
                "season": season
            })
    
    df = pd.DataFrame(data)
    return df

# ============================================================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ STREAMLIT
# ============================================================================

def main():
    """Основная функция Streamlit приложения"""
    
    # Заголовок приложения
    st.title("🌡️ Climate Analysis Dashboard")
    st.markdown("---")
    
    # ========================================================================
    # САЙДБАР: ЗАГРУЗКА ДАННЫХ И НАСТРОЙКИ
    # ========================================================================
    
    with st.sidebar:
        st.header("📂 Загрузка данных")
        
        # Загрузка файла с историческими данными
        uploaded_file = st.file_uploader(
            "Загрузите файл temperature_data.csv",
            type=['csv'],
            help="Если файл не загружен, будут использованы демонстрационные данные"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.success(f"✅ Файл загружен: {uploaded_file.name}")
            st.info(f"Записей: {len(df):,} | Городов: {len(df['city'].unique())}")
        else:
            df = generate_demo_data()
            st.info("ℹ️ Используются демонстрационные данные")
        
        st.markdown("---")
        
        st.header("🔑 Настройки API")
        
        # Ввод API ключа OpenWeatherMap
        api_key = st.text_input(
            "API Key OpenWeatherMap",
            type="password",
            help="Получите бесплатный ключ на https://openweathermap.org/api"
        )
        
        st.markdown("---")
        
        st.header("🎯 Выбор города")
        
        # Выбор города из выпадающего списка
        cities = sorted(df['city'].unique())
        selected_city = st.selectbox(
            "Выберите город для анализа:",
            cities,
            index=cities.index("Moscow") if "Moscow" in cities else 0
        )
        
        st.markdown("---")
        
        # Настройки анализа
        st.header("⚙️ Настройки анализа")
        parallel_analysis = st.checkbox(
            "Использовать параллельный анализ",
            value=True,
            help="Распараллеливание анализа исторических данных"
        )
        
        max_workers = st.slider(
            "Количество процессов",
            min_value=1,
            max_value=8,
            value=4,
            help="Количество процессов для параллельного анализа"
        )
        
        st.markdown("---")
        
        # Информация о приложении
        st.info("""
        **Информация:**
        - Данные: 2015-2020 гг.
        - Ежедневные измерения температуры
        - Аномалии: температура вне среднее ± 2σ
        """)
    
    # ========================================================================
    # ОСНОВНОЙ КОНТЕНТ
    # ========================================================================
    
    # Инициализация анализатора
    analyzer = HistoricalAnalyzer()
    analyzer.load_data(df)
    
    # Вкладки для разных разделов анализа
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Исторический анализ", 
        "🌡️ Текущая погода", 
        "📈 Визуализации",
        "⚡ Производительность"
    ])
    
    with tab1:
        st.header(f"Исторический анализ: {selected_city}")
        
        # Анализ данных для выбранного города
        with st.spinner("Анализ исторических данных..."):
            city_data = df[df['city'] == selected_city].copy()
            result = analyzer.analyze_city(city_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего наблюдений", f"{result['total_observations']:,}")
            
            with col2:
                st.metric("Аномалий обнаружено", result['num_anomalies'])
            
            with col3:
                anomaly_percent = (result['num_anomalies'] / result['total_observations']) * 100
                st.metric("Процент аномалий", f"{anomaly_percent:.2f}%")
            
            with col4:
                mean_temp = result['city_data']['temperature'].mean()
                st.metric("Средняя температура", f"{mean_temp:.1f}°C")
        
        st.subheader("📈 Описательная статистика")
        
        # Основная статистика
        stats_df = result['city_data']['temperature'].describe().reset_index()
        stats_df.columns = ['Метрика', 'Значение']
        stats_df['Значение'] = stats_df['Значение'].round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            seasonal_stats_display = result['seasonal_stats'][['season', 'mean', 'std', 'count', 'min', 'max']].copy()
            seasonal_stats_display.columns = ['Сезон', 'Среднее', 'Стд. отклонение', 'Количество', 'Минимум', 'Максимум']
            seasonal_stats_display = seasonal_stats_display.round(2)
            st.dataframe(seasonal_stats_display, use_container_width=True, hide_index=True)
        
        st.subheader("🔍 Детали аномалий")
        
        if result['num_anomalies'] > 0:
            anomalies_display = result['anomalies'][['timestamp', 'temperature', 'season', 'deviation', 'z_score']].copy()
            anomalies_display.columns = ['Дата', 'Температура', 'Сезон', 'Отклонение', 'Z-score']
            anomalies_display['Дата'] = anomalies_display['Дата'].dt.date
            anomalies_display['Температура'] = anomalies_display['Температура'].round(1)
            anomalies_display['Отклонение'] = anomalies_display['Отклонение'].round(1)
            anomalies_display['Z-score'] = anomalies_display['Z-score'].round(2)
            
            # Сортировка по абсолютному значению Z-score
            anomalies_display['abs_z'] = np.abs(anomalies_display['Z-score'])
            anomalies_display = anomalies_display.sort_values('abs_z', ascending=False).drop('abs_z', axis=1)
            
            st.dataframe(anomalies_display.head(20), use_container_width=True)
            
            if result['num_anomalies'] > 20:
                st.caption(f"Показаны 20 наиболее значимых аномалий из {result['num_anomalies']}")
        else:
            st.success("✅ Аномалий не обнаружено")
    
    with tab2:
        st.header(f"Текущая погода: {selected_city}")
        
        if api_key:
            # Создаем экземпляр API
            weather_api = WeatherAPI(api_key)
            
            # Получение текущей погоды
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌤️ Получение данных")
                
                # Кнопки для разных методов получения
                sync_col, async_col = st.columns(2)
                
                with sync_col:
                    if st.button("🔄 Синхронный запрос", use_container_width=True):
                        with st.spinner("Получение данных..."):
                            weather_data = weather_api.get_current_weather_sync(selected_city)
                            st.session_state['weather_data'] = weather_data
                            st.session_state['last_update'] = datetime.now()
                
                with async_col:
                    if st.button("⚡ Асинхронный запрос", use_container_width=True):
                        async def fetch_async():
                            async with aiohttp.ClientSession() as session:
                                return await weather_api.get_current_weather_async(selected_city, session)
                        
                        with st.spinner("Асинхронное получение данных..."):
                            weather_data = asyncio.run(fetch_async())
                            st.session_state['weather_data'] = weather_data
                            st.session_state['last_update'] = datetime.now()
            
            # Отображение текущей погоды
            if 'weather_data' in st.session_state:
                weather_data = st.session_state['weather_data']
                
                if weather_data['success']:
                    with col2:
                        st.subheader("📊 Текущие показатели")
                        
                        # Карточка с текущей погодой
                        temp_col, feels_col = st.columns(2)
                        
                        with temp_col:
                            st.metric(
                                "Температура",
                                f"{weather_data['temperature']:.1f}°C",
                                delta=f"Ощущается как {weather_data['feels_like']:.1f}°C"
                            )
                        
                        with feels_col:
                            st.metric("Влажность", f"{weather_data['humidity']}%")
                        
                        # Дополнительная информация
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.metric("Давление", f"{weather_data['pressure']} hPa")
                        
                        with info_col2:
                            st.metric("Ветер", f"{weather_data['wind_speed']} м/с")
                        
                        st.markdown(f"**Описание:** {weather_data['description'].capitalize()}")
                        
                        if 'last_update' in st.session_state:
                            st.caption(f"Последнее обновление: {st.session_state['last_update'].strftime('%H:%M:%S')}")
                    
                    # Проверка нормальности температуры
                    st.subheader("📊 Анализ нормальности температуры")
                    
                    # Определение текущего сезона
                    month_to_season = {
                        12: "winter", 1: "winter", 2: "winter",
                        3: "spring", 4: "spring", 5: "spring",
                        6: "summer", 7: "summer", 8: "summer",
                        9: "autumn", 10: "autumn", 11: "autumn"
                    }
                    
                    current_month = datetime.now().month
                    current_season = month_to_season.get(current_month, "winter")
                    
                    # Получаем историческую статистику для текущего сезона
                    if current_season in result['season_limits']:
                        season_stats = result['season_limits'][current_season]
                        current_temp = weather_data['temperature']
                        
                        # Проверяем нормальность
                        is_normal = (
                            season_stats['lower'] <= current_temp <= season_stats['upper']
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Текущая температура",
                                f"{current_temp:.1f}°C"
                            )
                        
                        with col2:
                            st.metric(
                                "Средняя для сезона",
                                f"{season_stats['mean']:.1f}°C",
                                delta=f"±{season_stats['std']:.1f}°C"
                            )
                        
                        with col3:
                            deviation = current_temp - season_stats['mean']
                            z_score = deviation / season_stats['std'] if season_stats['std'] > 0 else 0
                            
                            st.metric(
                                "Отклонение",
                                f"{deviation:+.1f}°C",
                                delta=f"Z-score: {z_score:.2f}"
                            )
                        
                        # Статус нормальности
                        st.markdown("---")
                        
                        if is_normal:
                            st.success(f"""
                            ✅ **Температура в пределах нормы для {current_season}**
                            
                            **Диапазон нормы:** {season_stats['lower']:.1f}°C - {season_stats['upper']:.1f}°C
                            """)
                        else:
                            st.error(f"""
                            ⚠️ **АНОМАЛЬНАЯ ТЕМПЕРАТУРА для {current_season}**
                            
                            **Нормальный диапазон:** {season_stats['lower']:.1f}°C - {season_stats['upper']:.1f}°C
                            **Текущая температура:** {current_temp:.1f}°C
                            **Отклонение:** {deviation:+.1f}°C
                            """)
                        
                        # Визуализация сравнения
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        # Гистограмма исторических данных
                        season_data = result['city_data'][result['city_data']['season'] == current_season]
                        ax.hist(season_data['temperature'], bins=30, alpha=0.5, 
                                label=f'Исторические данные ({current_season})', color='skyblue')
                        
                        # Границы нормы
                        ax.axvspan(season_stats['lower'], season_stats['upper'], 
                                  alpha=0.2, color='green', label='Нормальный диапазон (±2σ)')
                        
                        # Средняя линия
                        ax.axvline(season_stats['mean'], color='green', linestyle='--', 
                                 label=f'Среднее: {season_stats["mean"]:.1f}°C')
                        
                        # Текущая температура
                        ax.axvline(current_temp, color='red', linewidth=3, 
                                 label=f'Текущая: {current_temp:.1f}°C')
                        
                        ax.set_xlabel('Температура (°C)')
                        ax.set_ylabel('Частота')
                        ax.set_title(f'Сравнение текущей температуры с историческими данными ({current_season})')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                    else:
                        st.warning(f"Нет исторических данных для сезона {current_season}")
                
                else:
                    st.error(f"❌ Ошибка при получении данных: {weather_data.get('message', 'Неизвестная ошибка')}")
                    
                    if weather_data.get('error') == 'Invalid API key':
                        st.error("""
                        **Неверный API ключ.**
                        
                        Пожалуйста, проверьте:
                        1. Правильность введенного ключа
                        2. Активирован ли ключ (может потребоваться 2-3 часа после регистрации)
                        3. Не исчерпан ли лимит запросов
                        """)
            else:
                st.info("Нажмите кнопку для получения текущей температуры")
        
        else:
            st.warning("""
            ⚠️ **Для получения текущей погоды требуется API ключ OpenWeatherMap**
            
            **Как получить ключ:**
            1. Зарегистрируйтесь на [OpenWeatherMap](https://openweathermap.org/api)
            2. Получите бесплатный API ключ (до 1000 запросов в день)
            3. Вставьте ключ в поле ввода в боковой панели
            4. **Важно:** Ключ может активироваться 2-3 часа
            """)
            
            # Демо-режим
            st.subheader("🔧 Демо-режим")
            
            # Генерируем демо-данные согласно условию задания
            current_month = datetime.now().month
            month_to_season = {
                12: "winter", 1: "winter", 2: "winter",
                3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer",
                9: "autumn", 10: "autumn", 11: "autumn"
            }
            current_season = month_to_season.get(current_month, "winter")
            
            # Базовые температуры для демо
            base_temps = {
                "Berlin": 10, "Cairo": 25, "Dubai": 30,
                "Beijing": 13, "Moscow": 5
            }
            
            if selected_city in base_temps:
                base_temp = base_temps[selected_city]
                
                # Согласно условию задания
                if selected_city in ["Berlin", "Cairo", "Dubai"]:
                    # Нормальная температура
                    demo_temp = base_temp + np.random.uniform(-3, 3)
                    is_normal = True
                    status = "✅ В РАМКАХ НОРМЫ"
                else:
                    # Аномальная температура для Москвы и Пекина
                    demo_temp = base_temp + np.random.choice([-12, 15])
                    is_normal = False
                    status = "⚠️ АНОМАЛЬНАЯ"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Демо-температура", f"{demo_temp:.1f}°C")
                    st.metric("Сезон", current_season)
                
                with col2:
                    if is_normal:
                        st.success(status)
                    else:
                        st.error(status)
                
                st.info("""
                **Примечание:** Это демонстрационные данные согласно условию задания:
                - Берлин, Каир, Дубай: температура в рамках нормы
                - Пекин, Москва: аномальная температура
                """)
    
    with tab3:
        st.header(f"Визуализации: {selected_city}")
        
        # Временной ряд с аномалиями
        st.subheader("📈 Временной ряд температур")
        fig_time_series = create_temperature_time_series(
            result['city_data'], 
            result['anomalies']
        )
        st.plotly_chart(fig_time_series, use_container_width=True)
        
        # Сезонные профили
        st.subheader("🍂 Сезонные профили")
        fig_seasonal = create_seasonal_profile(result['seasonal_stats'])
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Распределение температур
            st.subheader("📊 Распределение температур")
            
            current_temp = None
            if 'weather_data' in st.session_state:
                weather_data = st.session_state['weather_data']
                if weather_data.get('success'):
                    current_temp = weather_data['temperature']
            
            fig_dist = create_distribution_plot(result['city_data'], current_temp)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Боксплот по сезонам
            st.subheader("📦 Box-plot по сезонам")
            fig_box = create_box_plot_by_season(result['city_data'])
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Тепловая карта по годам и месяцам
        st.subheader("🔥 Тепловая карта (год × месяц)")
        
        # Подготовка данных для тепловой карты
        heatmap_data = result['city_data'].copy()
        heatmap_data['year'] = heatmap_data['timestamp'].dt.year
        heatmap_data['month'] = heatmap_data['timestamp'].dt.month
        
        pivot_table = heatmap_data.pivot_table(
            values='temperature',
            index='year',
            columns='month',
            aggfunc='mean'
        )
        
        fig_heatmap = px.imshow(
            pivot_table,
            labels=dict(x="Месяц", y="Год", color="Температура (°C)"),
            x=['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
               'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'],
            color_continuous_scale='RdYlBu_r'
        )
        
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.header("⚡ Сравнение производительности")
        
        st.subheader("Распараллеливание анализа исторических данных")
        
        col1, col2 = st.columns(2)
        
        # Кнопки для запуска разных типов анализа
        with col1:
            if st.button("🚀 Запустить последовательный анализ", use_container_width=True):
                with st.spinner("Выполняется последовательный анализ..."):
                    start_time = time.time()
                    seq_results = analyzer.analyze_sequential()
                    elapsed_time = time.time() - start_time
                    st.session_state['seq_time'] = elapsed_time
                    st.session_state['seq_results'] = seq_results
                    st.success(f"Последовательный анализ завершен за {elapsed_time:.2f} секунд")
        
        with col2:
            if st.button("⚡ Запустить параллельный анализ", use_container_width=True):
                with st.spinner(f"Выполняется параллельный анализ ({max_workers} процессов)..."):
                    start_time = time.time()
                    par_results = analyzer.analyze_parallel(max_workers=max_workers)
                    elapsed_time = time.time() - start_time
                    st.session_state['par_time'] = elapsed_time
                    st.session_state['par_results'] = par_results
                    st.success(f"Параллельный анализ завершен за {elapsed_time:.2f} секунд")
        
        # Отображение результатов сравнения
        if 'seq_time' in st.session_state and 'par_time' in st.session_state:
            seq_time = st.session_state['seq_time']
            par_time = st.session_state['par_time']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Последовательный", f"{seq_time:.2f} сек")
            
            with col2:
                st.metric("Параллельный", f"{par_time:.2f} сек")
            
            with col3:
                if par_time > 0:
                    speedup = seq_time / par_time
                    efficiency = (speedup / max_workers) * 100
                    st.metric("Ускорение", f"{speedup:.2f}x", delta=f"Эффективность: {efficiency:.1f}%")
                else:
                    st.metric("Ускорение", "N/A")
            
            # Визуализация сравнения
            fig, ax = plt.subplots(figsize=(8, 4))
            
            methods = ['Последовательный', f'Параллельный ({max_workers} процессов)']
            times = [seq_time, par_time]
            
            bars = ax.bar(methods, times, color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel('Время (секунды)', fontsize=12)
            ax.set_title('Сравнение производительности анализа', fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(times) * 1.2)
            
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{time_val:.2f} сек', ha='center', va='bottom', fontsize=11)
            
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        st.subheader("📝 Рекомендации по выбору методов")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔄 Синхронные запросы
            
            **Когда использовать:**
            - Получение температуры для одного города
            - Простые приложения
            - Когда простота важнее производительности
            
            **Преимущества:**
            - Простота реализации
            - Легкая отладка
            - Понятный поток выполнения
            """)
        
        with col2:
            st.markdown("""
            ### ⚡ Асинхронные запросы
            
            **Когда использовать:**
            - Мониторинг нескольких городов одновременно
            - Высоконагруженные приложения
            - Когда важна отзывчивость интерфейса
            
            **Преимущества:**
            - Неблокирующие операции
            - Высокая производительность
            - Эффективное использование ресурсов
            """)
        
        # Таблица производительности по городам
        if 'par_results' in st.session_state:
            st.subheader("📊 Статистика аномалий по городам")
            
            anomaly_stats = []
            for city, city_result in st.session_state['par_results'].items():
                anomaly_stats.append({
                    'Город': city,
                    'Наблюдений': city_result['total_observations'],
                    'Аномалий': city_result['num_anomalies'],
                    'Процент аномалий': f"{(city_result['num_anomalies'] / city_result['total_observations'] * 100):.2f}%"
                })
            
            anomaly_df = pd.DataFrame(anomaly_stats)
            st.dataframe(
                anomaly_df.sort_values('Процент аномалий', ascending=False),
                use_container_width=True,
                hide_index=True
            )

# ============================================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == "__main__":
    # Установка стилей
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .css-1d391kg {
        padding-top: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Запуск основного приложения
    main()