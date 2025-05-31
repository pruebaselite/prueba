import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Importar desde nuestros módulos
import database_manager as db_m

# Cargar variables de entorno desde .env al inicio del script
load_dotenv()

# --- Configuración de Credenciales (leídas de .env o valores por defecto) ---
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '12345678'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER', 'ServerName')
# Usar barras dobles para compatibilidad en strings de Python, o usar raw strings r"path"
MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\MetaTrader 5\terminal64.exe")


# --- Funciones de Indicadores Técnicos ---

def calculate_sma(data_series, window):
    """Calcula la Media Móvil Simple (SMA)."""
    if not isinstance(data_series, pd.Series):
        raise ValueError("data_series debe ser una Serie de Pandas.")
    if len(data_series) < window:
        return pd.Series([np.nan] * len(data_series), index=data_series.index, name=f'SMA_{window}')
    return data_series.rolling(window=window).mean().rename(f'SMA_{window}')

def calculate_rsi(data_series, window=14):
    """Calcula el Índice de Fuerza Relativa (RSI)."""
    if not isinstance(data_series, pd.Series):
        raise ValueError("data_series debe ser una Serie de Pandas.")
    if len(data_series) <= window:
        return pd.Series([np.nan] * len(data_series), index=data_series.index, name=f'RSI_{window}')

    delta = data_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True) # Evitar inf si loss es 0
    rs.fillna(method='ffill', inplace=True) # Propagar para evitar NaNs intermedios si es posible

    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f'RSI_{window}')

def calculate_atr(df_ohlc, window=14):
    """Calcula el Average True Range (ATR)."""
    if not all(col in df_ohlc.columns for col in ['high', 'low', 'close']):
        raise ValueError("El DataFrame debe contener columnas 'high', 'low', 'close'.")
    if len(df_ohlc) < window:
        return pd.Series([np.nan] * len(df_ohlc), index=df_ohlc.index, name=f'ATR_{window}')

    df = df_ohlc.copy()
    df['h_minus_l'] = df['high'] - df['low']
    df['h_minus_cp'] = abs(df['high'] - df['close'].shift(1))
    df['l_minus_cp'] = abs(df['low'] - df['close'].shift(1))

    df['tr'] = df[['h_minus_l', 'h_minus_cp', 'l_minus_cp']].max(axis=1)
    # Usar ewm para la MMA de Wilder, min_periods=window para asegurar suficientes datos para el primer cálculo
    atr_series = df['tr'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return atr_series.rename(f'ATR_{window}')

def calculate_adx(df_ohlc, window=14):
    """Calcula el Average Directional Index (ADX), +DI, -DI."""
    if not all(col in df_ohlc.columns for col in ['high', 'low', 'close']):
        raise ValueError("El DataFrame debe contener columnas 'high', 'low', 'close'.")
    if len(df_ohlc) < window * 2: # ADX necesita más datos
        empty_series = pd.Series([np.nan] * len(df_ohlc), index=df_ohlc.index)
        return empty_series.rename(f'PLUS_DI_{window}'), empty_series.rename(f'MINUS_DI_{window}'), empty_series.rename(f'ADX_{window}')

    df = df_ohlc.copy()
    atr = calculate_atr(df, window) # Usar la función ATR ya definida
    df[f'ATR_{window}'] = atr

    df['move_up'] = df['high'].diff()
    df['move_down'] = -df['low'].diff()

    df['plus_dm'] = 0.0
    df.loc[(df['move_up'] > df['move_down']) & (df['move_up'] > 0), 'plus_dm'] = df['move_up']
    df['minus_dm'] = 0.0
    df.loc[(df['move_down'] > df['move_up']) & (df['move_down'] > 0), 'minus_dm'] = df['move_down']

    smooth_plus_dm = df['plus_dm'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    smooth_minus_dm = df['minus_dm'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    # Evitar división por cero si ATR es 0 o NaN
    plus_di = (smooth_plus_dm / df[f'ATR_{window}'].replace(0, np.nan)) * 100
    minus_di = (smooth_minus_dm / df[f'ATR_{window}'].replace(0, np.nan)) * 100

    # Rellenar NaN iniciales si ATR es NaN o 0, o si DM es NaN
    plus_di.fillna(0, inplace=True)
    minus_di.fillna(0, inplace=True)

    dx_denominator = (plus_di + minus_di).replace(0, np.nan) # Evitar división por cero
    dx = (abs(plus_di - minus_di) / dx_denominator) * 100
    dx.fillna(0, inplace=True)

    adx = dx.ewm(alpha=1/window, adjust=False, min_periods=window-1).mean() # ADX se suaviza sobre DX

    return plus_di.rename(f'PLUS_DI_{window}'), minus_di.rename(f'MINUS_DI_{window}'), adx.rename(f'ADX_{window}')

def calculate_bollinger_bands(data_series, window=20, num_std_dev=2):
    """Calcula las Bandas de Bollinger."""
    if not isinstance(data_series, pd.Series):
        raise ValueError("data_series debe ser una Serie de Pandas.")
    if len(data_series) < window:
         empty_series = pd.Series([np.nan] * len(data_series), index=data_series.index)
         return empty_series.rename(f'BB_MID_{window}'), empty_series.rename(f'BB_UPPER_{window}'), empty_series.rename(f'BB_LOWER_{window}')

    middle_band = data_series.rolling(window=window).mean()
    std_dev = data_series.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return middle_band.rename(f'BB_MID_{window}'), upper_band.rename(f'BB_UPPER_{window}'), lower_band.rename(f'BB_LOWER_{window}')

def calculate_macd(data_series, fast_period=12, slow_period=26, signal_period=9):
    """Calcula el MACD, la línea de señal y el histograma."""
    if not isinstance(data_series, pd.Series):
        raise ValueError("data_series debe ser una Serie de Pandas.")
    min_len = slow_period + signal_period -1
    if len(data_series) < min_len:
        empty_series = pd.Series([np.nan] * len(data_series), index=data_series.index)
        return empty_series.rename(f'MACD_{fast_period}_{slow_period}'), empty_series.rename(f'MACD_SIGNAL_{signal_period}'), empty_series.rename(f'MACD_HIST_{signal_period}')

    ema_fast = data_series.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = data_series.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd_line - signal_line

    return macd_line.rename(f'MACD_{fast_period}_{slow_period}'), signal_line.rename(f'MACD_SIGNAL_{signal_period}'), histogram.rename(f'MACD_HIST_{signal_period}')

# --- Conexión con MetaTrader 5 ---
# (Conditionally import MetaTrader5 and define constants if import fails - from previous step)
try:
    import MetaTrader5 as mt5
    # Timeframe constants from MT5
    TIMEFRAME_M1 = mt5.TIMEFRAME_M1
    TIMEFRAME_M5 = mt5.TIMEFRAME_M5
    TIMEFRAME_M15 = mt5.TIMEFRAME_M15
    TIMEFRAME_M30 = mt5.TIMEFRAME_M30
    TIMEFRAME_H1 = mt5.TIMEFRAME_H1
    TIMEFRAME_H4 = mt5.TIMEFRAME_H4
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1
    TIMEFRAME_W1 = mt5.TIMEFRAME_W1
    TIMEFRAME_MN1 = mt5.TIMEFRAME_MN1
    MT5_AVAILABLE = True
except ImportError:
    print("Advertencia: El paquete MetaTrader5 no está instalado. Funcionalidad MT5 no disponible.")
    MT5_AVAILABLE = False
    # Define placeholders for timeframe constants if MT5 is not available
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 2, 3, 4
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1, TIMEFRAME_W1, TIMEFRAME_MN1 = 5, 6, 7, 8, 9


def initialize_mt5_connection(login, password, server, path):
    if not MT5_AVAILABLE:
        print("MT5 no disponible, no se puede inicializar la conexión.")
        return False

    print(f"Intentando inicializar MT5 desde: {path}")
    # Intentar con el path primero
    init_kwargs = {'login': login, 'password': password, 'server': server}
    if path and path.lower() != 'none' and path.strip() != '': # Solo añadir path si es válido
        init_kwargs['path'] = path

    if not mt5.initialize(**init_kwargs):
        print(f"Error en mt5.initialize() con path '{path}': {mt5.last_error()}")
        # Intentar sin especificar el path si el primero falla o si no se proveyó path
        if 'path' in init_kwargs: # Si falló con path, reintentar sin él
            del init_kwargs['path']
            if not mt5.initialize(**init_kwargs):
                print(f"Error en mt5.initialize() (sin path): {mt5.last_error()}")
                return False
        else: # Si falló sin path, entonces es un fallo definitivo
            return False

    terminal_info = mt5.terminal_info()
    if not terminal_info:
        print(f"No se pudo obtener información del terminal: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print(f"Conexión a MT5 establecida: {terminal_info.name}, Version: {terminal_info.version}")
    return True

def shutdown_mt5_connection():
    if MT5_AVAILABLE:
        mt5.shutdown()
        print("Conexión con MT5 cerrada.")

def get_historical_data_mt5(symbol, timeframe_mt5_const, num_bars=1000, start_date=None, end_date=None):
    if not MT5_AVAILABLE or not mt5.terminal_info():
        print("MT5 no conectado o no disponible. No se pueden descargar datos.")
        return None

    print(f"Descargando datos para {symbol} en timeframe {timeframe_mt5_const}...")
    rates = None
    try:
        if start_date and end_date:
            rates = mt5.copy_rates_range(symbol, timeframe_mt5_const, start_date, end_date)
        elif start_date:
            rates = mt5.copy_rates_range(symbol, timeframe_mt5_const, start_date, datetime.now())
        else:
            rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5_const, 0, num_bars)
    except Exception as e:
        print(f"Excepción al llamar a mt5.copy_rates: {e}. MT5 last_error: {mt5.last_error()}")
        return None

    if rates is None or len(rates) == 0:
        print(f"No se pudieron obtener datos para {symbol}: {mt5.last_error()}")
        return None

    rates_df = pd.DataFrame(rates)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
    rates_df.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'}, inplace=True)
    if 'volume' not in rates_df.columns and 'real_volume' in rates_df.columns:
        rates_df.rename(columns={'real_volume': 'volume'}, inplace=True)
    elif 'volume' not in rates_df.columns:
         rates_df['volume'] = 0
    rates_df = rates_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    print(f"Descargados {len(rates_df)} registros para {symbol}.")
    return rates_df

def store_historical_data(df, symbol, timeframe_str):
    if df is None or df.empty:
        print("DataFrame vacío, no hay datos para almacenar.")
        return
    df_to_store = df.copy()
    df_to_store['symbol'] = symbol
    df_to_store['timeframe'] = timeframe_str
    data_list = df_to_store.to_dict(orient='records')

    db_session_gen = db_m.get_db_session()
    session = next(db_session_gen)
    try:
        db_m.add_historical_data(session, data_list)
        print(f"Datos para {symbol}/{timeframe_str} almacenados en la base de datos.")
    except Exception as e:
        print(f"Error al almacenar datos históricos en la BD: {e}")
    finally:
        session.close()

# --- Función Principal de Ejemplo / Prueba ---
if __name__ == "__main__":
    print("Ejecutando data_manager como script principal para pruebas...")
    db_m.initialize_database()

    mt5_connected = False
    if MT5_AVAILABLE: # Solo intentar conectar si el paquete está disponible
        print(f"Intentando conectar a MT5 con login: {MT5_LOGIN}, server: {MT5_SERVER}, path: {MT5_PATH}")
        mt5_connected = initialize_mt5_connection(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH)

    test_symbol = "SIM_EURUSD" # Cambiado para consistencia con pruebas anteriores
    timeframe_map_str_to_const = { "H1": TIMEFRAME_H1 } # Ejemplo
    test_timeframe_str = "H1"
    test_timeframe_mt5 = timeframe_map_str_to_const.get(test_timeframe_str)
    historical_df = None

    if mt5_connected and test_timeframe_mt5 is not None:
        print(f"\nDescargando datos de MT5 para {test_symbol}...")
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=60) # Más datos para indicadores complejos
        historical_df = get_historical_data_mt5(test_symbol, test_timeframe_mt5, start_date=start_dt, end_date=end_dt)
        if historical_df is not None and not historical_df.empty:
            print(f"\nDatos de {test_symbol} descargados de MT5. Almacenando...")
            store_historical_data(historical_df, test_symbol, test_timeframe_str)
        else:
            print(f"No se pudieron descargar datos de MT5 para {test_symbol}. Usando datos simulados.")
            historical_df = None # Asegurar que se usan los simulados

    if historical_df is None: # Si falló MT5 o no se conectó
        print("Generando datos simulados para pruebas de indicadores...")
        # Generar suficientes datos para que los indicadores complejos no den solo NaNs
        num_sim_points = 200
        sim_dates = pd.to_datetime([datetime(2023, 1, 1) + timedelta(hours=i) for i in range(num_sim_points)])
        data = {
            'timestamp': sim_dates,
            'open': np.random.rand(num_sim_points) * 10 + 100,
        }
        data['high'] = data['open'] + np.random.rand(num_sim_points) * 2
        data['low'] = data['open'] - np.random.rand(num_sim_points) * 2
        data['close'] = data['open'] + (np.random.rand(num_sim_points) - 0.5) * 3
        data['volume'] = np.random.randint(100, 1000, num_sim_points)
        historical_df = pd.DataFrame(data)
        # Almacenar los datos simulados para que otros módulos puedan usarlos si es necesario
        store_historical_data(historical_df, test_symbol, test_timeframe_str)


    if historical_df is not None and not historical_df.empty:
        df_for_indicators = historical_df.copy()
        if not isinstance(df_for_indicators.index, pd.DatetimeIndex) and 'timestamp' in df_for_indicators.columns:
             df_for_indicators.set_index('timestamp', inplace=True) # Necesario para algunos cálculos basados en diff, shift

        print("\n--- Calculando Indicadores ---")
        print(f"DataFrame original (primeras filas):\n{df_for_indicators.head()}")

        df_for_indicators['SMA_10'] = calculate_sma(df_for_indicators['close'], 10)
        df_for_indicators['RSI_14'] = calculate_rsi(df_for_indicators['close'], 14)

        macd_line, macd_signal, macd_hist = calculate_macd(df_for_indicators['close'])
        df_for_indicators['MACD_line'] = macd_line
        df_for_indicators['MACD_signal'] = macd_signal
        df_for_indicators['MACD_hist'] = macd_hist

        bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(df_for_indicators['close'])
        df_for_indicators['BB_Mid'] = bb_mid
        df_for_indicators['BB_Upper'] = bb_upper
        df_for_indicators['BB_Lower'] = bb_lower

        # ADX necesita df con high, low, close
        plus_di, minus_di, adx_series = calculate_adx(df_for_indicators[['high', 'low', 'close']])
        df_for_indicators['PLUS_DI'] = plus_di
        df_for_indicators['MINUS_DI'] = minus_di
        df_for_indicators['ADX'] = adx_series

        print("\nDataFrame con todos los indicadores (últimas 30 filas para ver cálculos):")
        # Mostrar columnas relevantes y las últimas filas donde los cálculos son más probables de no ser NaN
        cols_to_show = ['open', 'high', 'low', 'close', 'SMA_10', 'RSI_14', 'MACD_line', 'MACD_signal', 'BB_Mid', 'BB_Upper', 'ADX']
        print(df_for_indicators[cols_to_show].tail(30))
    else:
        print("No hay datos (ni reales ni simulados) para calcular indicadores.")

    if mt5_connected:
        shutdown_mt5_connection()

    print("\nPruebas de data_manager (con nuevos indicadores y .env) finalizadas.")
