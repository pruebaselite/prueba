import time
import datetime
import logging
import pandas as pd
import numpy as np # Para usar np.nan

# Importar desde nuestros módulos
import data_manager as dm
import ml_engine as mle
import database_manager as db_m

# --- Configuración del Logging ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("TradingBot")

# --- Clase Principal del Bot ---
class TradingBot:
    def __init__(self, symbol, timeframe_str, model_name_prefix, mt5_conn_params=None):
        # TEST CHANGE FOR GITHUB DESKTOP SYNC - v2
        """
        Inicializa el Trading Bot.
        :param symbol: Símbolo del instrumento (ej: "EURUSD").
        :param timeframe_str: Timeframe en formato string (ej: "H1").
        :param model_name_prefix: Prefijo del modelo a cargar (ej: "LGBM_EURUSD_H1").
        :param mt5_conn_params: Diccionario con parámetros de conexión a MT5
                                 (login, password, server, path). Si es None, no intenta conectar.
        """
        self.symbol = symbol
        self.timeframe_str = timeframe_str
        self.model_name_prefix = model_name_prefix
        self.mt5_connected = False
        self.model = None
        self.model_entry = None # Para guardar info del modelo cargado
        self.last_data_timestamp = None # Para evitar procesar la misma barra múltiples veces

        # Mapeo de timeframe string a constante MT5 (ejemplo)
        # Usamos las constantes definidas en data_manager
        self.timeframe_map = {
            "M1": dm.TIMEFRAME_M1, "M5": dm.TIMEFRAME_M5, "M15": dm.TIMEFRAME_M15,
            "M30": dm.TIMEFRAME_M30, "H1": dm.TIMEFRAME_H1, "H4": dm.TIMEFRAME_H4,
            "D1": dm.TIMEFRAME_D1, "W1": dm.TIMEFRAME_W1, "MN1": dm.TIMEFRAME_MN1
        }
        self.timeframe_mt5 = self.timeframe_map.get(self.timeframe_str)

        # Comprobar si el timeframe es válido (debe existir en el mapeo)
        # Y si MT5 está disponible, debe ser un entero (constante real de MT5)
        if self.timeframe_mt5 is None:
            logger.error(f"Timeframe string no reconocido: {self.timeframe_str}")
            raise ValueError(f"Timeframe string no reconocido: {self.timeframe_str}")

        if dm.MT5_AVAILABLE and not isinstance(self.timeframe_mt5, int):
            logger.error(f"Timeframe {self.timeframe_str} no es una constante MT5 válida a pesar de que MT5 está disponible.")
            # Esto podría indicar un problema en la definición de constantes en data_manager si MT5_AVAILABLE es True.
            # Por seguridad, podríamos impedir la continuación si se espera una conexión MT5.
            # raise ValueError(f"Timeframe {self.timeframe_str} inválido para MT5.")
            logger.warning(f"Timeframe {self.timeframe_str} podría no ser compatible con MT5 aunque esté disponible.")


        # Inicializar conexión con MT5 (si se proporcionan parámetros)
        if mt5_conn_params:
            self.mt5_connected = dm.initialize_mt5_connection(
                login=mt5_conn_params['login'],
                password=mt5_conn_params['password'],
                server=mt5_conn_params['server'],
                path=mt5_conn_params['path']
            )
            if not self.mt5_connected:
                logger.warning("No se pudo conectar a MT5. El bot operará con datos limitados o simulados si es posible.")
        else:
            logger.info("Parámetros de conexión MT5 no proporcionados. Se salta la conexión MT5.")

        # Cargar el último modelo entrenado
        self._load_model()

    def _load_model(self):
        """Carga el modelo de ML más reciente."""
        logger.info(f"Cargando modelo para {self.model_name_prefix}...")
        self.model, self.model_entry = mle.load_model(self.model_name_prefix)
        if self.model and self.model_entry:
            logger.info(f"Modelo {self.model_entry.model_name} v{self.model_entry.version} cargado exitosamente.")
        else:
            logger.error(f"No se pudo cargar el modelo para {self.model_name_prefix}. El bot no puede operar.")
            # Considerar si el bot debe detenerse o intentar recargar más tarde.

    def _get_latest_market_data(self, num_bars=100):
        """
        Obtiene los últimos datos de mercado y calcula indicadores.
        :param num_bars: Número de barras a solicitar para tener contexto para indicadores.
        :return: DataFrame con los datos más recientes y sus indicadores, o None.
        """
        if self.mt5_connected:
            logger.info(f"Obteniendo datos de mercado para {self.symbol}/{self.timeframe_str}...")
            # Descargar suficientes barras para calcular indicadores (ej. RSI 14 necesita al menos 15 barras)
            # El número exacto de barras depende de los indicadores y lags usados en `prepare_features`
            # Podríamos pedir más barras (ej. 100) y luego tomar la última para la predicción.

            # Intentar obtener las últimas `num_bars` barras
            # Nota: copy_rates_from_pos(..., 0, N) da las últimas N barras, la [N-1] es la más reciente cerrada.
            # Para una predicción en la barra actual (aún no cerrada), se necesitaría otro enfoque o esperar cierre.
            # Asumimos que operamos sobre barras cerradas.
            raw_data_df = dm.get_historical_data_mt5(self.symbol, self.timeframe_mt5, num_bars=num_bars)
        else:
            logger.warning("MT5 no conectado. Intentando obtener datos de la BD (simulación).")
            # Simulación: obtener los últimos N datos de la BD.
            # Esto es solo para prueba y no refleja un flujo en tiempo real sin MT5.
            db_session_gen = db_m.get_db_session()
            session = next(db_session_gen)
            try:
                # Esto no es lo ideal para tiempo real, pero sirve para probar la lógica.
                # Necesitaríamos una fuente de datos simulados que "avance" en el tiempo.
                data_from_db = db_m.get_historical_data(session, self.symbol, self.timeframe_str, limit=num_bars)
                if not data_from_db:
                    logger.error("No hay datos en la BD para simulación.")
                    return None

                # Convertir lista de objetos SQLAlchemy a DataFrame
                raw_data_df = pd.DataFrame([d.__dict__ for d in data_from_db])
                if '_sa_instance_state' in raw_data_df.columns: # Columna interna de SQLAlchemy
                    raw_data_df.drop(columns=['_sa_instance_state'], inplace=True)
                raw_data_df['timestamp'] = pd.to_datetime(raw_data_df['timestamp'])

            finally:
                session.close()

        if raw_data_df is None or raw_data_df.empty:
            logger.error("No se pudieron obtener datos de mercado.")
            return None

        # Calcular indicadores necesarios (los mismos que se usaron para entrenar el modelo)
        # Esto debería estar sincronizado con lo que `prepare_features` espera.
        # Asumimos que el modelo fue entrenado con SMA_10 y RSI_14 como en ml_engine_tests.
        raw_data_df['SMA_10'] = dm.calculate_sma(raw_data_df['close'], 10)
        raw_data_df['RSI_14'] = dm.calculate_rsi(raw_data_df['close'], 14)

        # Eliminar NaNs generados por indicadores (afecta a las primeras filas)
        # raw_data_df.dropna(inplace=True) # Esto podría eliminar demasiadas filas si num_bars es pequeño
                                         # Es mejor que prepare_features maneje los NaNs finales

        if raw_data_df.empty:
            logger.warning("DataFrame vacío después de calcular indicadores.")
            return None

        return raw_data_df

    def run_cycle(self):
        """Ejecuta un ciclo de operación del bot."""
        if not self.model:
            logger.error("Modelo no cargado. Saltando ciclo.")
            return

        logger.info("Iniciando nuevo ciclo de trading...")

        # 1. Obtener datos de mercado y calcular indicadores
        # Pedimos más barras (ej. 50) para asegurar que hay suficientes datos para indicadores y features.
        market_data_df = self._get_latest_market_data(num_bars=50)

        if market_data_df is None or market_data_df.empty:
            logger.warning("No hay datos de mercado disponibles para este ciclo.")
            return

        # Tomar la última fila completa para la predicción (la barra más reciente cerrada)
        # Asegurarse de que esta fila no tenga NaNs en las features que usará el modelo.
        latest_data_point = market_data_df.iloc[[-1]].copy() # .iloc[[-1]] mantiene el formato DataFrame

        # Verificar si es la misma barra que la última vez (si no es la primera ejecución)
        current_timestamp = latest_data_point['timestamp'].iloc[0]
        if self.last_data_timestamp and self.last_data_timestamp == current_timestamp:
            logger.info(f"No hay nueva barra desde {current_timestamp}. Saltando ciclo.")
            return
        self.last_data_timestamp = current_timestamp

        logger.info(f"Datos más recientes para predicción (barra cerrada en {current_timestamp}):\n{latest_data_point[['open', 'high', 'low', 'close', 'SMA_10', 'RSI_14']]}")

        # 2. Preparar características para el modelo
        # `prepare_features` crea 'target', pero para predicción no lo necesitamos y puede causar NaNs si no hay 'future_close'.
        # Modificamos ligeramente cómo llamamos o usamos `prepare_features` para predicción.
        # Lo ideal es tener una función `extract_live_features` que no dependa de 'target'.

        # Por ahora, vamos a pasar el DF y seleccionar las features que el modelo espera.
        # Asumimos que el modelo fue entrenado con 'SMA_10', 'RSI_14', y 'return_1p' (como en ml_engine)
        # Calculamos 'return_1p' para el punto de datos más reciente.
        # Necesitamos al menos el dato anterior para calcular 'return_1p'.
        if len(market_data_df) >= 2:
             # Calcular retorno sobre los datos completos antes de tomar el último punto
             market_data_df['return_1p'] = market_data_df['close'].pct_change(1)
             latest_data_point = market_data_df.iloc[[-1]].copy() # Recoger el último punto de nuevo con 'return_1p'
        else:
            latest_data_point['return_1p'] = np.nan # No se puede calcular

        # Seleccionar solo las columnas de características que el modelo espera (debe coincidir con el entrenamiento)
        # Esto es una simplificación. Una mejor manera es guardar las feature names con el modelo.
        # El modelo de LightGBM de scikit-learn recuerda los nombres de las features si se entrenó con un DataFrame.
        feature_columns = ['SMA_10', 'RSI_14', 'return_1p']
        # Si alguna de estas features no está en latest_data_point, la predicción fallará.
        # Revisar si hay NaNs en las features seleccionadas para la predicción.
        if latest_data_point[feature_columns].isnull().any().any():
            logger.warning(f"Hay valores NaN en las características para la predicción en {current_timestamp}. Saltando.")
            logger.debug(f"Valores de características:\n{latest_data_point[feature_columns]}")
            return

        current_features_df = latest_data_point[feature_columns]
        logger.info(f"Características para predicción:\n{current_features_df}")

        # 3. Obtener predicción del modelo
        prediction, probabilities = mle.make_prediction(self.model, current_features_df)

        if prediction is None:
            logger.error("No se pudo obtener predicción del modelo.")
            return

        logger.info(f"Predicción del modelo: {prediction} (Probabilidades: {probabilities})")

        # 4. Tomar decisión de trading (simulada)
        # Lógica de ejemplo: si predicción es 1 (subir), simular compra.
        # Si es 0 (bajar), simular venta. (Esto es muy básico)
        # En un bot real, se consideraría el nivel de confianza (probabilidades),
        # gestión de riesgo, estado actual de posiciones, etc.

        signal = "HOLD" # Por defecto
        if prediction == 1 and probabilities[1] > 0.6: # Umbral de confianza de ejemplo
            signal = "BUY_SIMULATED"
            logger.info(f"Decisión: {signal} para {self.symbol} basado en predicción {prediction} y prob {probabilities[1]:.2f}")
            self._execute_simulated_trade(signal_type="BUY", price=latest_data_point['close'].iloc[0], volume=0.1)
        elif prediction == 0 and probabilities[0] > 0.6: # Umbral de confianza
            signal = "SELL_SIMULATED"
            logger.info(f"Decisión: {signal} para {self.symbol} basado en predicción {prediction} y prob {probabilities[0]:.2f}")
            self._execute_simulated_trade(signal_type="SELL", price=latest_data_point['close'].iloc[0], volume=0.1)
        else:
            logger.info(f"Decisión: {signal}. Predicción: {prediction}, Probs: {probabilities}. No se cumple umbral o señal clara.")


    def _execute_simulated_trade(self, signal_type, price, volume):
        """
        Simula la ejecución de una orden y la registra en la BD.
        """
        trade_data = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc), # Usar UTC para consistencia
            'symbol': self.symbol,
            'order_type': signal_type, # "BUY" o "SELL"
            'price': price,
            'volume': volume,
            'stop_loss': None, # SL/TP se añadirían con gestión de riesgo más avanzada
            'take_profit': None,
            'profit': None, # Se actualizaría al cerrar el trade
            'status': "SIMULATED_OPEN", # Estado para trades simulados
            'mt5_ticket_id': None # No hay ticket real de MT5
        }

        db_session_gen = db_m.get_db_session()
        session = next(db_session_gen)
        try:
            trade_entry = db_m.add_trade(session, trade_data)
            if trade_entry:
                logger.info(f"Trade simulado registrado en BD: ID {trade_entry.id}, Tipo {signal_type}, Precio {price}, Volumen {volume}")
            else:
                logger.error("No se pudo registrar el trade simulado en la BD.")
        finally:
            session.close()

    def stop(self):
        """Detiene el bot y cierra conexiones."""
        logger.info("Deteniendo el bot...")
        if self.mt5_connected:
            dm.shutdown_mt5_connection()
        logger.info("Bot detenido.")


# --- Bucle Principal de Ejemplo ---
if __name__ == "__main__":
    logger.info("Iniciando Trading Bot (simulación) - VERIFICANDO ACTUALIZACIÓN GIT v2...")

    # --- Configuración ---
    # Estos parámetros deberían venir de un archivo de configuración o argumentos de línea de comandos
    # Ajustar para que coincida con el modelo guardado por ml_engine.py en su __main__
    SYMBOL = "SIM_EURUSD"
    TIMEFRAME = "H1"
    MODEL_NAME_PREFIX = f"LGBM_{SYMBOL}_{TIMEFRAME}"

    # Para pruebas locales, puedes poner tus credenciales de MT5 aquí (o mejor, usar .env)
    # Si no se proporcionan, el bot intentará usar datos de la BD (que deben existir)
    mt5_credentials = None
    # Ejemplo de cómo se podrían cargar desde .env (requiere python-dotenv y un archivo .env)
    # from dotenv import load_dotenv
    # load_dotenv()
    # mt5_login = os.getenv("MT5_LOGIN")
    # if mt5_login:
    #     mt5_credentials = {
    #         "login": int(mt5_login),
    #         "password": os.getenv("MT5_PASSWORD"),
    #         "server": os.getenv("MT5_SERVER"),
    #         "path": os.getenv("MT5_PATH", "C:\\Program Files\\MetaTrader 5\\terminal64.exe") # Path por defecto
    #     }

    # Inicializar la base de datos (asegura que las tablas existen)
    db_m.initialize_database()

    # Crear instancia del bot
    bot = TradingBot(symbol=SYMBOL,
                     timeframe_str=TIMEFRAME,
                     model_name_prefix=MODEL_NAME_PREFIX,
                     mt5_conn_params=mt5_credentials)

    if not bot.model:
        logger.error("El modelo no se cargó. El bot no puede iniciar. Revisa ml_engine y los modelos guardados.")
    else:
        # Bucle de ejecución (simulado)
        # En un bot real, esto podría ser un bucle infinito o gestionado por un programador de tareas.
        # El intervalo de tiempo debe coincidir con el timeframe del bot (ej. cada hora para H1).
        # Aquí, para prueba, lo ejecutamos unas pocas veces con un intervalo corto.

        # Intervalo de ciclo (en segundos) - debe ser mayor que el tiempo que tarda un ciclo.
        # Para H1, sería 3600 segundos. Para M1, 60 segundos.
        # Para prueba, usamos un intervalo corto.
        CYCLE_INTERVAL_SECONDS = 10 # Para pruebas rápidas
        NUM_CYCLES_TO_RUN = 5 # Número de ciclos de prueba

        try:
            for i in range(NUM_CYCLES_TO_RUN):
                logger.info(f"--- Iniciando Ciclo de Bot {i+1}/{NUM_CYCLES_TO_RUN} ---")
                bot.run_cycle()

                if i < NUM_CYCLES_TO_RUN - 1: # No esperar después del último ciclo
                    logger.info(f"Esperando {CYCLE_INTERVAL_SECONDS} segundos para el próximo ciclo...")
                    time.sleep(CYCLE_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Interrupción por teclado recibida.")
        finally:
            bot.stop()
            logger.info("Trading Bot finalizado.")

# Crear el archivo trading_bot.py con el contenido anterior
