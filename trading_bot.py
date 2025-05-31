import time
import datetime
import logging
import pandas as pd
import numpy as np
import os

# Importar desde nuestros módulos
import data_manager as dm
import ml_engine as mle
import database_manager as db_m # Alias db_m para consistencia

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("TradingBot")

class TradingBot:
    def __init__(self, symbol, timeframe_str, model_prefixes, mt5_conn_params=None):
        """
        Inicializa el Trading Bot.
        :param symbol: Símbolo del instrumento.
        :param timeframe_str: Timeframe en formato string.
        :param model_prefixes: Diccionario de prefijos de modelos, ej: {'lgbm': 'LGBM_...', 'xgb': 'XGB_...'}.
        :param mt5_conn_params: Parámetros de conexión a MT5.
        """
        self.symbol = symbol
        self.timeframe_str = timeframe_str
        self.model_prefixes = model_prefixes
        self.mt5_connected = False
        self.models_info = [] # Lista para almacenar diccionarios con {'type', 'model', 'feature_names'}
        self.last_data_timestamp = None

        self.timeframe_map = {
            "M1": dm.TIMEFRAME_M1, "M5": dm.TIMEFRAME_M5, "M15": dm.TIMEFRAME_M15,
            "M30": dm.TIMEFRAME_M30, "H1": dm.TIMEFRAME_H1, "H4": dm.TIMEFRAME_H4,
            "D1": dm.TIMEFRAME_D1, "W1": dm.TIMEFRAME_W1, "MN1": dm.TIMEFRAME_MN1
        }
        self.timeframe_mt5 = self.timeframe_map.get(self.timeframe_str)

        if self.timeframe_mt5 is None:
            raise ValueError(f"Timeframe string no reconocido: {self.timeframe_str}")
        if dm.MT5_AVAILABLE and not isinstance(self.timeframe_mt5, int):
            logger.warning(f"Timeframe {self.timeframe_str} podría no ser compatible con MT5.")

        if mt5_conn_params and mt5_conn_params.get('login'):
            # Pass mt5_conn_params directly as it's a dictionary
            self.mt5_connected = dm.initialize_mt5_connection(
                login=mt5_conn_params['login'],
                password=mt5_conn_params['password'],
                server=mt5_conn_params['server'],
                path=mt5_conn_params['path']
            )
            if not self.mt5_connected: logger.warning("No se pudo conectar a MT5. Se usarán datos de BD/simulados.")
        else: logger.info("Parámetros MT5 no proporcionados. Usando datos de BD/simulados.")

        self._load_all_models()

    def _load_all_models(self):
        """Carga todos los modelos especificados en self.model_prefixes."""
        self.models_info = []
        if not isinstance(self.model_prefixes, dict):
            logger.error("model_prefixes debe ser un diccionario. No se cargarán modelos.")
            return

        for model_key, prefix in self.model_prefixes.items():
            if not prefix:
                logger.info(f"Prefijo no proporcionado para '{model_key}'. Saltando.")
                continue
            logger.info(f"Cargando modelo '{model_key}' (prefijo: '{prefix}')...")
            model_obj, model_db_entry, feature_names = mle.load_model(prefix)
            if model_obj and model_db_entry:
                self.models_info.append({
                    'type': model_key, # ej. 'lgbm', 'xgb'
                    'model': model_obj,
                    'feature_names': feature_names,
                    'prefix': prefix, # Guardar también el prefijo por si se necesita
                    'version': model_db_entry.version # Guardar versión para logging
                })
                logger.info(f"Modelo {model_key} ({prefix} v{model_db_entry.version}) cargado. Features: {feature_names or 'N/A'}")
            else:
                logger.error(f"Fallo al cargar modelo {model_key} ({prefix}).")

        if not self.models_info:
            logger.error("ENSEMBLE: Ningún modelo fue cargado. El bot no puede operar.")

    def _get_latest_market_data(self, num_bars=150): # Aumentar num_bars para contexto
        raw_data_df = None
        if self.mt5_connected:
            raw_data_df = dm.get_historical_data_mt5(self.symbol, self.timeframe_mt5, num_bars=num_bars)

        if raw_data_df is None:
            logger.warning(f"No datos MT5 para {self.symbol}/{self.timeframe_str}. Usando BD.")
            db_s = next(db_m.get_db_session())
            try:
                data_db = db_m.get_historical_data(db_s, self.symbol, self.timeframe_str, limit=num_bars, order_desc=True) # Corrected: limit=num_bars
                if not data_db: logger.error(f"No datos en BD para {self.symbol}/{self.timeframe_str}."); return None
                raw_data_df = pd.DataFrame([d.__dict__ for d in data_db]); raw_data_df.drop(columns=['_sa_instance_state'],errors='ignore',inplace=True)
                raw_data_df['timestamp'] = pd.to_datetime(raw_data_df['timestamp'])
                raw_data_df.sort_values(by='timestamp', ascending=True, inplace=True); raw_data_df.reset_index(drop=True, inplace=True)
            finally: db_s.close()

        if raw_data_df is None or raw_data_df.empty: logger.error("No datos de mercado."); return None

        df_indic = raw_data_df.copy()
        # Set timestamp as index for indicator calculation if it's a column
        if 'timestamp' in df_indic.columns and not isinstance(df_indic.index, pd.DatetimeIndex):
            df_indic.set_index('timestamp', inplace=True)

        if not all(c in df_indic.columns for c in ['high','low','close']): logger.error("Faltan OHLC."); return None

        df_indic['SMA_10'] = dm.calculate_sma(df_indic['close'], 10)
        df_indic['RSI_14'] = dm.calculate_rsi(df_indic['close'], 14)
        m_l,m_s,m_h=dm.calculate_macd(df_indic['close']); df_indic['MACD_line']=m_l; df_indic['MACD_signal']=m_s; df_indic['MACD_hist']=m_h
        bb_m,bb_u,bb_l=dm.calculate_bollinger_bands(df_indic['close']); df_indic['BB_Mid']=bb_m; df_indic['BB_Upper']=bb_u; df_indic['BB_Lower']=bb_l
        p_di,m_di,adx=dm.calculate_adx(df_indic[['high','low','close']]); df_indic['PLUS_DI']=p_di; df_indic['MINUS_DI']=m_di; df_indic['ADX']=adx
        df_indic['return_1p'] = df_indic['close'].pct_change(1) * 100

        return df_indic.reset_index() # Return with timestamp as column for consistency

    def run_cycle(self):
        if not self.models_info: logger.error("ENSEMBLE: No hay modelos. Saltando ciclo."); return
        logger.info(f"--- Ciclo Ensemble para {self.symbol}/{self.timeframe_str} ---")
        market_data_df = self._get_latest_market_data()
        if market_data_df is None or market_data_df.empty: logger.warning("No datos de mercado."); return

        latest_data_point = market_data_df.iloc[[-1]].copy()
        current_ts = latest_data_point['timestamp'].iloc[0]
        if self.last_data_timestamp and self.last_data_timestamp == current_ts:
            now_utc = pd.Timestamp.now(tz='utc')
            last_ts_utc = pd.Timestamp(self.last_data_timestamp)
            if last_ts_utc.tzinfo is None: last_ts_utc = last_ts_utc.tz_localize('utc') # Assume UTC if naive

            if now_utc - last_ts_utc < pd.Timedelta(seconds=30): # Check if recently processed
                 logger.debug(f"Barra {current_ts} ya procesada recientemente."); return
            logger.info(f"Barra {current_ts} ya procesada."); return
        self.last_data_timestamp = current_ts

        individual_model_probas_p1 = []
        all_predictions_valid = True

        for model_info_dict in self.models_info:
            model_type = model_info_dict['type']
            model_obj = model_info_dict['model']
            expected_feats = model_info_dict['feature_names']

            if not expected_feats:
                logger.warning(f"ENSEMBLE: No hay feature_names para modelo {model_type}. Usando proba neutral (0.5).")
                individual_model_probas_p1.append(0.5)
                continue

            try:
                current_features_df = latest_data_point[expected_feats]
            except KeyError as e:
                logger.error(f"ENSEMBLE: Faltan features para {model_type}: {e}. Esperadas: {expected_feats}. Disponibles: {latest_data_point.columns.tolist()}. Usando proba neutral.")
                individual_model_probas_p1.append(0.5); all_predictions_valid = False
                continue

            if current_features_df.isnull().values.any():
                logger.warning(f"ENSEMBLE: NaNs en features para {model_type} en {current_ts}. Usando proba neutral.")
                logger.debug(f"Features con NaN para {model_type}:\n{current_features_df[current_features_df.isnull().any(axis=1)]}")
                individual_model_probas_p1.append(0.5); all_predictions_valid = False
                continue

            _, pred_probas_dict = mle.make_prediction(model_obj, current_features_df, expected_feats)
            if pred_probas_dict and 1 in pred_probas_dict:
                logger.info(f"ENSEMBLE: Predicción {model_type} P(1)={pred_probas_dict[1]:.3f}")
                individual_model_probas_p1.append(pred_probas_dict[1])
            else:
                logger.warning(f"ENSEMBLE: No se pudo obtener P(1) del modelo {model_type}. Usando proba neutral.")
                individual_model_probas_p1.append(0.5); all_predictions_valid = False

        if not individual_model_probas_p1 or not all_predictions_valid:
            logger.error("ENSEMBLE: No se pudieron obtener todas las predicciones individuales válidas. No se toma decisión.")
            return

        ensemble_weights = [1.0/len(individual_model_probas_p1)] * len(individual_model_probas_p1)

        final_pred_class, final_proba_p1 = mle.make_ensemble_prediction(individual_model_probas_p1, weights=ensemble_weights)

        if final_pred_class is None: logger.error("ENSEMBLE: Fallo al obtener predicción del ensemble."); return

        logger.info(f"ENSEMBLE RESULT: Clase={final_pred_class}, Probabilidad P(1)={final_proba_p1:.3f} (Pesos: {ensemble_weights})")

        signal_type_enum = None
        conf_thresh = float(os.getenv("BOT_ENSEMBLE_CONFIDENCE_THRESHOLD", "0.55"))

        if final_pred_class == 1 and final_proba_p1 >= conf_thresh: signal_type_enum = db_m.OrderTypeEnum.BUY
        elif final_pred_class == 0 and (1 - final_proba_p1) >= conf_thresh: signal_type_enum = db_m.OrderTypeEnum.SELL

        if signal_type_enum:
            logger.info(f"DECISIÓN ENSEMBLE: {signal_type_enum.value} (Proba P(1)={final_proba_p1:.3f}, Umbral={conf_thresh})")
            # Pasar current_features_df a _execute_simulated_trade
            self._execute_simulated_trade(signal_type_enum, latest_data_point['close'].iloc[0], 0.1, current_features_df)
        else: logger.info(f"DECISIÓN ENSEMBLE: HOLD (Proba P(1)={final_proba_p1:.3f}, Umbral={conf_thresh})")

    def _execute_simulated_trade(self, signal_type, price, volume, features_df_for_trade):
        """
        Simula la ejecución de una orden y la registra en la BD, incluyendo las features.
        signal_type debe ser un OrderTypeEnum.
        features_df_for_trade: DataFrame de Pandas con las features del trade.
        """
        features_json = None
        if features_df_for_trade is not None and not features_df_for_trade.empty:
            try:
                features_json = features_df_for_trade.to_json(orient='records')
            except Exception as e:
                logger.error(f"Error al convertir features a JSON: {e}")

        trade_data = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc),
            'symbol': self.symbol,
            'order_type': signal_type,
            'price': price,
            'volume': volume,
            'status': db_m.TradeStatusEnum.SIMULATED_OPEN,
            'open_features_json': features_json # Guardar features
        }

        db_s = next(db_m.get_db_session())
        try:
            entry = db_m.add_trade(db_s, trade_data)
            if entry:
                logger.info(f"Trade simulado (OPEN): ID {entry.id} {signal_type.value} @{price:.5f}, Vol:{volume}. Features guardadas: {'Sí' if features_json else 'No'}")
            else:
                logger.error("Fallo al registrar trade simulado en BD.")
        finally:
            db_s.close()

    def stop(self):
        logger.info("Deteniendo bot...");
        if self.mt5_connected: dm.shutdown_mt5_connection()
        logger.info("Bot detenido.")

if __name__ == "__main__":
    logger.info("Iniciando Trading Bot (con LÓGICA DE ENSEMBLE)...")

    default_sym = "SIM_EURUSD"; default_tf = "H1"
    sym = os.getenv("BOT_SYMBOL", default_sym)
    tf = os.getenv("BOT_TIMEFRAME", default_tf)

    model_prefixes = {
        'lgbm': os.getenv("BOT_LGBM_MODEL_PREFIX", f"LGBM_{sym}_{tf}_EnsembleTest"),
        'xgb': os.getenv("BOT_XGB_MODEL_PREFIX", f"XGB_{sym}_{tf}_EnsembleTest")
    }
    logger.info(f"Bot Ensemble: Symbol={sym}, TF={tf}")
    logger.info(f"LGBM Prefix: {model_prefixes['lgbm']}, XGB Prefix: {model_prefixes['xgb']}")

    mt5_creds = None
    if os.getenv("MT5_LOGIN"):
        try:
            mt5_creds = {"login":int(os.getenv("MT5_LOGIN")),"password":os.getenv("MT5_PASSWORD"),"server":os.getenv("MT5_SERVER"),"path":os.getenv("MT5_PATH")}
        except ValueError: logger.error("MT5_LOGIN no es entero.")

    db_m.initialize_database()
    bot = TradingBot(sym, tf, model_prefixes, mt5_creds)

    if not bot.models_info:
        logger.error("Ningún modelo se cargó para el ensemble. El bot no puede iniciar. "
                     "Asegúrate de haber ejecutado ml_engine.py para entrenar y guardar modelos "
                     f"con los prefijos: LGBM='{model_prefixes['lgbm']}', XGB='{model_prefixes['xgb']}'.")
    else:
        logger.info(f"Modelos cargados para ensemble: {[m['type'] for m in bot.models_info]}")
        cycles = int(os.getenv("BOT_NUM_CYCLES", "5"))
        interval = int(os.getenv("BOT_CYCLE_INTERVAL", "20"))
        try:
            for i in range(cycles):
                bot.run_cycle()
                if i < cycles - 1: logger.info(f"Esperando {interval}s..."); time.sleep(interval)
        except KeyboardInterrupt: logger.info("Interrupción.")
        finally: bot.stop()
