import time
import datetime
import logging
import pandas as pd
import numpy as np
import os
import json

import data_manager as dm
import ml_engine as mle
import database_manager as db_m

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("TradingBot")

def timeframe_to_timedelta(tf_str):
    if tf_str == "M1": return pd.Timedelta(minutes=1)
    if tf_str == "M5": return pd.Timedelta(minutes=5)
    if tf_str == "M15": return pd.Timedelta(minutes=15)
    if tf_str == "M30": return pd.Timedelta(minutes=30)
    if tf_str == "H1": return pd.Timedelta(hours=1)
    if tf_str == "H4": return pd.Timedelta(hours=4)
    if tf_str == "D1": return pd.Timedelta(days=1)
    return pd.Timedelta(hours=1)

class TradingBot:
    def __init__(self, symbol, timeframe_str, model_prefixes, mt5_conn_params=None):
        self.symbol = symbol
        self.timeframe_str = timeframe_str
        self.timeframe_delta = timeframe_to_timedelta(self.timeframe_str)
        self.model_prefixes = model_prefixes
        self.mt5_connected = False
        self.models_info = []
        self.last_data_timestamp = None
        self.trade_close_after_bars = int(os.getenv("BOT_TRADE_CLOSE_BARS", "5"))
        self.lgbm_model_key_for_update = os.getenv("BOT_LGBM_KEY_FOR_UPDATE", "lgbm")

        self.timeframe_map = {
            "M1":dm.TIMEFRAME_M1,"M5":dm.TIMEFRAME_M5,"M15":dm.TIMEFRAME_M15,"M30":dm.TIMEFRAME_M30,
            "H1":dm.TIMEFRAME_H1,"H4":dm.TIMEFRAME_H4,"D1":dm.TIMEFRAME_D1,"W1":dm.TIMEFRAME_W1,"MN1":dm.TIMEFRAME_MN1
        }
        self.timeframe_mt5 = self.timeframe_map.get(self.timeframe_str)
        if self.timeframe_mt5 is None: raise ValueError(f"Timeframe no reconocido: {self.timeframe_str}")
        if dm.MT5_AVAILABLE and not isinstance(self.timeframe_mt5, int): logger.warning(f"Timeframe {self.timeframe_str} podría no ser compatible con MT5.")

        if mt5_conn_params and mt5_conn_params.get('login'):
            self.mt5_connected = dm.initialize_mt5_connection(**mt5_conn_params)
            if not self.mt5_connected: logger.warning("No se pudo conectar a MT5.")
        else: logger.info("Parámetros MT5 no provistos.")
        self._load_all_models()

    def _load_all_models(self):
        self.models_info = []
        if not isinstance(self.model_prefixes, dict): logger.error("model_prefixes debe ser dict."); return
        for mk, pfx in self.model_prefixes.items():
            if not pfx: logger.info(f"Prefijo no dado para '{mk}'."); continue
            logger.info(f"Cargando modelo '{mk}' (prefijo: '{pfx}')...")
            m,e,fn = mle.load_model(pfx)
            if m and e:
                params_from_db = {}
                if e.parameters:
                    try: params_from_db = json.loads(e.parameters)
                    except json.JSONDecodeError: logger.warning(f"No se pudieron parsear parámetros JSON para {pfx}")
                self.models_info.append({'type':mk,'model':m,'feature_names':fn,'prefix':pfx,'version':e.version, 'params': params_from_db })
                logger.info(f"Modelo {mk} ({pfx} v{e.version}) cargado. Features: {fn or 'N/A'}")
            else: logger.error(f"Fallo al cargar modelo {mk} ({pfx}).")
        if not self.models_info: logger.error("ENSEMBLE: Ningún modelo cargado.")

    def _get_latest_market_data(self, num_bars=150):
        raw_data_df=None
        if self.mt5_connected:raw_data_df=dm.get_historical_data_mt5(self.symbol,self.timeframe_mt5,num_bars=num_bars)
        if raw_data_df is None:
            db_s=next(db_m.get_db_session())
            try:
                d_db=db_m.get_historical_data(db_s,self.symbol,self.timeframe_str,limit=num_bars,order_desc=True)
                if not d_db:logger.error(f"No datos BD para {self.symbol}/{self.timeframe_str}.");return None
                raw_data_df=pd.DataFrame([d.__dict__ for d in d_db]);raw_data_df.drop(columns=['_sa_instance_state'],errors='ignore',inplace=True)
                raw_data_df['timestamp']=pd.to_datetime(raw_data_df['timestamp']);raw_data_df.sort_values(by='timestamp',ascending=True,inplace=True);raw_data_df.reset_index(drop=True,inplace=True)
            finally:db_s.close()
        if raw_data_df is None or raw_data_df.empty:logger.error("No datos mercado.");return None
        df_i=raw_data_df.copy()
        if 'timestamp' in df_i.columns and not isinstance(df_i.index, pd.DatetimeIndex): # Ensure index is set for calculations
             df_i.set_index('timestamp', inplace=True)
        if not all(c in df_i.columns for c in ['high','low','close']):logger.error("Faltan OHLC.");return None
        df_i['SMA_10']=dm.calculate_sma(df_i['close'],10);df_i['RSI_14']=dm.calculate_rsi(df_i['close'],14)
        m_l,m_s,m_h=dm.calculate_macd(df_i['close']);df_i['MACD_line']=m_l;df_i['MACD_signal']=m_s;df_i['MACD_hist']=m_h
        bb_m,bb_u,bb_l=dm.calculate_bollinger_bands(df_i['close']);df_i['BB_Mid']=bb_m;df_i['BB_Upper']=bb_u;df_i['BB_Lower']=bb_l
        pdi,mdi,adx=dm.calculate_adx(df_i[['high','low','close']]);df_i['PLUS_DI']=pdi;df_i['MINUS_DI']=mdi;df_i['ADX']=adx
        df_i['return_1p']=df_i['close'].pct_change(1)*100
        return df_i.reset_index() # Return with timestamp as column

    def _check_and_close_trades(self, current_price_data):
        db_s = next(db_m.get_db_session())
        try:
            open_trades = db_m.get_open_trades(db_s, self.symbol)
            if not open_trades: return
            current_bar_timestamp = pd.to_datetime(current_price_data['timestamp'].iloc[0])
            current_close_price = current_price_data['close'].iloc[0]
            for trade in open_trades:
                trade_open_timestamp = pd.to_datetime(trade.timestamp)
                if current_bar_timestamp.tzinfo != trade_open_timestamp.tzinfo:
                    if current_bar_timestamp.tzinfo: current_bar_timestamp = current_bar_timestamp.tz_localize(None)
                    if trade_open_timestamp.tzinfo: trade_open_timestamp = trade_open_timestamp.tz_localize(None)
                time_since_open = current_bar_timestamp - trade_open_timestamp
                bars_since_open = time_since_open / self.timeframe_delta
                if bars_since_open >= self.trade_close_after_bars:
                    logger.info(f"Trade ID {trade.id} ({trade.order_type.value}) abierto por ~{bars_since_open:.1f} barras. Cerrando...")
                    profit = 0
                    if trade.order_type == db_m.OrderTypeEnum.BUY: profit = (current_close_price - trade.price) * trade.volume * 10000
                    elif trade.order_type == db_m.OrderTypeEnum.SELL: profit = (trade.price - current_close_price) * trade.volume * 10000
                    trade_outcome_target = -1
                    if trade.order_type == db_m.OrderTypeEnum.BUY: trade_outcome_target = 1 if current_close_price > trade.price else 0
                    elif trade.order_type == db_m.OrderTypeEnum.SELL: trade_outcome_target = 0 if current_close_price <= trade.price else 1
                    logger.info(f"Trade ID {trade.id} cerrado. P/L: {profit:.2f}. Outcome para modelo: {trade_outcome_target}")
                    update_data = {'status': db_m.TradeStatusEnum.SIMULATED_CLOSED,'profit': profit,'close_price': current_close_price,'close_timestamp': current_bar_timestamp}
                    db_m.update_trade(db_s, trade.id, update_data)
                    if trade.open_features_json and trade_outcome_target != -1:
                        lgbm_model_info = next((m for m in self.models_info if m['type'] == self.lgbm_model_key_for_update), None)
                        if lgbm_model_info:
                            try:
                                features_list = json.loads(trade.open_features_json)
                                if features_list:
                                    trade_features_df = pd.DataFrame(features_list)
                                    if lgbm_model_info['feature_names']:
                                        trade_features_df = trade_features_df[lgbm_model_info['feature_names']]
                                    current_model_params = lgbm_model_info.get('params', {})
                                    current_model_params.pop('feature_names', None)
                                    logger.info(f"Actualizando modelo LGBM {lgbm_model_info['prefix']} con trade ID {trade.id}...")
                                    mle.update_model_with_trade(lgbm_model_info['prefix'],trade_features_df,trade_outcome_target,'lgbm',current_model_params,lgbm_model_info['feature_names'])
                                    logger.info("Recargando modelo LGBM después de actualización...")
                                    upd_lgbm_m, upd_e, upd_f = mle.load_model(lgbm_model_info['prefix'])
                                    if upd_lgbm_m and upd_e:
                                        lgbm_model_info.update({'model':upd_lgbm_m, 'version':upd_e.version, 'feature_names':upd_f, 'params':json.loads(upd_e.parameters) if upd_e.parameters else {} })
                                        logger.info(f"Modelo LGBM {lgbm_model_info['prefix']} recargado a v{upd_e.version}.")
                                    else: logger.error(f"Fallo al recargar LGBM {lgbm_model_info['prefix']}.")
                            except Exception as e_upd: logger.error(f"Error procesando update por trade ID {trade.id}: {e_upd}")
                        else: logger.warning(f"No se encontró modelo LGBM '{self.lgbm_model_key_for_update}' para actualizar.")
        except Exception as e: logger.error(f"Error en _check_and_close_trades: {e}")
        finally: db_s.close()

    def run_cycle(self):
        if not self.models_info: logger.error("No hay modelos. Saltando ciclo."); return
        # logger.info(f"--- Ciclo Ensemble para {self.symbol}/{self.timeframe_str} ---") # Log más corto
        market_data_df = self._get_latest_market_data()
        if market_data_df is None or market_data_df.empty: logger.warning("No datos de mercado."); return
        latest_data_point = market_data_df.iloc[[-1]].copy()
        current_ts = pd.to_datetime(latest_data_point['timestamp'].iloc[0])
        self._check_and_close_trades(latest_data_point)
        if self.last_data_timestamp and self.last_data_timestamp == current_ts:
            if pd.Timestamp.now(tz='utc') - pd.Timestamp(self.last_data_timestamp, tz='utc' if getattr(self.last_data_timestamp, 'tzinfo', None) else None) < pd.Timedelta(seconds=30):
                 return
            return
        self.last_data_timestamp = current_ts
        individual_model_probas_p1 = []; all_preds_valid = True
        current_features_for_trade_decision = None
        for m_info in self.models_info:
            model_type=m_info['type'];model_obj=m_info['model'];expected_feats=m_info['feature_names']
            if not expected_feats: individual_model_probas_p1.append(0.5);continue
            try:
                current_features_df = latest_data_point[expected_feats]
                if current_features_for_trade_decision is None: current_features_for_trade_decision = current_features_df.copy()
            except KeyError as e: logger.error(f"ENSEMBLE: Faltan features para {model_type}: {e}.");individual_model_probas_p1.append(0.5);all_preds_valid=False;continue
            if current_features_df.isnull().values.any(): logger.warning(f"ENSEMBLE: NaNs en features para {model_type}. Usando proba neutral.");individual_model_probas_p1.append(0.5);all_preds_valid=False;continue
            _,pred_probas_dict = mle.make_prediction(model_obj,current_features_df,expected_feats)
            if pred_probas_dict and 1 in pred_probas_dict: individual_model_probas_p1.append(pred_probas_dict[1])
            else: logger.warning(f"ENSEMBLE: No P(1) de {model_type}. Usando proba neutral.");individual_model_probas_p1.append(0.5);all_preds_valid=False
        if not individual_model_probas_p1 or not all_preds_valid: logger.error("ENSEMBLE: No todas las preds individuales válidas."); return
        ensemble_weights = [0.5,0.5] if len(individual_model_probas_p1)==2 else None
        final_pred_class, final_proba_p1 = mle.make_ensemble_prediction(individual_model_probas_p1, weights=ensemble_weights)
        if final_pred_class is None: logger.error("ENSEMBLE: Fallo predicción ensemble."); return
        logger.info(f"ENSEMBLE RESULT ({self.symbol} {current_ts.strftime('%Y-%m-%d %H:%M')}): Clase={final_pred_class}, P(1)={final_proba_p1:.3f}")
        signal_type_enum = None; conf_thresh = float(os.getenv("BOT_ENSEMBLE_CONFIDENCE_THRESHOLD", "0.55"))
        if final_pred_class == 1 and final_proba_p1 >= conf_thresh: signal_type_enum = db_m.OrderTypeEnum.BUY
        elif final_pred_class == 0 and (1 - final_proba_p1) >= conf_thresh: signal_type_enum = db_m.OrderTypeEnum.SELL
        if signal_type_enum:
            logger.info(f"DECISIÓN ENSEMBLE: {signal_type_enum.value} (P(1)={final_proba_p1:.3f})")
            self._execute_simulated_trade(signal_type_enum, latest_data_point['close'].iloc[0], 0.1, current_features_for_trade_decision)
        # else: logger.info(f"DECISIÓN ENSEMBLE: HOLD (P(1)={final_proba_p1:.3f})") # Log HOLD si es necesario

    def _execute_simulated_trade(self, signal_type, price, volume, features_df_for_trade):
        features_json = None
        if features_df_for_trade is not None and not features_df_for_trade.empty:
            try: features_json = features_df_for_trade.to_json(orient='records')
            except Exception as e: logger.error(f"Error convirtiendo features a JSON: {e}")
        trade_data = {'timestamp':datetime.datetime.now(datetime.timezone.utc),'symbol':self.symbol,'order_type':signal_type,
                      'price':price,'volume':volume,'status':db_m.TradeStatusEnum.SIMULATED_OPEN, 'open_features_json':features_json}
        db_s = next(db_m.get_db_session())
        try:
            entry=db_m.add_trade(db_s,trade_data)
            if entry: logger.info(f"Trade simulado (OPEN): ID {entry.id} {signal_type.value} @{price:.5f}. Features: {'Sí' if features_json else 'No'}")
            else: logger.error("Fallo al registrar trade simulado.")
        finally: db_s.close()

    def stop(self):
        logger.info("Deteniendo bot...");
        if self.mt5_connected: dm.shutdown_mt5_connection()
        logger.info("Bot detenido.")

if __name__ == "__main__":
    logger.info("Iniciando Trading Bot (con Cierre de Trades y Aprendizaje por Trade)...")
    sym=os.getenv("BOT_SYMBOL","SIM_EURUSD");tf=os.getenv("BOT_TIMEFRAME","H1")

    # --- ACTUALIZACIÓN DE SUFIJOS DE MODELO ---
    model_prefixes = {
        'lgbm': os.getenv("BOT_LGBM_MODEL_PREFIX", f"LGBM_{sym}_{tf}_OptunaTestV4"), # Usar V4
        'xgb': os.getenv("BOT_XGB_MODEL_PREFIX", f"XGB_{sym}_{tf}_OptunaTestV4")    # Usar V4
    }
    # --- FIN DE ACTUALIZACIÓN ---

    logger.info(f"Bot: Symbol={sym}, TF={tf}, LGBM Prefix={model_prefixes['lgbm']}, XGB Prefix={model_prefixes['xgb']}")
    mt5_creds = None
    if os.getenv("MT5_LOGIN"):
        try: mt5_creds = {"login":int(os.getenv("MT5_LOGIN")),"password":os.getenv("MT5_PASSWORD"),"server":os.getenv("MT5_SERVER"),"path":os.getenv("MT5_PATH")}
        except ValueError: logger.error("MT5_LOGIN no es entero.")
    db_m.initialize_database()
    bot = TradingBot(sym, tf, model_prefixes, mt5_creds)
    if not bot.models_info:
        logger.error(f"Ningún modelo cargado. Asegúrate de ejecutar ml_engine.py para los prefijos: {model_prefixes}")
    else:
        logger.info(f"Modelos cargados para ensemble: {[m['type'] for m in bot.models_info]}")
        cycles = int(os.getenv("BOT_NUM_CYCLES", "20")) # Aumentar ciclos para ver más acción
        interval = int(os.getenv("BOT_CYCLE_INTERVAL", "3")) # Intervalo más corto
        logger.info(f"Ejecutando {cycles} ciclos, intervalo {interval}s. Cierre trades tras ~{bot.trade_close_after_bars} barras.")
        try:
            for i in range(cycles):
                logger.info(f"--- Ciclo Principal {i+1}/{cycles} ---")
                bot.run_cycle()
                if i < cycles - 1: logger.info(f"Esperando {interval}s..."); time.sleep(interval)
        except KeyboardInterrupt: logger.info("Interrupción.")
        finally: bot.stop()
