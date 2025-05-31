import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score
import os
import datetime
import json

import database_manager as db_m
import data_manager as dm

MODELS_DIR = "trained_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def prepare_features(df, target_period=1, indicators_to_use=None, feature_lags=None):
    # ... (código de prepare_features sin cambios) ...
    if df.empty: return pd.DataFrame(), pd.Series(), df, []
    df_processed = df.copy()
    if not isinstance(df_processed.index, pd.DatetimeIndex) and 'timestamp' in df_processed.columns:
        df_processed.set_index('timestamp', inplace=True)
    elif not isinstance(df_processed.index, pd.DatetimeIndex):
        raise ValueError("DataFrame debe tener un índice DatetimeIndex o una columna 'timestamp'.")
    df_processed.sort_index(inplace=True)
    df_processed['future_close'] = df_processed['close'].shift(-target_period)
    feature_columns = []
    if indicators_to_use:
        for indicator in indicators_to_use:
            if indicator in df_processed.columns: feature_columns.append(indicator)
            else: print(f"Advertencia: Indicador '{indicator}' no encontrado.")
    df_processed['return_1p'] = df_processed['close'].pct_change(1) * 100
    if 'return_1p' not in feature_columns: feature_columns.append('return_1p')
    if feature_lags:
        for col, lags_list in feature_lags.items():
            if col in df_processed.columns:
                for lag in lags_list:
                    lag_col_name = f'{col}_lag_{lag}'
                    df_processed[lag_col_name] = df_processed[col].shift(lag)
                    if lag_col_name not in feature_columns: feature_columns.append(lag_col_name)
            else: print(f"Advertencia: Columna '{col}' para lags no encontrada.")
    df_processed['target'] = (df_processed['future_close'] > df_processed['close']).astype(int)
    df_processed.dropna(subset=feature_columns + ['target', 'future_close'], inplace=True)
    if df_processed.empty or not feature_columns:
        print("No hay datos/features después del preprocesamiento y dropna.")
        return pd.DataFrame(), pd.Series(), df_processed, []
    X = df_processed[feature_columns]; y = df_processed['target']
    return X, y, df_processed, feature_columns

def optimize_hyperparameters(model_type, X_train, y_train, X_val, y_val, n_trials=25, study_name=None):
    # ... (código de optimize_hyperparameters sin cambios) ...
    if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
        print(f"Optuna ({model_type}): Datos vacíos. Saltando optimización.")
        return {}
    print(f"Optuna ({model_type}): Iniciando optimización con {n_trials} trials...")
    def objective(trial):
        if model_type == 'lgbm':
            lgbm_params = {'objective':'binary','metric':'binary_logloss','verbosity':-1,'boosting_type':'gbdt',
                           'n_estimators':trial.suggest_int('n_estimators',100,1000,step=50),'learning_rate':trial.suggest_float('learning_rate',0.01,0.2,log=True),
                           'num_leaves':trial.suggest_int('num_leaves',20,150),'max_depth':trial.suggest_int('max_depth',3,12),
                           'min_child_samples':trial.suggest_int('min_child_samples',5,100),'feature_fraction':trial.suggest_float('feature_fraction',0.5,1.0),
                           'bagging_fraction':trial.suggest_float('bagging_fraction',0.5,1.0),'bagging_freq':trial.suggest_int('bagging_freq',1,7),
                           'lambda_l1':trial.suggest_float('lambda_l1',1e-8,10.0,log=True),'lambda_l2':trial.suggest_float('lambda_l2',1e-8,10.0,log=True)}
            model = lgb.LGBMClassifier(**lgbm_params)
            model.fit(X_train,y_train,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(10,verbose=False)])
        elif model_type == 'xgb':
            xgb_params = {'objective':'binary:logistic','eval_metric':'logloss','use_label_encoder':False,
                          'n_estimators':trial.suggest_int('n_estimators',100,1000,step=50),'learning_rate':trial.suggest_float('learning_rate',0.01,0.2,log=True),
                          'max_depth':trial.suggest_int('max_depth',3,12),'min_child_weight':trial.suggest_int('min_child_weight',1,10),
                          'gamma':trial.suggest_float('gamma',1e-8,1.0,log=True),'subsample':trial.suggest_float('subsample',0.5,1.0),
                          'colsample_bytree':trial.suggest_float('colsample_bytree',0.5,1.0),'reg_alpha':trial.suggest_float('reg_alpha',1e-8,10.0,log=True),
                          'reg_lambda':trial.suggest_float('reg_lambda',1e-8,10.0,log=True)}
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=10,verbose=False)
        else: raise ValueError("model_type debe ser 'lgbm' o 'xgb'")
        preds_proba_val = model.predict_proba(X_val)[:,1]; return log_loss(y_val,preds_proba_val)
    study = optuna.create_study(direction='minimize',study_name=study_name or f"{model_type}_opt")
    try: study.optimize(objective,n_trials=n_trials,timeout=600)
    except Exception as e: print(f"Optuna ({model_type}): Excepción: {e}")
    print(f"Optuna ({model_type}): Mejor valor (log_loss): {study.best_value:.4f if study.best_value else 'N/A'}, Params: {study.best_params}")
    return study.best_params

def train_lgbm_model(X, y, params=None, initial_model_path=None, X_test_override=None, y_test_override=None):
    # ... (código existente) ...
    if X.empty or y.empty: print("LGBM: No hay datos."); return None, {}, {}
    X_train, y_train, X_test, y_test = (X,y, X_test_override, y_test_override) if X_test_override is not None else                                      (X.iloc[:int(len(X)*0.8)], y.iloc[:int(len(y)*0.8)], X.iloc[int(len(X)*0.8):], y.iloc[int(len(y)*0.8):]) if len(X) >= 50 else (X,y,X,y)
    if X_train.empty or X_test.empty : print("LGBM: Train/Test sets vacíos."); return None, {}, {}
    lgbm_base_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'n_estimators':100}
    if params: lgbm_base_params.update(params)
    else: lgbm_base_params.update({'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9})
    init_model_obj = None
    if initial_model_path:
        try: init_model_obj = joblib.load(initial_model_path).booster_ if initial_model_path.endswith(".joblib") else initial_model_path
        except Exception as e: print(f"LGBM: Error cargando modelo inicial: {e}"); init_model_obj = None
    model = lgb.LGBMClassifier(**lgbm_base_params)
    callbacks_list = [lgb.early_stopping(10, verbose=False)] if not X_test.equals(X_train) else []
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)] if not X_test.equals(X_train) else None,
              eval_metric=lgbm_base_params.get('metric'), callbacks=callbacks_list, init_model=init_model_obj)
    metrics = {"train_accuracy": accuracy_score(y_train, model.predict(X_train))}
    if not X_test.empty and not X_test.equals(X_train): # Asegurar que X_test no está vacío
        metrics.update({"test_accuracy": accuracy_score(y_test, model.predict(X_test)),
                        "test_precision": precision_score(y_test, model.predict(X_test), zero_division=0),
                        "test_recall": recall_score(y_test, model.predict(X_test), zero_division=0),
                        "test_logloss": log_loss(y_test, model.predict_proba(X_test)[:,1]),
                        "test_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) })
    print(f"Métricas LGBM (post-entrenamiento): {metrics if metrics else 'N/A'}"); return model, lgbm_base_params, metrics

def train_xgb_model(X, y, params=None, X_test_override=None, y_test_override=None):
    # ... (código existente) ...
    if X.empty or y.empty: print("XGB: No hay datos."); return None, {}, {}
    X_train, y_train, X_test, y_test = (X,y, X_test_override, y_test_override) if X_test_override is not None else                                      (X.iloc[:int(len(X)*0.8)], y.iloc[:int(len(y)*0.8)], X.iloc[int(len(X)*0.8):], y.iloc[int(len(y)*0.8):]) if len(X) >= 50 else (X,y,X,y)
    if X_train.empty or X_test.empty : print("XGB: Train/Test sets vacíos."); return None, {}, {}
    xgb_base_params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 100, 'use_label_encoder': False}
    if params: xgb_base_params.update(params)
    else: xgb_base_params.update({'learning_rate': 0.1, 'max_depth': 6})
    model = xgb.XGBClassifier(**xgb_base_params)
    fit_params = {'verbose': False}
    if not X_test.empty and not X_test.equals(X_train): # Asegurar que X_test no está vacío
        fit_params['eval_set'] = [(X_test, y_test)]
        fit_params['early_stopping_rounds'] = 10
    model.fit(X_train, y_train, **fit_params)
    metrics = {"train_accuracy": accuracy_score(y_train, model.predict(X_train))}
    if not X_test.empty and not X_test.equals(X_train): # Asegurar que X_test no está vacío
        metrics.update({"test_accuracy": accuracy_score(y_test, model.predict(X_test)),
                        "test_precision": precision_score(y_test, model.predict(X_test), zero_division=0),
                        "test_recall": recall_score(y_test, model.predict(X_test), zero_division=0),
                        "test_logloss": log_loss(y_test, model.predict_proba(X_test)[:,1]),
                        "test_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) })
    print(f"Métricas XGB (post-entrenamiento): {metrics if metrics else 'N/A'}"); return model, xgb_base_params, metrics

def save_model(model, model_name_prefix, params_dict, metrics_dict, feature_names=None):
    # ... (código existente) ...
    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S");v=f"v{ts}";name_v=f"{model_name_prefix}_{v}";path_joblib=os.path.join(MODELS_DIR,f"{name_v}.joblib")
    path_txt=os.path.join(MODELS_DIR,f"{name_v}.txt") if isinstance(model,lgb.LGBMClassifier) else None
    try:
        joblib.dump(model,path_joblib)
        if path_txt:model.booster_.save_model(path_txt)
        print(f"Modelo guardado: {path_joblib}{'' if not path_txt else ' y '+path_txt}")
        db_s=next(db_m.get_db_session())
        try:
            p_save=params_dict.copy() if params_dict else {}; p_save['feature_names']=feature_names or []
            entry=db_m.add_model_version(db_s,model_name_prefix,v,path_joblib,json.dumps(p_save),json.dumps(metrics_dict))
            return path_joblib,entry
        except Exception as e_db:print(f"Error BD: {e_db}");return path_joblib,None
        finally:db_s.close()
    except Exception as e_save:print(f"Error guardando: {e_save}");return None,None

def load_model(model_name_prefix, version_str=None):
    # ... (código existente) ...
    db_s=next(db_m.get_db_session());entry=None;feats=None
    try:
        entry=db_s.query(db_m.ModelVersion).filter(db_m.ModelVersion.model_name==model_name_prefix,db_m.ModelVersion.version==version_str).first() if version_str else db_m.get_latest_model_version(db_s,model_name_prefix)
        if entry and entry.file_path and os.path.exists(entry.file_path):
            model=joblib.load(entry.file_path)
            if entry.parameters:
                try: feats=json.loads(entry.parameters).get('feature_names')
                except json.JSONDecodeError: pass # No hacer nada si falla el parseo
            print(f"Modelo cargado: {entry.file_path}");return model,entry,feats
        elif entry:print(f"Archivo no encontrado: {entry.file_path}");return None,entry,None
        else:print(f"Modelo no encontrado en BD: {model_name_prefix} v:{version_str or 'latest'}");return None,None,None
    except Exception as e:print(f"Error cargando: {e}");return None,None,None
    finally:db_s.close()

def make_prediction(model, df_features, expected_feats=None):
    # ... (código existente) ...
    if model is None or df_features.empty : return None, None
    try:
        if expected_feats and list(df_features.columns) != expected_feats:
            try: df_features = df_features[expected_feats]
            except KeyError as e: print(f"Error features: {e}"); return None, None
        pred = model.predict(df_features); proba = model.predict_proba(df_features)
        return pred[0], {0: proba[0][0], 1: proba[0][1]}
    except Exception as e: print(f"Error predicción: {e}"); return None, None

def make_ensemble_prediction(model_predictions_proba, weights=None):
    # ... (código existente) ...
    if not model_predictions_proba: print("Ensemble: No hay predicciones."); return None, None
    probas_class_1 = []
    for prob_input in model_predictions_proba:
        if isinstance(prob_input, dict): probas_class_1.append(prob_input.get(1, 0.0))
        else: probas_class_1.append(prob_input)
    num_models = len(probas_class_1)
    if weights is None: weights = [1/num_models] * num_models
    elif len(weights) != num_models: print("Ensemble: Pesos no coinciden. Usando promedio."); weights = [1/num_models] * num_models
    elif not np.isclose(sum(weights), 1.0): print("Ensemble: Pesos no suman 1. Normalizando."); weights = [w / sum(weights) for w in weights]
    final_proba_class_1 = np.average(probas_class_1, weights=weights)
    final_class = 1 if final_proba_class_1 >= 0.5 else 0
    return final_class, final_proba_class_1

def retrain_all_models(symbol, timeframe, indicators_for_model, feature_lags=None,
                       use_optuna=False, optuna_trials=15, min_data_for_optuna=150,
                       lgbm_base_params=None, xgb_base_params=None):
    # ... (código existente) ...
    print(f"\n--- Iniciando Proceso de Re-entrenamiento para {symbol}/{timeframe} ---")
    print(f"Usar Optuna: {use_optuna}, Trials de Optuna: {optuna_trials if use_optuna else 'N/A'}")
    db_s = next(db_m.get_db_session())
    try: all_historical_data_db = db_m.get_historical_data(db_s, symbol, timeframe, limit=3000, order_desc=False)
    finally: db_s.close()
    if not all_historical_data_db or len(all_historical_data_db) < min_data_for_optuna :
        print(f"Datos insuficientes en BD ({len(all_historical_data_db) if all_historical_data_db else 0}) para re-entrenamiento. Se necesitan >={min_data_for_optuna}."); return False
    print(f"Re-entrenamiento: Usando {len(all_historical_data_db)} registros de la BD.")
    df_all_hist = pd.DataFrame([d.__dict__ for d in all_historical_data_db]); df_all_hist.drop(columns=['_sa_instance_state'], errors='ignore', inplace=True)
    df_all_hist['timestamp'] = pd.to_datetime(df_all_hist['timestamp'])
    df_indic = df_all_hist.set_index('timestamp') if 'timestamp' in df_all_hist.columns else df_all_hist.copy()
    base_indicators = ['SMA_10', 'RSI_14', 'MACD_line', 'MACD_signal', 'BB_Mid', 'ADX', 'PLUS_DI', 'MINUS_DI']
    calc_indicators = list(set(base_indicators) & set(indicators_for_model + ['close','high','low']))
    if 'SMA_10' in calc_indicators: df_indic['SMA_10'] = dm.calculate_sma(df_indic['close'],10)
    if 'RSI_14' in calc_indicators: df_indic['RSI_14'] = dm.calculate_rsi(df_indic['close'],14)
    if 'MACD_line' in calc_indicators or 'MACD_signal' in calc_indicators: m_l,m_s,_=dm.calculate_macd(df_indic['close']); df_indic['MACD_line']=m_l; df_indic['MACD_signal']=m_s
    if 'BB_Mid' in calc_indicators: bb_m,_,_=dm.calculate_bollinger_bands(df_indic['close']); df_indic['BB_Mid']=bb_m
    if 'ADX' in calc_indicators or 'PLUS_DI' in calc_indicators or 'MINUS_DI' in calc_indicators: pdi,mdi,adx_val=dm.calculate_adx(df_indic[['high','low','close']]); df_indic['ADX']=adx_val; df_indic['PLUS_DI']=pdi; df_indic['MINUS_DI']=mdi
    df_full_for_features = df_indic.reset_index()
    X_full, y_full, _, all_feat_names = prepare_features(df_full_for_features.copy(), 1, indicators_for_model, feature_lags)
    if X_full.empty or y_full.empty: print("Re-entrenamiento: No se generaron features. Abortando."); return False
    final_lgbm_params = lgbm_base_params or {}; final_xgb_params = xgb_base_params or {}
    if use_optuna:
        if len(X_full) < min_data_for_optuna + 50 : print(f"Re-entrenamiento: Datos insuficientes ({len(X_full)}) para Optuna. Saltando.")
        else:
            X_opt_tv, _, y_opt_tv, _ = train_test_split(X_full, y_full, test_size=0.2, shuffle=False)
            X_otr, X_ov, y_otr, y_ov = train_test_split(X_opt_tv, y_opt_tv, test_size=0.3, shuffle=False)
            if not X_otr.empty and not X_ov.empty:
                print(f"Re-entrenamiento: Optuna LGBM sobre {len(X_otr)} train / {len(X_ov)} val.")
                best_lgbm = optimize_hyperparameters('lgbm',X_otr,y_otr,X_ov,y_ov,optuna_trials,f"LGBM_Retrain_{symbol}_{timeframe}")
                if best_lgbm: final_lgbm_params.update(best_lgbm)
                print(f"Re-entrenamiento: Optuna XGB sobre {len(X_otr)} train / {len(X_ov)} val.")
                best_xgb = optimize_hyperparameters('xgb',X_otr,y_otr,X_ov,y_ov,optuna_trials,f"XGB_Retrain_{symbol}_{timeframe}")
                if best_xgb: final_xgb_params.update(best_xgb)
            else: print("Re-entrenamiento: Datos insuficientes para split de Optuna (otrain/oval). Saltando Optuna.")
    print(f"Re-entrenamiento LGBM final con params: {final_lgbm_params}")
    lgbm_model,_,lgbm_met=train_lgbm_model(X_full,y_full,final_lgbm_params);
    if lgbm_model:save_model(lgbm_model,f"LGBM_{symbol}_{timeframe}_Retrained",final_lgbm_params,lgbm_met,all_feat_names)
    else:print(f"Re-entrenamiento LGBM falló.")
    print(f"Re-entrenamiento XGB final con params: {final_xgb_params}")
    xgb_model,_,xgb_met=train_xgb_model(X_full,y_full,final_xgb_params);
    if xgb_model:save_model(xgb_model,f"XGB_{symbol}_{timeframe}_Retrained",final_xgb_params,xgb_met,all_feat_names)
    else:print(f"Re-entrenamiento XGB falló.")
    print(f"--- Proceso de Re-entrenamiento para {symbol}/{timeframe} completado ---"); return True

# --- NUEVA FUNCIÓN PARA ACTUALIZAR MODELO CON UN TRADE ---
def update_model_with_trade(model_prefix_to_update, trade_features_df, trade_outcome_target,
                            model_type='lgbm', current_model_params=None, feature_names=None):
    """
    Actualiza un modelo (principalmente LGBM) de forma incremental con el resultado de un nuevo trade.
    :param model_prefix_to_update: Prefijo del modelo a cargar y actualizar.
    :param trade_features_df: DataFrame (1 fila) con las features del momento del trade.
    :param trade_outcome_target: El target (0 o 1) real observado para ese trade.
    :param model_type: Tipo de modelo ('lgbm' o 'xgb'). Actualmente solo LGBM es bien soportado para esto.
    :param current_model_params: Parámetros actuales del modelo (para pasarlos al re-entrenamiento).
    :param feature_names: Lista de nombres de features.
    :return: Path al modelo actualizado, o None si falla.
    """
    if model_type.lower() != 'lgbm':
        print(f"Actualización por trade actualmente solo soportada para LGBM. Modelo '{model_type}' no actualizado.")
        return None

    if trade_features_df.empty or not isinstance(trade_outcome_target, int):
        print("Actualización por trade: Features vacías o target inválido.")
        return None

    print(f"Actualización por trade para {model_prefix_to_update}...")
    loaded_model, model_entry, loaded_feat_names = load_model(model_prefix_to_update)
    if not loaded_model or not model_entry:
        print(f"No se pudo cargar el modelo {model_prefix_to_update} para la actualización por trade.")
        return None

    active_feature_names = feature_names if feature_names else loaded_feat_names
    if not active_feature_names:
        print("No se pudieron determinar los nombres de las features para la actualización por trade.")
        return None

    try:
        X_update = trade_features_df[active_feature_names]
    except KeyError:
        print(f"Features en trade_features_df no coinciden con las esperadas: {active_feature_names}")
        return None

    y_update = pd.Series([trade_outcome_target], index=X_update.index)

    update_params = current_model_params.copy() if current_model_params else {}
    update_params['learning_rate'] = float(os.getenv("TRADE_UPDATE_LR", "0.005"))
    update_params['n_estimators'] = int(os.getenv("TRADE_UPDATE_ESTIMATORS", "5"))
    update_params['bagging_fraction'] = 1.0
    update_params['feature_fraction'] = 1.0

    initial_model_file_path = model_entry.file_path.replace(".joblib", ".txt")
    if not os.path.exists(initial_model_file_path):
        print(f"Archivo de texto del modelo no encontrado ({initial_model_file_path}). Guardando actual en texto.")
        try:
            loaded_model.booster_.save_model(initial_model_file_path)
            print(f"Modelo de texto guardado en {initial_model_file_path}")
        except Exception as e_save_txt:
            print(f"Error al guardar modelo de texto: {e_save_txt}. No se puede actualizar."); return None

    print(f"Entrenando incrementalmente {model_prefix_to_update} con 1 trade. Params: {update_params}")
    updated_model, _, updated_metrics = train_lgbm_model(
        X_update, y_update, params=update_params, initial_model_path=initial_model_file_path,
        X_test_override=X_update, y_test_override=y_update
    )

    if updated_model:
        print(f"Modelo actualizado por trade. Métricas (sobre muestra del trade): {updated_metrics}")
        saved_path, _ = save_model(updated_model, model_prefix_to_update, update_params, updated_metrics, active_feature_names)
        if saved_path:
            print(f"Modelo actualizado por trade guardado como nueva versión: {saved_path}")
            return saved_path
    else: print(f"Fallo al actualizar incrementalmente el modelo {model_prefix_to_update} con el trade.")
    return None


if __name__ == "__main__":
    print("Ejecutando ml_engine como script principal para pruebas...")
    db_m.initialize_database()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n--- 1. Preparando Datos para Optimización y Entrenamiento ---")
    test_symbol = os.getenv("BOT_SYMBOL", "SIM_EURUSD")
    test_timeframe = os.getenv("BOT_TIMEFRAME", "H1")
    num_opt_trials = int(os.getenv("OPTUNA_TRIALS", "10"))
    run_incremental_lgbm = os.getenv("RUN_INCREMENTAL_LGBM", "true").lower() == "true"
    run_full_retrain_test = os.getenv("RUN_FULL_RETRAIN_TEST", "true").lower() == "true"
    run_trade_update_test = os.getenv("RUN_TRADE_UPDATE_TEST", "true").lower() == "true" # Nueva flag

    db_s = next(db_m.get_db_session()); db_data = db_m.get_historical_data(db_s, test_symbol, test_timeframe, 700, order_desc=False); db_s.close()
    historical_df = None
    if db_data and len(db_data) >= 300:
        print(f"Usando {len(db_data)} registros de BD para {test_symbol}/{test_timeframe}")
        historical_df = pd.DataFrame([d.__dict__ for d in db_data]); historical_df.drop(columns=['_sa_instance_state'], errors='ignore', inplace=True)
        historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
    else:
        print(f"Datos BD insuficientes ({len(db_data) if db_data else 0}). Generando más datos simulados...")
        num_pts = 700; dates = pd.to_datetime([datetime.datetime(2021,1,1)+datetime.timedelta(hours=i) for i in range(num_pts)])
        s_data = {'timestamp': dates, 'open': np.random.rand(num_pts)*20+1.0, 'volume': np.random.randint(100,2000,num_pts)}
        s_data['high']=s_data['open']+np.random.rand(num_pts)*0.01; s_data['low']=s_data['open']-np.random.rand(num_pts)*0.01
        s_data['close']=s_data['open']+(np.random.rand(num_pts)-0.5)*0.02
        s_data['low']=np.minimum.reduce([s_data['low'],s_data['open'],s_data['close']]); s_data['high']=np.maximum.reduce([s_data['high'],s_data['open'],s_data['close']])
        historical_df = pd.DataFrame(s_data)

    df_indic = historical_df.set_index('timestamp') if 'timestamp' in historical_df.columns else historical_df.copy()
    features_config_indicators = ['SMA_10', 'RSI_14', 'MACD_line', 'MACD_signal', 'BB_Mid', 'ADX', 'PLUS_DI', 'MINUS_DI']
    for col in features_config_indicators: df_indic[col] = np.nan
    df_indic['SMA_10'] = dm.calculate_sma(df_indic['close'],10); df_indic['RSI_14'] = dm.calculate_rsi(df_indic['close'],14)
    m_l,m_s,_=dm.calculate_macd(df_indic['close']); df_indic['MACD_line']=m_l; df_indic['MACD_signal']=m_s
    bb_m,_,_=dm.calculate_bollinger_bands(df_indic['close']); df_indic['BB_Mid']=bb_m
    pdi,mdi,adx_val=dm.calculate_adx(df_indic[['high','low','close']]); df_indic['ADX']=adx_val; df_indic['PLUS_DI']=pdi; df_indic['MINUS_DI']=mdi
    hist_df_indic = df_indic.reset_index()
    features_config_lags = None
    print(f"Barras post-indicadores: {len(hist_df_indic)}")
    if hist_df_indic.empty or len(hist_df_indic) < 150: print(f"Datos insuficientes ({len(hist_df_indic)}) para pruebas."); exit()

    X_all, y_all, _, feat_names_all = prepare_features(hist_df_indic.copy(), 1, features_config_indicators, features_config_lags)
    if X_all.empty: print("No hay datos X,y. Saliendo."); exit()

    idx_opt_end = int(len(X_all) * 0.6); X_opt_base = X_all.iloc[:idx_opt_end]; y_opt_base = y_all.iloc[:idx_opt_end]
    X_incremental_data = X_all.iloc[idx_opt_end:]; y_incremental_data = y_all.iloc[idx_opt_end:]
    if len(X_opt_base) < 50: print(f"Datos insuficientes para optimización ({len(X_opt_base)})."); exit()
    X_otrain, X_oval, y_otrain, y_oval = train_test_split(X_opt_base, y_opt_base, test_size=0.3, shuffle=False)
    print(f"Optuna: Train set ({len(X_otrain)}), Val set ({len(X_oval)})")

    print("\n--- 2. Optimización y Entrenamiento LGBM ---")
    best_lgbm_p = optimize_hyperparameters('lgbm', X_otrain, y_otrain, X_oval, y_oval, num_opt_trials, f"LGBM_OptMain_{test_symbol}")
    prefix_lgbm = f"LGBM_{test_symbol}_{test_timeframe}_OptunaTest"
    path_init_lgbm, opt_lgbm_model, lgbm_final_params = None, None, best_lgbm_p.copy() if best_lgbm_p else {}
    if best_lgbm_p:
        opt_lgbm_model, lgbm_final_params, met_lgbm = train_lgbm_model(X_opt_base, y_opt_base, best_lgbm_p)
        if opt_lgbm_model: path_init_lgbm, _ = save_model(opt_lgbm_model, prefix_lgbm, lgbm_final_params, met_lgbm, feat_names_all)
    else: print("No mejores params LGBM.")

    if run_incremental_lgbm and path_init_lgbm and not X_incremental_data.empty:
        print("\n--- 3. Aprendizaje Incremental LGBM ---") # ... (similar a antes)
        path_txt_lgbm = path_init_lgbm.replace(".joblib",".txt")
        if os.path.exists(path_txt_lgbm):
            p_inc_lgbm = lgbm_final_params.copy(); p_inc_lgbm['learning_rate'] = lgbm_final_params.get('learning_rate',0.05)*0.5
            X_inc_tr,y_inc_tr,X_inc_ev,y_inc_ev = train_test_split(X_incremental_data,y_incremental_data,test_size=0.3,shuffle=False) if len(X_incremental_data)>=50 else (X_incremental_data,y_incremental_data,None,None)
            if not X_inc_tr.empty:
                inc_lgbm, p_i_l, m_i_l = train_lgbm_model(X_inc_tr,y_inc_tr,p_inc_lgbm,path_txt_lgbm,X_inc_ev,y_inc_ev)
                if inc_lgbm: save_model(inc_lgbm,prefix_lgbm,p_i_l,m_i_l,feat_names_all); opt_lgbm_model=inc_lgbm; lgbm_final_params = p_i_l
            else: print("No datos train LGBM inc.")
        else: print(f"No .txt para LGBM inc: {path_txt_lgbm}")
    else: print("LGBM optimizado no guardado o no datos inc, saltando inc. LGBM.")

    print("\n--- 4. Optimización y Entrenamiento XGBoost ---") # ... (similar a antes)
    best_xgb_p = optimize_hyperparameters('xgb', X_otrain, y_otrain, X_oval, y_oval, num_opt_trials, f"XGB_OptMain_{test_symbol}")
    prefix_xgb = f"XGB_{test_symbol}_{test_timeframe}_OptunaTest"
    opt_xgb_model, xgb_final_params = None, best_xgb_p.copy() if best_xgb_p else {}
    if best_xgb_p:
        opt_xgb_model, xgb_final_params, met_xgb = train_xgb_model(X_opt_base, y_opt_base, best_xgb_p)
        if opt_xgb_model: save_model(opt_xgb_model, prefix_xgb, xgb_final_params, met_xgb, feat_names_all)
    else: print("No mejores params XGB.")

    print("\n--- 5. Pruebas Ensemble ---") # ... (similar a antes)
    if opt_lgbm_model and opt_xgb_model and not X_all.empty:
        sample_ens=X_all.iloc[[-1]]; _,lgbm_d=make_prediction(opt_lgbm_model,sample_ens,feat_names_all); _,xgb_d=make_prediction(opt_xgb_model,sample_ens,feat_names_all)
        if lgbm_d and xgb_d: print(f"LGBM P(1):{lgbm_d[1]:.4f}, XGB P(1):{xgb_d[1]:.4f}");c,p1=make_ensemble_prediction([lgbm_d[1],xgb_d[1]]);print(f"Ens Avg:C={c},P(1)={p1:.4f}")

    if run_full_retrain_test: # ... (similar a antes)
        print("\n--- 6. Prueba de Re-entrenamiento Completo (con Optuna) ---")
        retrain_indicators = ['SMA_10', 'RSI_14', 'ADX', 'PLUS_DI', 'MINUS_DI']
        retrain_all_models(test_symbol,test_timeframe,retrain_indicators,None,True,5,100)
        print("\n--- Prueba de Re-entrenamiento (sin Optuna) ---")
        retrain_all_models(test_symbol,test_timeframe,retrain_indicators,None,False)
    else: print("\nSaltando prueba de Re-entrenamiento Completo.")

    if run_trade_update_test:
        print("\n--- 7. Prueba de Actualización de Modelo por Trade (LGBM) ---")
        latest_lgbm, lgbm_entry, lgbm_feats = load_model(prefix_lgbm) # Cargar el LGBM más reciente (puede ser el incremental)
        if latest_lgbm and not X_all.empty and lgbm_feats:
            sim_trade_feats = X_all.iloc[[-1]][lgbm_feats]
            sim_trade_target = y_all.iloc[-1] # Usar el target real de la última muestra para simular

            print(f"Actualizando modelo {prefix_lgbm} (v{lgbm_entry.version if lgbm_entry else 'N/A'}) con 1 trade.")
            print(f"Features del trade:\n{sim_trade_feats}\nTarget del trade: {sim_trade_target}")

            current_lgbm_params_from_db = {}
            if lgbm_entry and lgbm_entry.parameters:
                try: current_lgbm_params_from_db = json.loads(lgbm_entry.parameters)
                except: pass
            current_lgbm_params_from_db.pop('feature_names', None) # No es hiperparámetro

            updated_path = update_model_with_trade(prefix_lgbm, sim_trade_feats, sim_trade_target,
                                                   'lgbm', current_lgbm_params_from_db, lgbm_feats)
            if updated_path: print(f"Prueba actualización por trade OK. Modelo actualizado: {updated_path}")
            else: print("Prueba actualización por trade falló.")
        else: print("No se pudo cargar LGBM o no hay datos para prueba de actualización por trade.")
    else: print("\nSaltando prueba de Actualización por Trade.")

    print("\nPruebas de ml_engine (con Actualización por Trade) finalizadas.")
'''

with open("ml_engine.py", "w") as f:
    f.write(ML_ENGINE_TRADE_UPDATE_CONTENT)

print("ml_engine.py ha sido sobrescrito con la nueva versión que incluye update_model_with_trade y pruebas actualizadas.")

echo "Subtask de actualización de ml_engine.py (con update_model_with_trade) finalizado."
