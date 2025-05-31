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
    # ... (igual que la última versión) ...
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
    df_processed['return_1p'] = df_processed['close'].pct_change(1) * 100
    if 'return_1p' not in feature_columns: feature_columns.append('return_1p')
    if feature_lags:
        for col, lags_list in feature_lags.items():
            if col in df_processed.columns:
                for lag in lags_list:
                    lag_col_name = f'{col}_lag_{lag}'
                    df_processed[lag_col_name] = df_processed[col].shift(lag)
                    if lag_col_name not in feature_columns: feature_columns.append(lag_col_name)
    df_processed['target'] = (df_processed['future_close'] > df_processed['close']).astype(int) # Convertido a int aquí
    df_processed.dropna(subset=feature_columns + ['target', 'future_close'], inplace=True)
    if df_processed.empty or not feature_columns:
        return pd.DataFrame(), pd.Series(), df_processed, []
    X = df_processed[feature_columns]; y = df_processed['target']
    return X, y, df_processed, feature_columns

def optimize_hyperparameters(model_type, X_train, y_train, X_val, y_val, n_trials=10, study_name=None): # Reducido n_trials para velocidad
    # ... (igual que la última versión, asegurando y_train/y_val son Series) ...
    if X_train.empty or y_train.empty or X_val.empty or y_val.empty: return {}
    if isinstance(y_train, pd.DataFrame): y_train = y_train.iloc[:,0].astype(int)
    else: y_train = y_train.astype(int)
    if isinstance(y_val, pd.DataFrame): y_val = y_val.iloc[:,0].astype(int)
    else: y_val = y_val.astype(int)
    def objective(trial): # ... (resto de la función objective igual) ...
        if model_type == 'lgbm':
            lgbm_params = {'objective':'binary','metric':'binary_logloss','verbosity':-1,'boosting_type':'gbdt',
                           'n_estimators':trial.suggest_int('n_estimators',50,300,step=50),'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True),
                           'num_leaves':trial.suggest_int('num_leaves',10,80),'max_depth':trial.suggest_int('max_depth',3,10),
                           'min_child_samples':trial.suggest_int('min_child_samples',5,50)}
            model = lgb.LGBMClassifier(**lgbm_params)
            model.fit(X_train,y_train,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(5,verbose=False)])
        elif model_type == 'xgb':
            xgb_params = {'objective':'binary:logistic','eval_metric':'logloss','use_label_encoder':False,
                          'n_estimators':trial.suggest_int('n_estimators',50,300,step=50),'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True),
                          'max_depth':trial.suggest_int('max_depth',3,10),'min_child_weight':trial.suggest_int('min_child_weight',1,10)}
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=5,verbose=False)
        else: raise ValueError("model_type debe ser 'lgbm' o 'xgb'")
        preds_proba_val = model.predict_proba(X_val)[:,1]; return log_loss(y_val,preds_proba_val)
    study = optuna.create_study(direction='minimize',study_name=study_name or f"{model_type}_opt")
    try: study.optimize(objective,n_trials=n_trials,timeout=180)
    except Exception as e: print(f"Optuna ({model_type}): Excepción: {e}")
    return study.best_params if study.best_value is not None else {}

def train_lgbm_model(X, y, params=None, initial_model_path=None, X_test_override=None, y_test_override=None):
    if X.empty or y.empty: print("LGBM: No hay datos."); return None, {}, {}

    # --- INICIO DE CORRECCIÓN Y_TRAIN / Y_TEST ---
    if isinstance(y, pd.DataFrame): y = y.iloc[:,0]
    y = y.astype(int) # Asegurar que y es int antes de cualquier split
    if y_test_override is not None:
        if isinstance(y_test_override, pd.DataFrame): y_test_override = y_test_override.iloc[:,0]
        y_test_override = y_test_override.astype(int)
    # --- FIN DE CORRECCIÓN ---

    X_train, y_train, X_test, y_test = (X,y, X_test_override, y_test_override) if X_test_override is not None else                                      (X.iloc[:int(len(X)*0.8)], y.iloc[:int(len(y)*0.8)], X.iloc[int(len(X)*0.8):], y.iloc[int(len(y)*0.8):]) if len(X) >= 20 else (X,y,X,y)
    if X_train.empty or X_test.empty : print("LGBM: Train/Test sets vacíos."); return None, {}, {}

    # --- INICIO DE CORRECCIÓN Y_TRAIN / Y_TEST (después del split) ---
    if isinstance(y_train, pd.DataFrame): y_train = y_train.iloc[:,0]
    y_train = y_train.astype(int)
    if y_test is not None and not y_test.empty:
        if isinstance(y_test, pd.DataFrame): y_test = y_test.iloc[:,0]
        y_test = y_test.astype(int)
    # --- FIN DE CORRECCIÓN ---

    lgbm_base_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'n_estimators':50}
    if params: lgbm_base_params.update(params);
    else: lgbm_base_params.update({'num_leaves': 20, 'learning_rate': 0.05})
    init_model_obj = None
    if initial_model_path:
        try: init_model_obj = joblib.load(initial_model_path).booster_ if initial_model_path.endswith(".joblib") else initial_model_path
        except Exception: init_model_obj = None
    model = lgb.LGBMClassifier(**lgbm_base_params)
    callbacks_list = [lgb.early_stopping(5, verbose=False)] if not X_test.equals(X_train) else []
    eval_set_pairs = [(X_test, y_test)] if not X_test.empty and not X_test.equals(X_train) else None
    model.fit(X_train, y_train, eval_set=eval_set_pairs, eval_metric=lgbm_base_params.get('metric'), callbacks=callbacks_list, init_model=init_model_obj)

    metrics = {"train_accuracy": accuracy_score(y_train, model.predict(X_train))}
    if not X_test.empty and not X_test.equals(X_train):
        metrics.update({"test_accuracy": accuracy_score(y_test, model.predict(X_test)), "test_logloss": log_loss(y_test, model.predict_proba(X_test)[:,1])})
    return model, lgbm_base_params, metrics

def train_xgb_model(X, y, params=None, X_test_override=None, y_test_override=None):
    if X.empty or y.empty: print("XGB: No hay datos."); return None, {}, {}

    # --- INICIO DE CORRECCIÓN Y_TRAIN / Y_TEST ---
    if isinstance(y, pd.DataFrame): y = y.iloc[:,0]
    y = y.astype(int)
    if y_test_override is not None:
        if isinstance(y_test_override, pd.DataFrame): y_test_override = y_test_override.iloc[:,0]
        y_test_override = y_test_override.astype(int)
    # --- FIN DE CORRECCIÓN ---

    X_train, y_train, X_test, y_test = (X,y, X_test_override, y_test_override) if X_test_override is not None else                                      (X.iloc[:int(len(X)*0.8)], y.iloc[:int(len(y)*0.8)], X.iloc[int(len(X)*0.8):], y.iloc[int(len(y)*0.8):]) if len(X) >= 20 else (X,y,X,y)
    if X_train.empty or X_test.empty : print("XGB: Train/Test sets vacíos."); return None, {}, {}

    # --- INICIO DE CORRECCIÓN Y_TRAIN / Y_TEST (después del split) ---
    if isinstance(y_train, pd.DataFrame): y_train = y_train.iloc[:,0]
    y_train = y_train.astype(int)
    if y_test is not None and not y_test.empty:
        if isinstance(y_test, pd.DataFrame): y_test = y_test.iloc[:,0]
        y_test = y_test.astype(int)
    # --- FIN DE CORRECCIÓN ---

    xgb_base_params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 50, 'use_label_encoder': False}
    if params: xgb_base_params.update(params)
    else: xgb_base_params.update({'learning_rate': 0.1, 'max_depth': 5})
    model = xgb.XGBClassifier(**xgb_base_params)
    fit_params = {'verbose': False}
    if not X_test.empty and not X_test.equals(X_train):
        fit_params['eval_set'] = [(X_test, y_test)]; fit_params['early_stopping_rounds'] = 5
    model.fit(X_train, y_train, **fit_params)
    metrics = {"train_accuracy": accuracy_score(y_train, model.predict(X_train))}
    if not X_test.empty and not X_test.equals(X_train):
        metrics.update({"test_accuracy": accuracy_score(y_test, model.predict(X_test)), "test_logloss": log_loss(y_test, model.predict_proba(X_test)[:,1])})
    return model, xgb_base_params, metrics

def save_model(model, model_name_prefix, params_dict, metrics_dict, feature_names=None):
    # ... (igual que antes) ...
    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S");v=f"v{ts}";name_v=f"{model_name_prefix}_{v}";path_joblib=os.path.join(MODELS_DIR,f"{name_v}.joblib")
    path_txt=os.path.join(MODELS_DIR,f"{name_v}.txt") if isinstance(model,lgb.LGBMClassifier) else None
    try:
        joblib.dump(model,path_joblib)
        if path_txt:model.booster_.save_model(path_txt)
        db_s=next(db_m.get_db_session())
        try:
            p_save=params_dict.copy() if params_dict else {}; p_save['feature_names']=feature_names or []
            entry=db_m.add_model_version(db_s,model_name_prefix,v,path_joblib,json.dumps(p_save),json.dumps(metrics_dict))
            return path_joblib, entry if entry else db_m.ModelVersion(version=v)
        except Exception: return path_joblib, db_m.ModelVersion(version=v)
        finally:db_s.close()
    except Exception: return None, None

def load_model(model_name_prefix, version_str=None):
    # ... (igual que antes) ...
    db_s=next(db_m.get_db_session());entry=None;feats=None
    try:
        entry=db_s.query(db_m.ModelVersion).filter(db_m.ModelVersion.model_name==model_name_prefix,db_m.ModelVersion.version==version_str).first() if version_str else db_m.get_latest_model_version(db_s,model_name_prefix)
        if entry and entry.file_path and os.path.exists(entry.file_path):
            model=joblib.load(entry.file_path)
            if entry.parameters:
                try: feats=json.loads(entry.parameters).get('feature_names')
                except json.JSONDecodeError: pass
            return model,entry,feats
        return None,None,None
    except Exception: return None,None,None
    finally:db_s.close()

def make_prediction(model, df_features, expected_feats=None):
    # ... (igual que antes) ...
    if model is None or df_features.empty : return None, None
    try:
        if expected_feats and list(df_features.columns) != expected_feats:
            try: df_features = df_features[expected_feats]
            except KeyError: return None, None
        pred = model.predict(df_features); proba = model.predict_proba(df_features)
        return pred[0], {0: proba[0][0], 1: proba[0][1]}
    except Exception: return None, None

def make_ensemble_prediction(model_predictions_proba, weights=None):
    # ... (igual que antes) ...
    if not model_predictions_proba: return None, None
    probas_class_1 = [p.get(1,0.0) if isinstance(p,dict) else p for p in model_predictions_proba]
    n_models = len(probas_class_1)
    if weights is None or len(weights) != n_models: weights = [1/n_models]*n_models
    if not np.isclose(sum(weights),1.0): weights = [w/sum(weights) for w in weights]
    final_p1 = np.average(probas_class_1, weights=weights)
    return (1 if final_p1 >= 0.5 else 0), final_p1

def retrain_all_models(symbol, timeframe, indicators_for_model, feature_lags=None,
                       use_optuna=False, optuna_trials=10, min_data_for_optuna=100,
                       lgbm_base_params=None, xgb_base_params=None):
    # ... (igual que antes) ...
    print(f"\n--- Re-entrenamiento para {symbol}/{timeframe}, Optuna={use_optuna} ---")
    db_s = next(db_m.get_db_session()); hist_data_db = db_m.get_historical_data(db_s,symbol,timeframe,3000,False); db_s.close()
    if not hist_data_db or len(hist_data_db) < min_data_for_optuna: print(f"Datos insuficientes."); return False
    df_hist = pd.DataFrame([d.__dict__ for d in hist_data_db]); df_hist.drop(columns=['_sa_instance_state'],errors='ignore',inplace=True)
    df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
    df_i = df_hist.set_index('timestamp') if 'timestamp' in df_hist.columns else df_hist.copy()
    for col in indicators_for_model: df_i[col]=np.nan
    if 'SMA_10' in indicators_for_model: df_i['SMA_10'] = dm.calculate_sma(df_i['close'],10)
    if 'RSI_14' in indicators_for_model: df_i['RSI_14'] = dm.calculate_rsi(df_i['close'],14)
    df_feat_ready = df_i.reset_index()
    X_full, y_full, _, all_feats = prepare_features(df_feat_ready.copy(),1,indicators_for_model,feature_lags)
    if X_full.empty: print("Re-entrenamiento: No features."); return False

    final_lgbm_p = lgbm_base_params or {}; final_xgb_p = xgb_base_params or {}
    if use_optuna and len(X_full) >= min_data_for_optuna + 20:
        X_ot,X_ov,y_ot,y_ov = train_test_split(X_full,y_full,test_size=0.25,shuffle=False)
        if not X_ot.empty and not X_ov.empty:
            best_l = optimize_hyperparameters('lgbm',X_ot,y_ot,X_ov,y_ov,optuna_trials,f"LGBM_Retrain_{symbol}")
            if best_l: final_lgbm_p.update(best_l)
            best_x = optimize_hyperparameters('xgb',X_ot,y_ot,X_ov,y_ov,optuna_trials,f"XGB_Retrain_{symbol}")
            if best_x: final_xgb_p.update(best_x)

    lgbm_mod,_,l_met=train_lgbm_model(X_full,y_full,final_lgbm_p)
    if lgbm_mod:save_model(lgbm_mod,f"LGBM_{symbol}_{timeframe}_Retrained",final_lgbm_p,l_met,all_feats)
    xgb_mod,_,x_met=train_xgb_model(X_full,y_full,final_xgb_p)
    if xgb_mod:save_model(xgb_mod,f"XGB_{symbol}_{timeframe}_Retrained",final_xgb_p,x_met,all_feats)
    print(f"--- Re-entrenamiento {symbol}/{timeframe} completado ---"); return True

def update_model_with_trade(model_prefix, trade_feats_df, trade_target, model_type='lgbm', current_params=None, feat_names=None):
    # ... (igual que antes, y_upd se convertirá a int en train_lgbm_model) ...
    if model_type.lower()!='lgbm': print(f"Update por trade solo para LGBM."); return None
    if trade_feats_df.empty or not isinstance(trade_target,int): print("Update por trade: datos inválidos."); return None
    model_ld, entry_ld, feats_ld = load_model(model_prefix)
    if not model_ld or not entry_ld: print(f"No se cargó {model_prefix} para update."); return None
    active_feats = feat_names if feat_names else feats_ld
    if not active_feats: print("No feature names para update."); return None
    try: X_upd = trade_feats_df[active_feats]
    except KeyError: print(f"Features en trade_feats_df no coinciden: {active_feats}"); return None
    y_upd = pd.Series([trade_target], index=X_upd.index)
    upd_params = current_params.copy() if current_params else {}
    upd_params.update({'learning_rate':float(os.getenv("TRADE_UPDATE_LR","0.005")), 'n_estimators':int(os.getenv("TRADE_UPDATE_ESTIMATORS","1")),
                       'bagging_fraction':1.0, 'feature_fraction':1.0 })
    path_txt = entry_ld.file_path.replace(".joblib",".txt")
    if not os.path.exists(path_txt):
        try: model_ld.booster_.save_model(path_txt)
        except Exception as e: print(f"Error guardando .txt: {e}"); return None
    upd_model,_,upd_met = train_lgbm_model(X_upd,y_upd,upd_params,path_txt,X_upd,y_upd)
    if upd_model:
        s_path,_=save_model(upd_model,model_prefix,upd_params,upd_met,active_feats)
        return s_path
    return None

if __name__ == "__main__":
    # ... (El bloque if __name__ es el mismo que la versión V3 con corrección y_train) ...
    print("Ejecutando ml_engine como script principal para pruebas...")
    db_m.initialize_database(); optuna.logging.set_verbosity(optuna.logging.WARNING)
    sym=os.getenv("BOT_SYMBOL","SIM_EURUSD");tf=os.getenv("BOT_TIMEFRAME","H1")
    n_opt_trials=int(os.getenv("OPTUNA_TRIALS","5")); run_inc=os.getenv("RUN_INCREMENTAL_LGBM","true").lower()=='true'
    run_retrain=os.getenv("RUN_FULL_RETRAIN_TEST","true").lower()=='true';run_trade_upd=os.getenv("RUN_TRADE_UPDATE_TEST","true").lower()=='true'
    db_s=next(db_m.get_db_session());db_d=db_m.get_historical_data(db_s,sym,tf,700,False);db_s.close()
    df_h=None
    if db_d and len(db_d)>=300:df_h=pd.DataFrame([d.__dict__ for d in db_d]);df_h.drop(columns=['_sa_instance_state'],errors='ignore',inplace=True);df_h['timestamp']=pd.to_datetime(df_h['timestamp'])
    else:
        n_pts=700;dts=pd.to_datetime([datetime.datetime(2021,1,1)+datetime.timedelta(hours=i) for i in range(n_pts)])
        s_d={'timestamp':dts,'open':np.random.rand(n_pts)*20+1.0,'volume':np.random.randint(100,2000,n_pts)}
        s_d['high']=s_d['open']+np.random.rand(n_pts)*0.01;s_d['low']=s_d['open']-np.random.rand(n_pts)*0.01;s_d['close']=s_d['open']+(np.random.rand(n_pts)-0.5)*0.02
        s_d['low']=np.minimum.reduce([s_d['low'],s_d['open'],s_d['close']]);s_d['high']=np.maximum.reduce([s_d['high'],s_d['open'],s_d['close']])
        df_h=pd.DataFrame(s_d)
    df_i=df_h.set_index('timestamp') if 'timestamp' in df_h.columns else df_h.copy()
    conf_indic=['SMA_10','RSI_14','MACD_line','MACD_signal','BB_Mid','ADX','PLUS_DI','MINUS_DI']
    for c in conf_indic: df_i[c]=np.nan
    if 'SMA_10' in conf_indic: df_i['SMA_10']=dm.calculate_sma(df_i['close'],10)
    if 'RSI_14' in conf_indic: df_i['RSI_14']=dm.calculate_rsi(df_i['close'],14)
    if 'MACD_line' in conf_indic: m_l,m_s,_=dm.calculate_macd(df_i['close']);df_i['MACD_line']=m_l;df_i['MACD_signal']=m_s
    if 'BB_Mid' in conf_indic: bb_m,_,_=dm.calculate_bollinger_bands(df_i['close']);df_i['BB_Mid']=bb_m
    if 'ADX' in conf_indic: pdi,mdi,adx_v=dm.calculate_adx(df_i[['high','low','close']]);df_i['ADX']=adx_v;df_i['PLUS_DI']=pdi;df_i['MINUS_DI']=mdi
    hist_df_i=df_i.reset_index()
    if len(hist_df_i)<150: print("Datos insuficientes."); exit()
    X_all,y_all,_,feats_all=prepare_features(hist_df_i.copy(),1,conf_indic,None)
    if X_all.empty: print("No X,y. Saliendo."); exit()
    idx_opt=int(len(X_all)*0.6);X_opt_b=X_all.iloc[:idx_opt];y_opt_b=y_all.iloc[:idx_opt]
    X_inc_d=X_all.iloc[idx_opt:];y_inc_d=y_all.iloc[idx_opt:]
    if len(X_opt_b)<50: print(f"No datos para opt ({len(X_opt_b)})."); exit()
    X_otr,X_ov,y_otr,y_ov=train_test_split(X_opt_b,y_opt_b,test_size=0.3,shuffle=False)
    print("\n--- 2. Optimización y Entrenamiento LGBM ---")
    best_l_p=optimize_hyperparameters('lgbm',X_otr,y_otr,X_ov,y_ov,n_opt_trials,f"LGBM_Opt_{sym}")
    pfix_l=f"LGBM_{sym}_{tf}_OptunaTestV4";path_i_lgbm,opt_l_mod=(None,None) # V4
    lgbm_final_params = best_l_p.copy() if best_l_p else {}
    if best_l_p:opt_l_mod,lgbm_final_params,met_l=train_lgbm_model(X_opt_b,y_opt_b,best_l_p);
    if opt_l_mod:path_i_lgbm,_=save_model(opt_l_mod,pfix_l,lgbm_final_params,met_l,feats_all)
    if run_inc and path_i_lgbm and not X_inc_d.empty:
        print("\n--- 3. Aprendizaje Incremental LGBM ---")
        path_txt_l=path_i_lgbm.replace(".joblib",".txt")
        if os.path.exists(path_txt_l):
            p_inc_l=lgbm_final_params.copy();p_inc_l['learning_rate']=lgbm_final_params.get('learning_rate',0.05)*0.5
            X_inc_tr,y_inc_tr,X_inc_ev,y_inc_ev=train_test_split(X_inc_d,y_inc_d,test_size=0.3,shuffle=False) if len(X_inc_d)>=20 else (X_inc_d,y_inc_d,X_inc_d,y_inc_d)
            if not X_inc_tr.empty:
                inc_lgbm,p_il,m_il=train_lgbm_model(X_inc_tr,y_inc_tr,p_inc_l,path_txt_l,X_inc_ev,y_inc_ev)
                if inc_lgbm:save_model(inc_lgbm,pfix_l,p_il,m_il,feats_all);opt_l_mod=inc_lgbm; lgbm_final_params=p_il
        else: print(f"No .txt para LGBM inc: {path_txt_l}")
    print("\n--- 4. Optimización y Entrenamiento XGBoost ---")
    best_x_p=optimize_hyperparameters('xgb',X_otr,y_otr,X_ov,y_ov,n_opt_trials,f"XGB_Opt_{sym}")
    pfix_x=f"XGB_{sym}_{tf}_OptunaTestV4";opt_x_mod=None # V4
    if best_x_p:opt_x_mod,_,met_x=train_xgb_model(X_opt_b,y_opt_b,best_x_p)
    if opt_x_mod:save_model(opt_x_mod,pfix_x,best_x_p,met_x,feats_all)
    print("\n--- 5. Pruebas Ensemble ---")
    if opt_l_mod and opt_x_mod and not X_all.empty:
        sample_e=X_all.iloc[[-1]];_,p_l_d=make_prediction(opt_l_mod,sample_e,feats_all);_,p_x_d=make_prediction(opt_x_mod,sample_e,feats_all)
        if p_l_d and p_x_d:print(f"P(1) LGBM:{p_l_d[1]:.4f},P(1) XGB:{p_x_d[1]:.4f}");e_c,e_p1=make_ensemble_prediction([p_l_d[1],p_x_d[1]]);print(f"Ens Prom:C={e_c},P(1)={e_p1:.4f}")
    if run_retrain:
        print("\n--- 6. Prueba Re-entrenamiento (Optuna=True) ---")
        retrain_all_models(sym,tf,conf_indic,None,True,3,100)
        print("\n--- Prueba Re-entrenamiento (Optuna=False) ---")
        retrain_all_models(sym,tf,conf_indic,None,False)
    if run_trade_upd:
        print("\n--- 7. Prueba Update por Trade (LGBM) ---")
        if opt_l_mod and pfix_l and not X_all.empty and feats_all:
            sim_trade_feats=X_all.iloc[[-2]][feats_all] if len(X_all) >=2 else X_all.iloc[[-1]][feats_all]
            sim_trade_out=y_all.iloc[-2] if len(X_all) >=2 else y_all.iloc[-1]
            curr_p = lgbm_final_params.copy()
            curr_p.pop('feature_names',None)
            upd_path=update_model_with_trade(pfix_l,sim_trade_feats,sim_trade_out,'lgbm',curr_p,feats_all)
            if upd_path:print(f"Update por trade OK: {upd_path}")
        else: print("Saltando update por trade.")
    print("\nPruebas de ml_engine (V4 con corrección y_train DTYPE) finalizadas.")
