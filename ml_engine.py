import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split # Aunque no se usa directamente para split final
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
import os
import datetime
import json

# Importar desde nuestros módulos
import database_manager as db_m
import data_manager as dm

MODELS_DIR = "trained_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def prepare_features(df, target_period=1, indicators_to_use=None, feature_lags=None):
    """
    Prepara características (X) y variable objetivo (y) a partir de un DataFrame.
    :param df: DataFrame con datos históricos (OHLCV y opcionalmente indicadores calculados).
               Debe tener una columna 'timestamp' si no está como índice.
    :param target_period: Número de periodos futuros para predecir el movimiento del precio.
    :param indicators_to_use: Lista de nombres de columnas de indicadores a usar.
    :param feature_lags: Diccionario {'col_name': [lag1, lag2], ...} para crear lags.
    :return: Tuple (X, y, DataFrame original con target y features, lista de nombres de features)
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(), df, []

    df_processed = df.copy()
    # Asegurar que el df tiene un índice de tiempo o una columna 'timestamp' para shifts
    if not isinstance(df_processed.index, pd.DatetimeIndex) and 'timestamp' in df_processed.columns:
        df_processed.set_index('timestamp', inplace=True)
    elif not isinstance(df_processed.index, pd.DatetimeIndex):
        raise ValueError("DataFrame debe tener un índice DatetimeIndex o una columna 'timestamp'.")

    df_processed.sort_index(inplace=True) # Asegurar orden cronológico

    df_processed['future_close'] = df_processed['close'].shift(-target_period)
    # No eliminar NaNs de future_close aquí, sino después de crear todas las features.
    # Esto es para que los lags no se calculen sobre datos ya reducidos.

    feature_columns = []
    if indicators_to_use:
        for indicator in indicators_to_use:
            if indicator in df_processed.columns:
                feature_columns.append(indicator)
            else:
                print(f"Advertencia: Indicador '{indicator}' no encontrado en el DataFrame.")

    df_processed['return_1p'] = df_processed['close'].pct_change(1) * 100
    if 'return_1p' not in feature_columns: feature_columns.append('return_1p')

    if feature_lags:
        for col, lags in feature_lags.items():
            if col in df_processed.columns:
                for lag in lags:
                    lag_col_name = f'{col}_lag_{lag}'
                    df_processed[lag_col_name] = df_processed[col].shift(lag)
                    if lag_col_name not in feature_columns: feature_columns.append(lag_col_name)
            else:
                print(f"Advertencia: Columna '{col}' para lags no encontrada.")

    # Ahora eliminar todas las filas que tengan algún NaN en target o features
    # Esto incluye NaNs de future_close, y NaNs generados por pct_change o lags.
    df_processed.dropna(subset=feature_columns + ['target', 'future_close'], inplace=True)


    if df_processed.empty or not feature_columns:
        print("No hay suficientes datos o características después del preprocesamiento y dropna.")
        return pd.DataFrame(), pd.Series(), df_processed, []

    X = df_processed[feature_columns]
    y = df_processed['target']

    # print(f"Características seleccionadas: {feature_columns}")
    # print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")

    return X, y, df_processed, feature_columns


def train_lgbm_model(X, y, params=None, initial_model_path=None, X_test_override=None, y_test_override=None):
    if X.empty or y.empty:
        print("No hay datos para entrenar el modelo.")
        return None, {}, {}

    if X_test_override is not None and y_test_override is not None:
        X_train, y_train = X, y
        X_test, y_test = X_test_override, y_test_override
    else:
        # División secuencial para series temporales
        # Asegurarse de que hay suficientes datos para un split significativo
        if len(X) < 50 : # Umbral mínimo arbitrario para un split
            print(f"Muy pocos datos ({len(X)}) para un split train/test significativo. Usando todo para entrenar y evaluar.")
            X_train, X_test = X, X
            y_train, y_test = y, y
        else:
            split_index = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if X_train.empty or (X_test_override is None and X_test.empty): # X_test puede estar vacío si X_test_override se usa pero es None
        print("Conjuntos de entrenamiento o prueba vacíos después de la división.")
        return None, {}, {}

    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}, Prueba: {X_test.shape if X_test is not None else 'N/A'}")

    lgbm_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9,
        'verbose': -1, 'n_estimators': 100
    }
    if params: lgbm_params.update(params)

    init_model_obj = None
    if initial_model_path:
        try:
            if initial_model_path.endswith(".joblib"):
                loaded_model_wrapper = joblib.load(initial_model_path)
                init_model_obj = loaded_model_wrapper.booster_
                print(f"Modelo inicial (booster) cargado desde joblib: {initial_model_path}")
            else:
                init_model_obj = initial_model_path
                print(f"Modelo inicial (path o booster) para LightGBM: {initial_model_path}")
        except Exception as e:
            print(f"Error al cargar el modelo inicial desde {initial_model_path}: {e}. Entrenando desde cero.")
            init_model_obj = None

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)] if X_test is not None and not X_test.empty else None, # Solo pasar eval_set si es válido
              eval_metric=lgbm_params.get('metric', 'binary_logloss'),
              callbacks=[lgb.early_stopping(10, verbose=False)] if X_test is not None and not X_test.empty else None,
              init_model=init_model_obj if init_model_obj else None)

    y_pred_train = model.predict(X_train)
    metrics = {"train_accuracy": accuracy_score(y_train, y_pred_train)}
    if X_test is not None and not X_test.empty:
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        metrics.update({
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
            "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
            "test_logloss": log_loss(y_test, y_proba_test)
        })
    print(f"Métricas del modelo: {metrics}")
    return model, lgbm_params, metrics

def save_model(model, model_name_prefix, params_dict, metrics_dict, feature_names=None):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_str = f"v{timestamp_str}"
    model_name_with_version = f"{model_name_prefix}_{version_str}"
    file_path_joblib = os.path.join(MODELS_DIR, f"{model_name_with_version}.joblib")
    file_path_txt = os.path.join(MODELS_DIR, f"{model_name_with_version}.txt")
    try:
        joblib.dump(model, file_path_joblib)
        model.booster_.save_model(file_path_txt)
        print(f"Modelo guardado en .joblib y .txt con versión {version_str}")
        db_session_gen = db_m.get_db_session()
        session = next(db_session_gen)
        try:
            full_params_to_save = params_dict.copy() if params_dict else {}
            if feature_names: full_params_to_save['feature_names'] = feature_names
            model_entry = db_m.add_model_version(
                session, model_name=model_name_prefix, version=version_str,
                file_path=file_path_joblib,
                parameters=json.dumps(full_params_to_save) if full_params_to_save else None,
                performance_metrics=json.dumps(metrics_dict) if metrics_dict else None
            )
            return file_path_joblib, model_entry
        except Exception as e_db: print(f"Error al registrar modelo en BD: {e_db}"); return file_path_joblib, None
        finally: session.close()
    except Exception as e_save: print(f"Error al guardar modelo: {e_save}"); return None, None

def load_model(model_name_prefix, version_str=None):
    model_entry = None; feature_names = None
    db_session_gen = db_m.get_db_session()
    session = next(db_session_gen)
    try:
        if version_str:
             model_entry = session.query(db_m.ModelVersion)                .filter(db_m.ModelVersion.model_name == model_name_prefix, db_m.ModelVersion.version == version_str).first()
        else: model_entry = db_m.get_latest_model_version(session, model_name_prefix)
        if model_entry and model_entry.file_path and os.path.exists(model_entry.file_path):
            model = joblib.load(model_entry.file_path)
            print(f"Modelo cargado desde: {model_entry.file_path}")
            if model_entry.parameters:
                try: feature_names = json.loads(model_entry.parameters).get('feature_names')
                except json.JSONDecodeError: print("Advertencia: No se pudieron parsear los parámetros del modelo.")
            return model, model_entry, feature_names
        elif model_entry: print(f"Error: Archivo de modelo no encontrado en {model_entry.file_path}"); return None, model_entry, None
        else: print(f"No se encontró modelo '{model_name_prefix}' (versión: {version_str if version_str else 'latest'}) en BD."); return None, None, None
    except Exception as e: print(f"Error al cargar modelo: {e}"); return None, None, None
    finally: session.close()

def make_prediction(model, current_features_df, expected_feature_names=None):
    if model is None or current_features_df.empty : return None, None
    try:
        if expected_feature_names:
            if list(current_features_df.columns) != expected_feature_names:
                print("Advertencia: Reordenando features de entrada para coincidir con el entrenamiento.")
                try: current_features_df = current_features_df[expected_feature_names]
                except KeyError as e: print(f"Error fatal: Faltan features esperadas: {e}"); return None, None
        prediction = model.predict(current_features_df)
        proba = model.predict_proba(current_features_df)
        return prediction[0], proba[0]
    except Exception as e: print(f"Error durante la predicción: {e}"); return None, None

if __name__ == "__main__":
    print("Ejecutando ml_engine como script principal para pruebas...")
    # if os.path.exists(db_m.DATABASE_NAME):
    #     print(f"Eliminando BD existente: {db_m.DATABASE_NAME} para prueba limpia de ml_engine.")
    #     os.remove(db_m.DATABASE_NAME)
    db_m.initialize_database()

    # --- Obtener o generar datos ---
    print("\n--- 1. Preparando Datos para el Modelo ---")
    test_symbol = os.getenv("BOT_SYMBOL", "SIM_EURUSD")
    test_timeframe = os.getenv("BOT_TIMEFRAME", "H1")

    # Intentar cargar desde BD (poblada por data_manager.py)
    db_session_gen = db_m.get_db_session()
    session = next(db_session_gen)
    try:
        # Pedir suficientes datos para indicadores y splits (ej. 300-500)
        db_data = db_m.get_historical_data(session, symbol=test_symbol, timeframe=test_timeframe, limit=500, order_desc=False) # Ascendente
    finally:
        session.close()

    historical_df = None
    if db_data and len(db_data) >= 150: # Umbral para considerar datos de BD suficientes
        print(f"Usando {len(db_data)} registros de la BD para {test_symbol}/{test_timeframe}")
        historical_df = pd.DataFrame([d.__dict__ for d in db_data])
        if '_sa_instance_state' in historical_df.columns:
            historical_df.drop(columns=['_sa_instance_state'], inplace=True)
        historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
        # No establecer índice aquí, prepare_features lo manejará
    else:
        print(f"Datos de BD insuficientes ({len(db_data) if db_data else 0} registros) o no encontrados. Generando datos simulados...")
        num_points_total = 400 # AUMENTADO para tener suficientes datos para features y splits
        sim_dates = pd.to_datetime([datetime.datetime(2022, 1, 1) + datetime.timedelta(hours=i) for i in range(num_points_total)])
        sim_data_dict = {
            'timestamp': sim_dates, 'open': np.random.rand(num_points_total) * 20 + 1.0,
            'volume': np.random.randint(100, 2000, num_points_total)
        }
        sim_data_dict['high'] = sim_data_dict['open'] + np.random.rand(num_points_total) * 0.01
        sim_data_dict['low'] = sim_data_dict['open'] - np.random.rand(num_points_total) * 0.01
        sim_data_dict['close'] = sim_data_dict['open'] + (np.random.rand(num_points_total) - 0.5) * 0.02
        sim_data_dict['low'] = np.minimum(sim_data_dict['low'], sim_data_dict['open']); sim_data_dict['low'] = np.minimum(sim_data_dict['low'], sim_data_dict['close'])
        sim_data_dict['high'] = np.maximum(sim_data_dict['high'], sim_data_dict['open']); sim_data_dict['high'] = np.maximum(sim_data_dict['high'], sim_data_dict['close'])
        historical_df = pd.DataFrame(sim_data_dict)
        # Opcional: guardar estos datos simulados en la BD para la próxima ejecución
        # dm.store_historical_data(historical_df.copy(), test_symbol, test_timeframe) # Pasar copia


    # Calcular indicadores
    # Asegurar que el df tenga índice de tiempo para los cálculos de indicadores que lo requieran
    df_for_indicators = historical_df.copy()
    if not isinstance(df_for_indicators.index, pd.DatetimeIndex) and 'timestamp' in df_for_indicators.columns:
        df_for_indicators.set_index('timestamp', inplace=True)
    elif not isinstance(df_for_indicators.index, pd.DatetimeIndex): # Si no hay columna timestamp tampoco
         raise ValueError("DataFrame para indicadores debe tener un índice DatetimeIndex o una columna 'timestamp'.")

    df_for_indicators['SMA_10'] = dm.calculate_sma(df_for_indicators['close'], 10)
    df_for_indicators['RSI_14'] = dm.calculate_rsi(df_for_indicators['close'], 14)
    macd_line, macd_signal, _ = dm.calculate_macd(df_for_indicators['close'])
    df_for_indicators['MACD_line'] = macd_line; df_for_indicators['MACD_signal'] = macd_signal
    bb_mid, _, _ = dm.calculate_bollinger_bands(df_for_indicators['close'])
    df_for_indicators['BB_Mid'] = bb_mid
    _, _, adx = dm.calculate_adx(df_for_indicators[['high','low','close']]) # ADX usa un DF
    df_for_indicators['ADX'] = adx

    # Los indicadores pueden haber reducido la longitud del DF si generaron NaNs al principio
    # y no fueron manejados internamente por cada función de indicador para devolver NaNs.
    # El df que se pasa a prepare_features es df_for_indicators.reset_index() si el índice era timestamp.
    # O simplemente df_for_indicators si el índice no era timestamp (y 'timestamp' es una columna).

    # Resetear índice si 'timestamp' era el índice, para pasarlo como columna a prepare_features
    if isinstance(df_for_indicators.index, pd.DatetimeIndex):
        df_to_prepare = df_for_indicators.reset_index()
    else: # Si 'timestamp' ya es una columna
        df_to_prepare = df_for_indicators.copy()


    print(f"Total de barras después de calcular indicadores: {len(df_to_prepare)}")
    if df_to_prepare.empty: print("DataFrame vacío después de calcular indicadores. No se puede continuar."); exit()

    # --- Definir Features ---
    indicators_for_model = ['SMA_10', 'RSI_14', 'MACD_line', 'MACD_signal', 'BB_Mid', 'ADX']
    # lags_for_model = {'close': [1, 2], 'RSI_14': [1]} # Ejemplo de lags
    lags_for_model = None


    # --- Dividir datos para simular entrenamiento inicial y luego datos nuevos ---
    # Asegurarse de que df_initial_train_raw y df_incremental_new_data_raw no estén vacíos
    if len(df_to_prepare) < 100 : # Necesita suficientes datos para dos fases de entrenamiento + preparación de features
        print(f"No hay suficientes datos ({len(df_to_prepare)}) para simular entrenamiento inicial e incremental de forma robusta. Se requiere > 100.")
        exit()

    num_initial_train_raw = int(len(df_to_prepare) * 0.7) # 70% para el primer entrenamiento
    df_initial_train_raw = df_to_prepare.iloc[:num_initial_train_raw]
    df_incremental_new_data_raw = df_to_prepare.iloc[num_initial_train_raw:]

    print(f"Datos crudos para entrenamiento inicial: {len(df_initial_train_raw)} filas")
    print(f"Datos crudos para aprendizaje incremental: {len(df_incremental_new_data_raw)} filas")

    # --- Entrenamiento Inicial ---
    print("\n--- 2. Entrenamiento Inicial del Modelo ---")
    X_initial, y_initial, _, feat_names_initial = prepare_features(
        df_initial_train_raw, target_period=1, # df_initial_train_raw ya tiene 'timestamp' como columna
        indicators_to_use=indicators_for_model, feature_lags=lags_for_model
    )

    model_prefix = f"LGBM_{test_symbol}_{test_timeframe}_Incremental"

    saved_path_initial_joblib = None
    if not X_initial.empty:
        initial_model, params_initial, metrics_initial = train_lgbm_model(X_initial, y_initial)
        if initial_model:
            print(f"Modelo inicial entrenado. Test Accuracy: {metrics_initial.get('test_accuracy', 'N/A')}")
            saved_path_initial_joblib, _ = save_model(initial_model, model_prefix, params_initial, metrics_initial, feature_names=feat_names_initial)
            if saved_path_initial_joblib: print(f"Modelo inicial guardado (joblib): {saved_path_initial_joblib}")
        else: print("Fallo al entrenar modelo inicial.")
    else: print("No se pudieron generar X, y iniciales para el entrenamiento.")

    # --- Aprendizaje Incremental ---
    if saved_path_initial_joblib: # Solo si el modelo inicial se guardó
        print("\n--- 3. Aprendizaje Incremental con Nuevos Datos ---")
        X_new, y_new, _, feat_names_new = prepare_features(
            df_incremental_new_data_raw, target_period=1, # df_incremental_new_data_raw ya tiene 'timestamp' como columna
            indicators_to_use=indicators_for_model, feature_lags=lags_for_model
        )
        if feat_names_initial != feat_names_new: print("¡Advertencia! Nombres de features difieren entre conjuntos.")

        if not X_new.empty:
            initial_model_txt_path = saved_path_initial_joblib.replace(".joblib", ".txt")
            if os.path.exists(initial_model_txt_path):
                params_incremental = params_initial.copy(); params_incremental['learning_rate'] *= 0.5

                # Dividir los nuevos datos para entrenar y evaluar el paso incremental
                if len(X_new) < 50: # Si hay muy pocos datos nuevos
                    X_new_train, y_new_train = X_new, y_new
                    X_new_eval, y_new_eval = None, None # No evaluar en este caso
                    print("Pocos datos nuevos, usando todo para entrenar incrementalmente sin set de evaluación interno.")
                else:
                    split_idx_new = int(len(X_new) * 0.7)
                    X_new_train, X_new_eval = X_new.iloc[:split_idx_new], X_new.iloc[split_idx_new:]
                    y_new_train, y_new_eval = y_new.iloc[:split_idx_new], y_new.iloc[split_idx_new:]

                if not X_new_train.empty:
                    incremental_model, params_incr, metrics_incr = train_lgbm_model(
                        X_new_train, y_new_train, params=params_incremental,
                        initial_model_path=initial_model_txt_path,
                        X_test_override=X_new_eval, y_test_override=y_new_eval
                    )
                    if incremental_model:
                        print(f"Modelo incremental entrenado. Test Accuracy: {metrics_incr.get('test_accuracy', 'N/A')}")
                        save_model(incremental_model, model_prefix, params_incr, metrics_incr, feature_names=feat_names_new)
                    else: print("Fallo al entrenar modelo incremental.")
                else: print("No hay datos de entrenamiento para el paso incremental.")
            else: print(f"No se encontró archivo .txt: {initial_model_txt_path}. Saltando aprendizaje incremental.")
        else: print("No hay nuevas características/target para aprendizaje incremental.")
    else: print("Modelo inicial no guardado, saltando aprendizaje incremental.")

    print("\nPruebas de ml_engine (con aprendizaje incremental y datos simulados aumentados) finalizadas.")
