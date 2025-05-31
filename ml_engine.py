import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
import os
import datetime
import json

import database_manager as db_m
import data_manager as dm

MODELS_DIR = "trained_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def prepare_features(df, target_period=1, indicators_to_use=None, feature_lags=None):
    """
    Prepara características (X) y variable objetivo (y) a partir de un DataFrame.
    :param df: DataFrame con datos históricos (OHLCV y opcionalmente indicadores).
    :param target_period: Número de periodos futuros para predecir el movimiento del precio.
    :param indicators_to_use: Lista de nombres de columnas de indicadores a usar.
    :param feature_lags: Diccionario {'col_name': [lag1, lag2], ...} para crear lags.
    :return: Tuple (X, y, DataFrame original con target y features, lista de nombres de features)
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(), df, []

    df_processed = df.copy()

    df_processed['future_close'] = df_processed['close'].shift(-target_period)
    df_processed.dropna(subset=['future_close'], inplace=True)

    if df_processed.empty:
        return pd.DataFrame(), pd.Series(), df_processed, []

    df_processed['target'] = (df_processed['future_close'] > df_processed['close']).astype(int)

    feature_columns = []
    if indicators_to_use:
        for indicator in indicators_to_use:
            if indicator in df_processed.columns:
                feature_columns.append(indicator)
            else:
                print(f"Advertencia: Indicador '{indicator}' no encontrado en el DataFrame.")

    df_processed['return_1p'] = df_processed['close'].pct_change(1) * 100 # En porcentaje
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

    df_processed.dropna(subset=feature_columns + ['target'], inplace=True)

    if df_processed.empty or not feature_columns:
        print("No hay suficientes datos o características después del preprocesamiento.")
        return pd.DataFrame(), pd.Series(), df_processed, []

    X = df_processed[feature_columns]
    y = df_processed['target']

    print(f"Características seleccionadas: {feature_columns}")
    print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")

    return X, y, df_processed, feature_columns


def train_lgbm_model(X, y, params=None, initial_model_path=None, X_test_override=None, y_test_override=None):
    """
    Entrena un modelo LightGBM, opcionalmente de forma incremental.
    :param X: DataFrame de características para entrenamiento.
    :param y: Serie de la variable objetivo para entrenamiento.
    :param params: Diccionario de parámetros para LightGBM.
    :param initial_model_path: Path al modelo LightGBM (.txt o .joblib) para continuar el entrenamiento.
    :param X_test_override, y_test_override: Conjuntos de prueba explícitos. Si son None, se dividen de X, y.
    :return: Modelo LightGBM entrenado, parámetros usados, y métricas de evaluación.
    """
    if X.empty or y.empty:
        print("No hay datos para entrenar el modelo.")
        return None, {}, {}

    # Determinar conjunto de entrenamiento y prueba
    if X_test_override is not None and y_test_override is not None:
        X_train, y_train = X, y
        X_test, y_test = X_test_override, y_test_override
        print("Usando conjuntos de entrenamiento/prueba explícitos.")
    else:
        # División secuencial para series temporales
        split_index = int(len(X) * 0.8)
        if split_index < 1 or split_index >= len(X) -1 : # No hay suficientes datos para split
             print("No hay suficientes datos para dividir en entrenamiento y prueba. Usando todo para entrenar y evaluar.")
             X_train, X_test = X, X
             y_train, y_test = y, y
        else:
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        print("Dividiendo datos de entrada en entrenamiento (80%) y prueba (20%).")


    if X_train.empty or X_test.empty: # Podría pasar si el split_index es 0 o len(X)
        print("Conjuntos de entrenamiento o prueba vacíos después de la división.")
        return None, {}, {}

    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

    lgbm_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9,
        'verbose': -1, 'n_estimators': 100 # Añadido n_estimators
    }
    if params: # Sobrescribir defaults con params provistos
        lgbm_params.update(params)

    init_model_obj = None
    if initial_model_path:
        try:
            # LightGBM puede continuar desde un archivo .txt (modelo de texto) o un booster serializado.
            # Si es un LGBMClassifier de scikit-learn guardado con joblib:
            if initial_model_path.endswith(".joblib"):
                loaded_model_wrapper = joblib.load(initial_model_path)
                init_model_obj = loaded_model_wrapper.booster_ # Obtener el booster subyacente
                print(f"Modelo inicial (booster) cargado desde joblib: {initial_model_path}")
            else: # Asumir que es un archivo de texto de LightGBM o un booster
                init_model_obj = initial_model_path # Pasar el path directamente a lgb.train o lgb.LGBMClassifier
                print(f"Modelo inicial (path o booster) para LightGBM: {initial_model_path}")
        except Exception as e:
            print(f"Error al cargar el modelo inicial desde {initial_model_path}: {e}. Entrenando desde cero.")
            init_model_obj = None

    # Crear el clasificador
    # Si init_model_obj es un path a un modelo de texto, LGBMClassifier lo maneja.
    # Si es un objeto Booster, también.
    model = lgb.LGBMClassifier(**lgbm_params)

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric=lgbm_params.get('metric', 'binary_logloss'), # Usar la métrica definida en params
              callbacks=[lgb.early_stopping(10, verbose=False)],
              init_model=init_model_obj if init_model_obj else None) # Pasar el modelo inicial

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1] # Probabilidad de la clase positiva

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
        "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
        "test_logloss": log_loss(y_test, y_proba_test)
    }
    print(f"Métricas del modelo: {metrics}")

    return model, lgbm_params, metrics


def save_model(model, model_name_prefix, params_dict, metrics_dict, feature_names=None):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_str = f"v{timestamp_str}"
    model_name_with_version = f"{model_name_prefix}_{version_str}"

    # Guardar el modelo scikit-learn wrapper (LGBMClassifier)
    file_path_joblib = os.path.join(MODELS_DIR, f"{model_name_with_version}.joblib")

    # Guardar también el modelo de texto de LightGBM (opcional, pero útil para init_model)
    file_path_txt = os.path.join(MODELS_DIR, f"{model_name_with_version}.txt")

    try:
        joblib.dump(model, file_path_joblib)
        print(f"Modelo (wrapper) guardado en: {file_path_joblib}")

        model.booster_.save_model(file_path_txt) # Guardar el booster nativo
        print(f"Modelo (booster nativo) guardado en: {file_path_txt}")

        db_session_gen = db_m.get_db_session()
        session = next(db_session_gen)
        try:
            # Guardar también los nombres de las features usados para entrenar este modelo
            # Esto es crucial para asegurar consistencia al predecir.
            full_params_to_save = params_dict.copy() if params_dict else {}
            if feature_names:
                full_params_to_save['feature_names'] = feature_names

            model_entry = db_m.add_model_version(
                session, model_name=model_name_prefix, version=version_str,
                file_path=file_path_joblib, # Guardar path al .joblib como principal
                parameters=json.dumps(full_params_to_save) if full_params_to_save else None,
                performance_metrics=json.dumps(metrics_dict) if metrics_dict else None
            )
            return file_path_joblib, model_entry
        except Exception as e_db:
            print(f"Error al registrar modelo en BD: {e_db}")
            return file_path_joblib, None
        finally:
            session.close()
    except Exception as e_save:
        print(f"Error al guardar modelo: {e_save}")
        return None, None

def load_model(model_name_prefix, version_str=None):
    model_entry = None
    db_session_gen = db_m.get_db_session()
    session = next(db_session_gen)
    try:
        if version_str:
             model_entry = session.query(db_m.ModelVersion)                .filter(db_m.ModelVersion.model_name == model_name_prefix, db_m.ModelVersion.version == version_str)                .first()
        else:
            model_entry = db_m.get_latest_model_version(session, model_name_prefix)

        if model_entry and model_entry.file_path and os.path.exists(model_entry.file_path):
            model = joblib.load(model_entry.file_path)
            print(f"Modelo cargado desde: {model_entry.file_path}")

            # Cargar feature_names desde los parámetros guardados
            feature_names = None
            if model_entry.parameters:
                try:
                    params_json = json.loads(model_entry.parameters)
                    feature_names = params_json.get('feature_names')
                except json.JSONDecodeError:
                    print("Advertencia: No se pudieron parsear los parámetros del modelo desde la BD.")

            return model, model_entry, feature_names
        elif model_entry: # Entrada de BD existe pero archivo no
             print(f"Error: Archivo de modelo no encontrado en {model_entry.file_path} (referenciado en BD).")
             return None, model_entry, None
        else:
            print(f"No se encontró versión del modelo '{model_name_prefix}' (versión: {version_str if version_str else 'latest'}) en la BD.")
            return None, None, None
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return None, None, None
    finally:
        session.close()

def make_prediction(model, current_features_df, expected_feature_names=None):
    if model is None: return None, None
    if current_features_df.empty: return None, None

    try:
        # Asegurar que las columnas están en el orden esperado si se proporcionó expected_feature_names
        if expected_feature_names:
            if list(current_features_df.columns) != expected_feature_names:
                print("Advertencia: El orden/nombre de las features de entrada no coincide con el esperado. Reordenando.")
                try:
                    current_features_df = current_features_df[expected_feature_names]
                except KeyError as e:
                    print(f"Error fatal: Faltan features esperadas en la entrada: {e}")
                    return None, None

        prediction = model.predict(current_features_df)
        proba = model.predict_proba(current_features_df)
        return prediction[0], proba[0]
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return None, None

# --- Ejemplo de Uso / Pruebas ---
if __name__ == "__main__":
    print("Ejecutando ml_engine como script principal para pruebas...")
    db_m.initialize_database()

    print("\n--- Preparando Datos Simulados para el Modelo ---")
    # Generar más datos para permitir splits y aprendizaje incremental
    num_points_total = 500
    sim_dates = pd.to_datetime([datetime.datetime(2023, 1, 1) + datetime.timedelta(hours=i) for i in range(num_points_total)])
    sim_data = {
        'timestamp': sim_dates, 'open': np.random.rand(num_points_total) * 10 + 100,
        'volume': np.random.randint(100, 1000, num_points_total)
    }
    sim_data['high'] = sim_data['open'] + np.random.rand(num_points_total) * 2
    sim_data['low'] = sim_data['open'] - np.random.rand(num_points_total) * 2
    sim_data['close'] = sim_data['open'] + (np.random.rand(num_points_total) - 0.5) * 3
    historical_df = pd.DataFrame(sim_data)
    historical_df.set_index('timestamp', inplace=True)

    # Calcular indicadores
    historical_df['SMA_10'] = dm.calculate_sma(historical_df['close'], 10)
    historical_df['RSI_14'] = dm.calculate_rsi(historical_df['close'], 14)
    macd_line, macd_signal, _ = dm.calculate_macd(historical_df['close'])
    historical_df['MACD_line'] = macd_line
    historical_df['MACD_signal'] = macd_signal
    historical_df.dropna(inplace=True)

    if historical_df.empty:
        print("DataFrame vacío después de calcular indicadores. No se puede continuar.")
        exit()

    # Dividir datos para simular entrenamiento inicial y luego datos nuevos
    num_initial_train = int(len(historical_df) * 0.6)
    df_initial_train = historical_df.iloc[:num_initial_train]
    df_incremental_new_data = historical_df.iloc[num_initial_train:]

    print(f"Datos iniciales para entrenamiento: {len(df_initial_train)} filas")
    print(f"Nuevos datos para aprendizaje incremental: {len(df_incremental_new_data)} filas")

    # --- Entrenamiento Inicial ---
    print("\n--- 1. Entrenamiento Inicial del Modelo ---")
    # Usar .reset_index() si prepare_features no espera un DatetimeIndex
    X_initial, y_initial, _, feat_names_initial = prepare_features(
        df_initial_train.reset_index(), target_period=1,
        indicators_to_use=['SMA_10', 'RSI_14', 'MACD_line', 'MACD_signal']
    )

    model_symbol = "SIM_EURUSD"
    model_tf = "H1"
    model_prefix = f"LGBM_{model_symbol}_{model_tf}_Incremental"

    if not X_initial.empty:
        initial_model, params_initial, metrics_initial = train_lgbm_model(X_initial, y_initial)
        if initial_model:
            print(f"Modelo inicial entrenado. Test Accuracy: {metrics_initial.get('test_accuracy', 'N/A')}")
            saved_path_initial, entry_initial = save_model(initial_model, model_prefix, params_initial, metrics_initial, feature_names=feat_names_initial)
            if saved_path_initial:
                print(f"Modelo inicial guardado en: {saved_path_initial}")

                # --- Aprendizaje Incremental ---
                print("\n--- 2. Aprendizaje Incremental con Nuevos Datos ---")
                # Preparar características para los nuevos datos
                X_new, y_new, _, feat_names_new = prepare_features(
                    df_incremental_new_data.reset_index(), target_period=1,
                    indicators_to_use=['SMA_10', 'RSI_14', 'MACD_line', 'MACD_signal']
                )
                # Asegurarse de que las features son las mismas
                if feat_names_initial != feat_names_new:
                    print("¡Advertencia! Los nombres de las características difieren entre el conjunto inicial y el nuevo.")

                if not X_new.empty:
                    # Cargar el último modelo (que debería ser el inicial que acabamos de guardar)
                    # Opcionalmente, podemos pasar `saved_path_initial` directamente si es el archivo .txt
                    # Para LGBMClassifier.fit(init_model=...), necesitamos el objeto clasificador o el path al modelo de texto.
                    # El `save_model` ahora guarda un .txt también.

                    # Construir el path al archivo .txt del modelo inicial
                    initial_model_txt_path = saved_path_initial.replace(".joblib", ".txt") if saved_path_initial else None

                    if initial_model_txt_path and os.path.exists(initial_model_txt_path):
                        # Para el aprendizaje incremental, es común usar un learning_rate más pequeño
                        params_incremental = params_initial.copy()
                        params_incremental['learning_rate'] = params_incremental.get('learning_rate', 0.05) * 0.5

                        # Dividir X_new, y_new para tener un conjunto de evaluación para el entrenamiento incremental
                        split_idx_new = int(len(X_new) * 0.7) # Usar 70% de los nuevos datos para entrenar, 30% para evaluar
                        X_new_train, X_new_eval = X_new.iloc[:split_idx_new], X_new.iloc[split_idx_new:]
                        y_new_train, y_new_eval = y_new.iloc[:split_idx_new], y_new.iloc[split_idx_new:]

                        if not X_new_train.empty and not X_new_eval.empty:
                            incremental_model, params_incr, metrics_incr = train_lgbm_model(
                                X_new_train, y_new_train,
                                params=params_incremental,
                                initial_model_path=initial_model_txt_path, # Pasar el path al modelo de texto
                                X_test_override=X_new_eval, # Evaluar sobre una porción de los nuevos datos
                                y_test_override=y_new_eval
                            )
                            if incremental_model:
                                print(f"Modelo incremental entrenado. Test Accuracy: {metrics_incr.get('test_accuracy', 'N/A')}")
                                # Guardar el modelo incremental (tendrá nuevo timestamp/versión)
                                save_model(incremental_model, model_prefix, params_incr, metrics_incr, feature_names=feat_names_new)
                            else:
                                print("Fallo al entrenar modelo incremental.")
                        else:
                             print("No hay suficientes datos nuevos para dividir y hacer entrenamiento incremental.")
                    else:
                        print(f"No se encontró el archivo de modelo de texto: {initial_model_txt_path}. Saltando aprendizaje incremental.")
                else:
                    print("No hay nuevas características/target para aprendizaje incremental.")
            else:
                print("Fallo al guardar el modelo inicial.")
        else:
            print("Fallo al entrenar modelo inicial.")
    else:
        print("No se pudieron generar X, y iniciales.")

    print("\nPruebas de ml_engine (con aprendizaje incremental) finalizadas.")
