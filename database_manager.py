import os
import datetime
import json # Aunque no se usa directamente aquí, es relevante para ModelVersion
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index, Enum as SQLAlchemyEnum
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Importar enum de Python si se usa para definir valores
import enum

# Nombre de la base de datos (eventualmente se leerá de .env)
# Para mantener la consistencia con data_manager, asumimos que .env ya está cargado
# o usamos un default.
DATABASE_NAME = os.getenv("DB_NAME", "trading_bot.db")
DATABASE_URL = f"sqlite:///{DATABASE_NAME}"

engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Definiciones de Enum (para mejorar integridad de datos) ---
class TradeStatusEnum(enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    SIMULATED_OPEN = "SIMULATED_OPEN"
    SIMULATED_CLOSED = "SIMULATED_CLOSED"

class OrderTypeEnum(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    # Podrían añadirse otros como BUY_LIMIT, SELL_LIMIT, etc.

# --- Definición de Modelos (Esquemas de Tablas) ---
class HistoricalData(Base):
    __tablename__ = "historical_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True) # Ya indexado
    symbol = Column(String, nullable=False, index=True)    # Ya indexado
    timeframe = Column(String, nullable=False, index=True) # Ya indexado
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<HistoricalData(id={self.id}, symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}')>"

# Índice compuesto para HistoricalData (mejora consultas comunes)
Index('ix_histdata_symbol_timeframe_timestamp', HistoricalData.symbol, HistoricalData.timeframe, HistoricalData.timestamp.desc())


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True) # Ya indexado
    symbol = Column(String, nullable=False, index=True) # Añadir índice
    # Usar SQLAlchemyEnum para order_type y status
    order_type = Column(SQLAlchemyEnum(OrderTypeEnum), nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    status = Column(SQLAlchemyEnum(TradeStatusEnum), default=TradeStatusEnum.OPEN, index=True)
    mt5_ticket_id = Column(Integer, nullable=True, unique=True, index=True) # Ya unique e indexado

    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', order_type='{self.order_type.value if self.order_type else None}', status='{self.status.value if self.status else None}')>"

# Índice compuesto para Trades (ej: para buscar trades abiertos de un símbolo)
Index('ix_trades_symbol_status', Trade.symbol, Trade.status)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String, nullable=False, index=True) # Ya indexado
    version = Column(String, nullable=False, unique=True) # Ya unique, por ende indexado
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True) # Añadir índice
    file_path = Column(String, nullable=False)
    parameters = Column(String, nullable=True) # JSON string
    performance_metrics = Column(String, nullable=True) # JSON string

    def __repr__(self):
        return f"<ModelVersion(model_name='{self.model_name}', version='{self.version}')>"

# Índice compuesto para ModelVersion (para buscar la última versión de un modelo)
Index('ix_modelversions_name_timestamp', ModelVersion.model_name, ModelVersion.timestamp.desc())


# --- Funciones de Gestión de Base de Datos ---
def initialize_database():
    """Crea todas las tablas en la base de datos si no existen."""
    try:
        Base.metadata.create_all(bind=engine)
        print("Base de datos inicializada y tablas (con índices actualizados) creadas.")
    except SQLAlchemyError as e:
        print(f"Error al inicializar la base de datos: {e}")

def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Funciones CRUD (adaptadas para Enums donde sea necesario) ---

def add_historical_data(session, data_list):
    try:
        objects = [HistoricalData(**item) for item in data_list]
        session.add_all(objects)
        session.commit()
        print(f"{len(objects)} registros de datos históricos añadidos.")
        return objects
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error al añadir datos históricos: {e}")
        return []

def get_historical_data(session, symbol, timeframe, start_date=None, end_date=None, limit=None, order_desc=True):
    try:
        query = session.query(HistoricalData).filter_by(symbol=symbol, timeframe=timeframe)
        if start_date:
            query = query.filter(HistoricalData.timestamp >= start_date)
        if end_date:
            query = query.filter(HistoricalData.timestamp <= end_date)

        if order_desc:
            query = query.order_by(HistoricalData.timestamp.desc()) # Más común para obtener los últimos
        else:
            query = query.order_by(HistoricalData.timestamp.asc())

        if limit:
            query = query.limit(limit)

        results = query.all()
        print(f"Recuperados {len(results)} registros para {symbol}/{timeframe}.")
        return results
    except SQLAlchemyError as e:
        print(f"Error al obtener datos históricos: {e}")
        return []

def add_trade(session, trade_data):
    """
    Añade un nuevo trade. trade_data es un diccionario.
    'order_type' y 'status' pueden ser strings o los Enums correspondientes.
    """
    try:
        # Convertir strings a Enums si es necesario
        if 'order_type' in trade_data and isinstance(trade_data['order_type'], str):
            trade_data['order_type'] = OrderTypeEnum[trade_data['order_type'].upper()]
        if 'status' in trade_data and isinstance(trade_data['status'], str):
            trade_data['status'] = TradeStatusEnum[trade_data['status'].upper()]

        trade = Trade(**trade_data)
        session.add(trade)
        session.commit()
        session.refresh(trade)
        print(f"Trade añadido: {trade}")
        return trade
    except (SQLAlchemyError, KeyError) as e: # KeyError si el string no es un Enum válido
        session.rollback()
        print(f"Error al añadir trade: {e}")
        return None

def get_trade_by_id(session, trade_id):
    try:
        return session.query(Trade).filter(Trade.id == trade_id).first()
    except SQLAlchemyError as e:
        print(f"Error al obtener trade por ID: {e}")
        return None

def update_trade_status(session, trade_id, new_status, profit=None):
    """ 'new_status' puede ser string o Enum """
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if trade:
            if isinstance(new_status, str):
                trade.status = TradeStatusEnum[new_status.upper()]
            else:
                trade.status = new_status # Asume que es el Enum correcto

            if profit is not None:
                trade.profit = profit
            session.commit()
            print(f"Trade {trade_id} actualizado a estado {trade.status.value if trade.status else 'N/A'}.")
            return trade
        return None
    except (SQLAlchemyError, KeyError) as e:
        session.rollback()
        print(f"Error al actualizar trade: {e}")
        return None

def add_model_version(session, model_name, version, file_path, parameters=None, performance_metrics=None):
    try:
        model_version = ModelVersion(
            model_name=model_name, version=version, file_path=file_path,
            parameters=parameters, performance_metrics=performance_metrics
        )
        session.add(model_version)
        session.commit()
        session.refresh(model_version)
        print(f"Versión de modelo añadida: {model_version}")
        return model_version
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error al añadir versión de modelo: {e}")
        return None

def get_latest_model_version(session, model_name):
    try:
        # El índice ix_modelversions_name_timestamp debería ayudar aquí
        return session.query(ModelVersion)            .filter(ModelVersion.model_name == model_name)            .order_by(ModelVersion.timestamp.desc())            .first()
    except SQLAlchemyError as e:
        print(f"Error al obtener la última versión del modelo: {e}")
        return None

# Ejemplo de uso (para pruebas)
if __name__ == "__main__":
    print(f"Inicializando la base de datos en: {DATABASE_URL}")
    # Eliminar la BD existente para probar la creación de índices desde cero
    if os.path.exists(DATABASE_NAME):
        print(f"Eliminando base de datos existente: {DATABASE_NAME}")
        os.remove(DATABASE_NAME)

    initialize_database()

    db_session_gen = get_db_session()
    session = next(db_session_gen)
    try:
        print("\n--- Probando HistoricalData con nuevo índice ---")
        sample_data = [
            {'timestamp': datetime.datetime(2023, 1, 1, 10, 0, 0), 'symbol': 'EURUSD', 'timeframe': 'H1', 'open': 1.0500, 'high': 1.0510, 'low': 1.0490, 'close': 1.0505, 'volume': 1000},
            {'timestamp': datetime.datetime(2023, 1, 1, 11, 0, 0), 'symbol': 'EURUSD', 'timeframe': 'H1', 'open': 1.0505, 'high': 1.0515, 'low': 1.0500, 'close': 1.0510, 'volume': 1200},
        ]
        add_historical_data(session, sample_data)
        eurusd_data = get_historical_data(session, symbol='EURUSD', timeframe='H1', limit=5, order_desc=True)
        for row in eurusd_data: print(row)

        print("\n--- Probando Trade con Enums e índices ---")
        new_trade_data_buy = {
            'symbol': 'EURUSD', 'order_type': OrderTypeEnum.BUY, 'price': 1.0510,
            'volume': 0.1, 'status': TradeStatusEnum.OPEN, 'mt5_ticket_id': 12345
        }
        trade1 = add_trade(session, new_trade_data_buy)

        new_trade_data_sell_str = { # Usando strings para Enums
            'symbol': 'EURUSD', 'order_type': "SELL", 'price': 1.0520,
            'volume': 0.05, 'status': "SIMULATED_OPEN", 'mt5_ticket_id': 12346
        }
        trade2 = add_trade(session, new_trade_data_sell_str)

        if trade1:
            retrieved_trade = get_trade_by_id(session, trade1.id)
            print(f"Trade recuperado: ID {retrieved_trade.id}, Tipo {retrieved_trade.order_type.value}, Estado {retrieved_trade.status.value}")
            update_trade_status(session, trade1.id, TradeStatusEnum.CLOSED, profit=25.50)
            retrieved_trade_updated = get_trade_by_id(session, trade1.id)
            print(f"Trade actualizado: Estado {retrieved_trade_updated.status.value}, Profit: {retrieved_trade_updated.profit}")

        print("\n--- Probando ModelVersion con nuevo índice ---")
        params = json.dumps({'learning_rate': 0.1})
        model_v1 = add_model_version(session, model_name='LGBM_EURUSD_H1', version='v2.0.0', file_path='/models/v2.pkl', parameters=params)
        model_v2 = add_model_version(session, model_name='LGBM_EURUSD_H1', version='v2.0.1', file_path='/models/v2.1.pkl', parameters=params)

        latest_model = get_latest_model_version(session, 'LGBM_EURUSD_H1')
        if latest_model:
            print(f"Último modelo para LGBM_EURUSD_H1: {latest_model.version}")

    except Exception as e:
        print(f"Ocurrió un error durante las pruebas: {e}")
    finally:
        session.close()
        print("\nPruebas de database_manager (con optimizaciones) finalizadas.")
