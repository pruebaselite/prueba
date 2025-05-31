import os
import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index, Enum as SQLAlchemyEnum, Text # Text para JSON más largo
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import enum

# Cargar variables de entorno para DB_NAME
from dotenv import load_dotenv
load_dotenv()

DATABASE_NAME = os.getenv("DB_NAME", "trading_bot.db")
DATABASE_URL = f"sqlite:///{DATABASE_NAME}"

engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TradeStatusEnum(enum.Enum):
    OPEN = "OPEN"; CLOSED = "CLOSED"; CANCELLED = "CANCELLED"
    SIMULATED_OPEN = "SIMULATED_OPEN"; SIMULATED_CLOSED = "SIMULATED_CLOSED"

class OrderTypeEnum(enum.Enum):
    BUY = "BUY"; SELL = "SELL"

class HistoricalData(Base):
    __tablename__ = "historical_data"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    timeframe = Column(String, nullable=False, index=True)
    open = Column(Float, nullable=False); high = Column(Float, nullable=False)
    low = Column(Float, nullable=False); close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)
Index('ix_histdata_symbol_timeframe_timestamp', HistoricalData.symbol, HistoricalData.timeframe, HistoricalData.timestamp.desc())

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    symbol = Column(String, nullable=False, index=True)
    order_type = Column(SQLAlchemyEnum(OrderTypeEnum), nullable=False, index=True)
    price = Column(Float, nullable=False) # Precio de apertura
    volume = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    profit = Column(Float, nullable=True) # Profit/Loss del trade
    status = Column(SQLAlchemyEnum(TradeStatusEnum), default=TradeStatusEnum.OPEN, index=True)
    mt5_ticket_id = Column(Integer, nullable=True, unique=True, index=True)

    # NUEVA COLUMNA para guardar las features que llevaron a abrir el trade
    open_features_json = Column(Text, nullable=True)
    # NUEVA COLUMNA para el precio de cierre (para trades simulados)
    close_price = Column(Float, nullable=True)
    # NUEVA COLUMNA para el timestamp de cierre
    close_timestamp = Column(DateTime, nullable=True)


Index('ix_trades_symbol_status', Trade.symbol, Trade.status)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False, unique=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    file_path = Column(String, nullable=False)
    parameters = Column(Text, nullable=True) # Usar Text para JSON más largo
    performance_metrics = Column(Text, nullable=True) # Usar Text
Index('ix_modelversions_name_timestamp', ModelVersion.model_name, ModelVersion.timestamp.desc())

def initialize_database():
    try:
        Base.metadata.create_all(bind=engine)
        print("BD inicializada y tablas (con nuevas columnas en Trade) creadas/actualizadas.")
    except SQLAlchemyError as e: print(f"Error inicializando BD: {e}")

def get_db_session():
    db = SessionLocal();
    try: # Corrected: remove "처리:" which seems to be a copy-paste artifact
        yield db
    finally:
        db.close()

def add_historical_data(session, data_list):
    # ... (sin cambios) ...
    try:
        objects = [HistoricalData(**item) for item in data_list]
        session.add_all(objects); session.commit(); return objects
    except SQLAlchemyError as e: session.rollback(); print(f"Error añadiendo datos hist: {e}"); return []

def get_historical_data(session, symbol, timeframe, start_date=None, end_date=None, limit=None, order_desc=True):
    # ... (sin cambios) ...
    try:
        q = session.query(HistoricalData).filter_by(symbol=symbol,timeframe=timeframe)
        if start_date: q=q.filter(HistoricalData.timestamp >= start_date)
        if end_date: q=q.filter(HistoricalData.timestamp <= end_date)
        q = q.order_by(HistoricalData.timestamp.desc() if order_desc else HistoricalData.timestamp.asc())
        if limit: q=q.limit(limit)
        return q.all()
    except SQLAlchemyError as e: print(f"Error obteniendo datos hist: {e}"); return []

def add_trade(session, trade_data):
    # ... (modificado para aceptar open_features_json) ...
    try:
        if 'order_type' in trade_data and isinstance(trade_data['order_type'],str): trade_data['order_type']=OrderTypeEnum[trade_data['order_type'].upper()]
        if 'status' in trade_data and isinstance(trade_data['status'],str): trade_data['status']=TradeStatusEnum[trade_data['status'].upper()]
        trade = Trade(**trade_data); session.add(trade); session.commit(); session.refresh(trade)
        print(f"Trade añadido: ID {trade.id}, Features guardadas: {'Sí' if trade.open_features_json else 'No'}")
        return trade
    except (SQLAlchemyError, KeyError) as e: session.rollback(); print(f"Error añadiendo trade: {e}"); return None

def update_trade(session, trade_id, update_data):
    """Actualiza un trade existente con los datos proporcionados en update_data."""
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if trade:
            for key, value in update_data.items():
                if key == 'status' and isinstance(value, str): # Convertir string de status a Enum
                    setattr(trade, key, TradeStatusEnum[value.upper()])
                elif key == 'order_type' and isinstance(value, str): # Aunque order_type no debería cambiar
                     setattr(trade, key, OrderTypeEnum[value.upper()])
                else:
                    setattr(trade, key, value)
            session.commit()
            print(f"Trade {trade_id} actualizado.")
            return trade
        return None
    except (SQLAlchemyError, KeyError) as e:
        session.rollback(); print(f"Error actualizando trade {trade_id}: {e}"); return None

def get_open_trades(session, symbol=None):
    """Obtiene todos los trades con estado OPEN o SIMULATED_OPEN."""
    try:
        query = session.query(Trade).filter(Trade.status.in_([TradeStatusEnum.OPEN, TradeStatusEnum.SIMULATED_OPEN]))
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        return query.all()
    except SQLAlchemyError as e:
        print(f"Error obteniendo trades abiertos: {e}"); return []

# ... (get_trade_by_id, add_model_version, get_latest_model_version sin cambios significativos) ...
def get_trade_by_id(session, trade_id):
    try: return session.query(Trade).filter(Trade.id == trade_id).first()
    except SQLAlchemyError as e: print(f"Error obteniendo trade ID {trade_id}: {e}"); return None

def add_model_version(session,model_name,version,file_path,parameters=None,performance_metrics=None):
    try:
        mv=ModelVersion(model_name=model_name,version=version,file_path=file_path,parameters=parameters,performance_metrics=performance_metrics)
        session.add(mv);session.commit();session.refresh(mv);return mv
    except SQLAlchemyError as e:session.rollback();print(f"Error añadiendo ModelVersion: {e}");return None

def get_latest_model_version(session, model_name):
    try:
        return session.query(ModelVersion).filter(ModelVersion.model_name==model_name).order_by(ModelVersion.timestamp.desc()).first()
    except SQLAlchemyError as e:print(f"Error obteniendo latest ModelVersion: {e}");return None

if __name__ == "__main__":
    print(f"Inicializando BD en: {DATABASE_URL}")
    if os.path.exists(DATABASE_NAME): os.remove(DATABASE_NAME) # Limpiar para prueba de esquema
    initialize_database()
    db_s = next(get_db_session())
    try:
        # Prueba de nueva columna en Trade
        test_features = pd.DataFrame([{'feat1': 0.5, 'feat2': 0.7}]).to_json(orient='records')
        add_trade(db_s, {'symbol':'EURUSD','order_type':OrderTypeEnum.BUY,'price':1.1,'volume':0.1,'status':TradeStatusEnum.SIMULATED_OPEN, 'open_features_json': test_features})
        open_trades = get_open_trades(db_s, 'EURUSD')
        if open_trades: print(f"Trade abierto encontrado: {open_trades[0].id}, Features: {open_trades[0].open_features_json}")
        update_trade(db_s, open_trades[0].id, {'status': TradeStatusEnum.SIMULATED_CLOSED, 'profit': 10.0, 'close_price': 1.11, 'close_timestamp': datetime.datetime.utcnow()})
        closed_trade = get_trade_by_id(db_s, open_trades[0].id)
        if closed_trade : print(f"Trade cerrado: {closed_trade.status.value if closed_trade.status else ''}, Profit: {closed_trade.profit}")

    finally: db_s.close(); print("Pruebas DB (con open_features_json) finalizadas.")
