from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DataConfig:
    """Config para de tablas y columnas."""

    input_data_table: str = "hive_metastore.silver.m_transaction_slv"
    
    # Columnas que consideraré
    col_cliente_id: str = "cliente_id"
    col_fecha: str = "fecha_pedido_dt"
    col_canal: str = "canal_pedido_cd"
    col_facturacion: str = "facturacion_usd_val"
    col_materiales: str = "materiales_distintos_val"
    col_dias_ultimo: str = "dias_desde_ultimo_pedido"
    col_frecuencia: str = "frecuencia_visitas_cd"
    col_pais: str = "pais_cd"
    col_madurez: str = "madurez_digital_cd"
    
    # Descripcion de los canales
    canal_digital: str = "DIGITAL"
    canal_telefono: str = "TELEFONO"
    canal_vendedor: str = "VENDEDOR"


@dataclass
class FeatureConfig:
    """Config para feature engineering"""

    ventana_tendencia_dias: int = 90
    ventana_reciente_dias: int = 30
    n_lags_canal: int = 3 # los ultimos 3 pedidos
    min_pedidos_cliente: int = 2 # filtro para clientes con pocos pedidos, con esto quito rapidamente clientes con datos menos confiables
    
    madurez_encoding: Dict[str, float] = None # Mapping para codificar la madurez digital
    
    def __post_init__(self):
        if self.madurez_encoding is None:
            self.madurez_encoding = {
                'ALTA': 0.75,   # EDA: 75% pedidos digitales
                'MEDIA': 0.55,  # EDA: 55% pedidos digitales
                'BAJA': 0.35    # EDA: 35% pedidos digitales
            }


@dataclass
class ModelConfig:
    """Config para modelo base (POC) + entrenamiento"""

    test_size_months: int = 1
    random_state: int = 42
    
    lgbm_params: Dict = None # setearé en notebook
    
    threshold: float = 0.5 # threshold base
    
    top_k_percent: float = 0.20 # para el lift en el modeling
    
    def __post_init__(self):
        if self.lgbm_params is None:
            self.lgbm_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }


@dataclass
class MLflowConfig:
    """Config para MLflow"""

    experiment_name: str = "/Users/jhernandezr@uni.pe/apex_test"
    model_name: str = "digital_propensity_model"
    
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {
                'project': 'apex-digital',
                'team': 'data-science'
            }

# Instancias globales de configuración
DATA_CONFIG = DataConfig()
FEATURE_CONFIG = FeatureConfig()
MODEL_CONFIG = ModelConfig()
MLFLOW_CONFIG = MLflowConfig()
