import pandas as pd
import numpy as np
from typing import Tuple, List
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import DATA_CONFIG, FEATURE_CONFIG

def calcular_features_historicos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features basados en comportamiento histórico del cliente.
    
    Args:
        df: DataFrame transaccional
        
    Returns:
        DataFrame con features históricos agregados por cliente
    """
    
    # Features por cliente
    cliente_features = df.groupby(DATA_CONFIG.col_cliente_id).agg({
        DATA_CONFIG.col_canal: 'count',
        DATA_CONFIG.col_facturacion: ['mean', 'std', 'sum'],
        DATA_CONFIG.col_materiales: ['mean', 'std'],
        DATA_CONFIG.col_dias_ultimo: 'mean'
    })
    
    cliente_features.columns = [
        'total_pedidos',
        'facturacion_promedio',
        'facturacion_std',
        'facturacion_total',
        'materiales_promedio',
        'materiales_std',
        'dias_ultimo_promedio'
    ]
    
    # Porcentaje de pedidos digitales histórico
    canal_digital = DATA_CONFIG.canal_digital
    
    pedidos_digitales = df[df[DATA_CONFIG.col_canal] == canal_digital].groupby(
        DATA_CONFIG.col_cliente_id
    ).size()

    cliente_features['pedidos_digitales_total'] = pedidos_digitales.fillna(0)

    cliente_features['pct_digital_historico'] = (
        cliente_features['pedidos_digitales_total'] / cliente_features['total_pedidos']
    )
    
    # Features por canal
    for canal in [DATA_CONFIG.canal_digital, DATA_CONFIG.canal_telefono, DATA_CONFIG.canal_vendedor]:
        canal_data = df[df[DATA_CONFIG.col_canal] == canal]
        facturacion_canal = canal_data.groupby(DATA_CONFIG.col_cliente_id)[
            DATA_CONFIG.col_facturacion
        ].mean()
        cliente_features[f'facturacion_promedio_{canal.lower()}'] = facturacion_canal
        
        materiales_canal = canal_data.groupby(DATA_CONFIG.col_cliente_id)[
            DATA_CONFIG.col_materiales
        ].mean()
        cliente_features[f'materiales_promedio_{canal.lower()}'] = materiales_canal
    
    cliente_features.fillna(0, inplace=True)
    
    return cliente_features.reset_index()


def calcular_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features basados en tendencias temporales recientes.
    
    Args:
        df: DataFrame transaccional.
        
    Returns:
        DataFrame con features temporales por cliente
    """
    df_sorted = df.sort_values([DATA_CONFIG.col_cliente_id, DATA_CONFIG.col_fecha])
    canal_digital = DATA_CONFIG.canal_digital
    
    # Fecha máxima de referencia
    fecha_max = df[DATA_CONFIG.col_fecha].max()
    fecha_ventana_90d = fecha_max - pd.Timedelta(days=FEATURE_CONFIG.ventana_tendencia_dias)
    fecha_ventana_30d = fecha_max - pd.Timedelta(days=FEATURE_CONFIG.ventana_reciente_dias)
    
    # Filtros temporales
    df_90d = df[df[DATA_CONFIG.col_fecha] >= fecha_ventana_90d]
    df_30d = df[df[DATA_CONFIG.col_fecha] >= fecha_ventana_30d]
    
    # Porcentaje digital en últimos 90 días
    pedidos_90d = df_90d.groupby(DATA_CONFIG.col_cliente_id).size()
    digitales_90d = df_90d[df_90d[DATA_CONFIG.col_canal] == canal_digital].groupby(
        DATA_CONFIG.col_cliente_id
    ).size()
    pct_digital_90d = (digitales_90d / pedidos_90d).fillna(0)
    
    # Porcentaje digital en últimos 30 días
    pedidos_30d = df_30d.groupby(DATA_CONFIG.col_cliente_id).size()
    digitales_30d = df_30d[df_30d[DATA_CONFIG.col_canal] == canal_digital].groupby(
        DATA_CONFIG.col_cliente_id
    ).size()
    pct_digital_30d = (digitales_30d / pedidos_30d).fillna(0)
    
    # Tendencia (aceleración digital)
    tendencia_digital = pct_digital_30d - pct_digital_90d
    
    # Días desde último pedido digital
    ultimos_digitales = df[df[DATA_CONFIG.col_canal] == canal_digital].groupby(
        DATA_CONFIG.col_cliente_id
    )[DATA_CONFIG.col_fecha].max()
    dias_desde_ultimo_digital = (fecha_max - ultimos_digitales).dt.days
    
    # Consolidar features temporales
    # Usar los índices de las series directamente sin especificar index
    features_temp = pd.DataFrame({
        DATA_CONFIG.col_cliente_id: pedidos_90d.index,
        'pct_digital_90d': pct_digital_90d.values,
        'pct_digital_30d': pct_digital_30d.reindex(pedidos_90d.index, fill_value=0).values,
        'tendencia_digital': tendencia_digital.reindex(pedidos_90d.index, fill_value=0).values,
        'pedidos_ultimos_90d': pedidos_90d.values,
        'pedidos_ultimos_30d': pedidos_30d.reindex(pedidos_90d.index, fill_value=0).values,
        'dias_desde_ultimo_digital': dias_desde_ultimo_digital.reindex(
            pedidos_90d.index, fill_value=999
        ).values
    })
    
    return features_temp


def calcular_features_secuenciales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features basados en la secuencia de pedidos.
    
    Args:
        df: DataFrame transaccional
        
    Returns:
        DataFrame con features secuenciales por cliente
    """
    df_sorted = df.sort_values([DATA_CONFIG.col_cliente_id, DATA_CONFIG.col_fecha])
    
    # Crear lags de canal (es decir, ultimo, penultino, antepenultimo)
    for i in range(1, FEATURE_CONFIG.n_lags_canal + 1):
        df_sorted[f'canal_lag_{i}'] = df_sorted.groupby(DATA_CONFIG.col_cliente_id)[
            DATA_CONFIG.col_canal
        ].shift(i)
    
    # Ultimo canal usado (más reciente)
    ultimo_canal = df_sorted.groupby(DATA_CONFIG.col_cliente_id)[
        DATA_CONFIG.col_canal
    ].last()
    
    # Penúltimo canal
    penultimo_canal = df_sorted.groupby(DATA_CONFIG.col_cliente_id)['canal_lag_1'].last()
    
    # Transición de canal (penúltimo → último)
    transicion = penultimo_canal + '_TO_' + ultimo_canal
    transicion = transicion.fillna('UNKNOWN')
    
    # Racha actual (cuántos pedidos consecutivos en el mismo canal)
    df_sorted['cambio_canal'] = (
        df_sorted.groupby(DATA_CONFIG.col_cliente_id)[DATA_CONFIG.col_canal].shift(1) != 
        df_sorted[DATA_CONFIG.col_canal]
    ).astype(int)
    df_sorted['grupo_racha'] = df_sorted.groupby(DATA_CONFIG.col_cliente_id)[
        'cambio_canal'
    ].cumsum()

    racha_actual = df_sorted.groupby([DATA_CONFIG.col_cliente_id, 'grupo_racha']).size()
    racha_actual = racha_actual.groupby(DATA_CONFIG.col_cliente_id).last()
    
    features_seq = pd.DataFrame({
        DATA_CONFIG.col_cliente_id: ultimo_canal.index,
        'ultimo_canal': ultimo_canal.values,
        'penultimo_canal': penultimo_canal.fillna('UNKNOWN').values,
        'transicion_canal': transicion.values,
        'racha_canal_actual': racha_actual.reindex(ultimo_canal.index, fill_value=1).values
    })
    
    return features_seq


def calcular_features_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara features categóricos con encoding apropiado.
    
    Args:
        df: DataFrame con variables categóricas
        
    Returns:
        DataFrame con último valor de categóricos por cliente
    """
    df_sorted = df.sort_values([DATA_CONFIG.col_cliente_id, DATA_CONFIG.col_fecha])
    
    # Último valor de cada categórico
    categoricos = df_sorted.groupby(DATA_CONFIG.col_cliente_id).last()[[
        DATA_CONFIG.col_pais,
        DATA_CONFIG.col_madurez,
        DATA_CONFIG.col_frecuencia
    ]].reset_index()
    
    # Encoding de madurez digital basado en análisis EDA
    categoricos['madurez_digital_encoded'] = categoricos[DATA_CONFIG.col_madurez].map(
        FEATURE_CONFIG.madurez_encoding
    )
    
    return categoricos


def target_encoding(df_train: pd.DataFrame, 
                   df_test: pd.DataFrame,
                   feature_col: str,
                   target_col: str,
                   min_samples: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica target encoding a una variable categórica con suavizado.
    
    Args:
        df_train: DataFrame de entrenamiento
        df_test: DataFrame de prueba
        feature_col: Nombre de la columna a encodear
        target_col: Nombre de la columna target
        min_samples: Mínimo de muestras para suavizado
        
    Returns:
        Tupla con (df_train_encoded, df_test_encoded)
    """
    # Calcular media global
    global_mean = df_train[target_col].mean()
    
    # Calcular estadísticas por categoría
    stats = df_train.groupby(feature_col).agg({
        target_col: ['mean', 'count']
    })
    stats.columns = ['mean', 'count']
    
    # Suavizado bayesiano
    smoothing = 1 / (1 + np.exp(-(stats['count'] - min_samples) / min_samples))
    stats['smoothed_mean'] = (
        smoothing * stats['mean'] + (1 - smoothing) * global_mean
    )
    
    # Aplicar encoding
    encoding_map = stats['smoothed_mean'].to_dict()
    new_col_name = f'{feature_col}_target_encoded'
    
    df_train[new_col_name] = df_train[feature_col].map(encoding_map).fillna(global_mean)
    df_test[new_col_name] = df_test[feature_col].map(encoding_map).fillna(global_mean)
    
    return df_train, df_test


def construir_dataset_completo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    
    Args:
        df: DataFrame raw con datos transaccionales
        
    Returns:
        DataFrame con todos los features construidos a nivel cliente
    """
    # Asegurar tipos correctos
    df[DATA_CONFIG.col_fecha] = pd.to_datetime(df[DATA_CONFIG.col_fecha])
    df_sorted = df.sort_values([DATA_CONFIG.col_cliente_id, DATA_CONFIG.col_fecha])
    
    # Marcar el último pedido de cada cliente (el que queremos predecir)
    df_sorted['es_ultimo_pedido'] = False
    ultimo_idx = df_sorted.groupby(DATA_CONFIG.col_cliente_id).tail(1).index
    df_sorted.loc[ultimo_idx, 'es_ultimo_pedido'] = True
    
    # Filtrar solo clientes con al menos 2 pedidos
    pedidos_por_cliente = df_sorted.groupby(DATA_CONFIG.col_cliente_id).size()
    clientes_validos = pedidos_por_cliente[pedidos_por_cliente >= 2].index
    df_sorted = df_sorted[df_sorted[DATA_CONFIG.col_cliente_id].isin(clientes_validos)]
    
    # Calcular features EXCLUYENDO el último pedido de cada cliente
    df_sin_ultimo = df_sorted[~df_sorted['es_ultimo_pedido']].copy()
    
    # Construir features sobre pedidos históricos (sin el último)
    features_hist = calcular_features_historicos(df_sin_ultimo)
    features_temp = calcular_features_temporales(df_sin_ultimo)
    features_seq = calcular_features_secuenciales(df_sin_ultimo)
    features_cat = calcular_features_categoricos(df_sin_ultimo)
    
    # Merge de todos los features
    df_final = features_hist.merge(
        features_temp, on=DATA_CONFIG.col_cliente_id, how='left'
    ).merge(
        features_seq, on=DATA_CONFIG.col_cliente_id, how='left'
    ).merge(
        features_cat, on=DATA_CONFIG.col_cliente_id, how='left'
    )
    
    # Obtener el canal del último pedido como target
    df_ultimo_pedido = df_sorted[df_sorted['es_ultimo_pedido']][[
        DATA_CONFIG.col_cliente_id,
        DATA_CONFIG.col_canal
    ]].rename(columns={DATA_CONFIG.col_canal: 'canal_ultimo_pedido'})
    
    # Merge con target
    df_final = df_final.merge(df_ultimo_pedido, on=DATA_CONFIG.col_cliente_id, how='left')
    
    # Crear target: 1 si último pedido es digital
    df_final['target'] = (
        df_final['canal_ultimo_pedido'] == DATA_CONFIG.canal_digital
    ).astype(int)
    
    # Remover columna auxiliar
    df_final = df_final.drop(columns=['canal_ultimo_pedido'])
    
    # Llenar nulos
    df_final.fillna(0, inplace=True)
    
    print(f"{len(df_final):,} clientes")
    print(f"{df_final['target'].mean():.2%} target")
    
    return df_final
