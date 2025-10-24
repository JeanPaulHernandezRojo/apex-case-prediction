"""
Módulo de modelado para predicción de canal digital.
Incluye entrenamiento, evaluación y tracking con MLflow.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix,
    classification_report
)
import mlflow
import mlflow.lightgbm

from src.config import DATA_CONFIG, MODEL_CONFIG, MLFLOW_CONFIG


def split_temporal(df: pd.DataFrame, 
                   fecha_col: str,
                   test_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza split temporal de los datos para validación realista.
    
    Args:
        df: DataFrame con todos los datos
        fecha_col: Nombre de la columna de fecha
        test_months: Número de meses para conjunto de prueba
        
    Returns:
        Tupla (df_train, df_test)
    """
    df_sorted = df.sort_values(fecha_col)
    fecha_max = df_sorted[fecha_col].max()
    fecha_corte = fecha_max - pd.DateOffset(months=test_months)
    
    df_train = df_sorted[df_sorted[fecha_col] < fecha_corte]
    df_test = df_sorted[df_sorted[fecha_col] >= fecha_corte]
    
    return df_train, df_test


def preparar_datos_modelo(df: pd.DataFrame,
                          target_col: str,
                          features_categoricos: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara features y target para el modelo.
    
    Args:
        df: DataFrame con features y target
        target_col: Nombre de la columna target
        features_categoricos: Lista de features categóricos para one-hot encoding
        
    Returns:
        Tupla (X, y, feature_names)
    """
    # Separar features y target
    y = df[target_col]
    
    # Columnas a excluir
    excluir = [
        target_col, 
        DATA_CONFIG.col_cliente_id,
        DATA_CONFIG.col_fecha,
        'proximo_canal'
    ]
    
    # Features numéricos
    features_numericos = df.select_dtypes(include=[np.number]).columns.tolist()
    features_numericos = [f for f in features_numericos if f not in excluir]
    
    # One-hot encoding de categóricos
    X = df[features_numericos].copy()
    
    for col in features_categoricos:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def entrenar_lightgbm(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_val: pd.DataFrame,
                      y_val: pd.Series,
                      params: Dict[str, Any]) -> lgb.Booster:
    """
    Entrena modelo LightGBM con early stopping.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        params: Diccionario con parámetros de LightGBM
        
    Returns:
        Modelo entrenado
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def calcular_metricas(y_true: np.ndarray,
                     y_pred_proba: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calcula métricas de evaluación del modelo.
    
    Args:
        y_true: Valores reales
        y_pred_proba: Probabilidades predichas
        threshold: Umbral de clasificación
        
    Returns:
        Diccionario con métricas
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Average Precision (PR-AUC)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Métricas derivadas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metricas = {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }
    
    return metricas


def calcular_lift_score(y_true: np.ndarray,
                       y_pred_proba: np.ndarray,
                       top_k_percent: float) -> float:
    """
    Calcula lift score en el top K% de predicciones.
    Métrica clave para segmentación comercial.
    
    Args:
        y_true: Valores reales
        y_pred_proba: Probabilidades predichas
        top_k_percent: Porcentaje top a evaluar (ej: 0.20 para top 20%)
        
    Returns:
        Lift score
    """
    # Ordenar por probabilidad descendente
    df_lift = pd.DataFrame({
        'true': y_true,
        'proba': y_pred_proba
    }).sort_values('proba', ascending=False)
    
    # Tomar top K%
    n_top = int(len(df_lift) * top_k_percent)
    top_k = df_lift.head(n_top)
    
    # Calcular lift
    precision_top_k = top_k['true'].mean()
    baseline = y_true.mean()
    lift = precision_top_k / baseline if baseline > 0 else 0
    
    return lift


def segmentar_clientes_propension(df: pd.DataFrame,
                                  proba_col: str,
                                  n_segmentos: int) -> pd.DataFrame:
    """
    Segmenta clientes en cuartiles o deciles de propensión.
    Decil 1 = Mayor propensión (mejor score)
    Decil 10 = Menor propensión (peor score)
    
    Args:
        df: DataFrame con predicciones
        proba_col: Nombre de columna con probabilidades
        n_segmentos: Número de segmentos (4=cuartiles, 10=deciles)
        
    Returns:
        DataFrame con columna de segmento
    """
    if n_segmentos == 10:
        # Invertir orden: D1 = mejor, D10 = peor
        labels = ['D10', 'D9', 'D8', 'D7', 'D6', 'D5', 'D4', 'D3', 'D2', 'D1']
    else:
        labels = [f'S{i+1}' for i in range(n_segmentos)]
    
    df['segmento_propension'] = pd.qcut(
        df[proba_col],
        q=n_segmentos,
        labels=labels,
        duplicates='drop'
    )
    
    return df


def obtener_feature_importance(model: lgb.Booster,
                               feature_names: List[str],
                               top_n: int = 20) -> pd.DataFrame:
    """
    Obtiene feature importance del modelo.
    
    Args:
        model: Modelo LightGBM entrenado
        feature_names: Nombres de features
        top_n: Número de features más importantes a retornar
        
    Returns:
        DataFrame con feature importance ordenado
    """
    importance = model.feature_importance(importance_type='gain')
    
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    return fi_df


def pipeline_entrenamiento_completo(df_train: pd.DataFrame,
                                   df_test: pd.DataFrame,
                                   features_categoricos: List[str],
                                   usar_mlflow: bool = True) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Pipeline completo de entrenamiento con tracking de MLflow.
    
    Args:
        df_train: Datos de entrenamiento
        df_test: Datos de prueba
        features_categoricos: Lista de features categóricos
        usar_mlflow: Si se debe usar MLflow para tracking
        
    Returns:
        Tupla (modelo_entrenado, diccionario_resultados)
    """
    if usar_mlflow:
        mlflow.set_experiment(MLFLOW_CONFIG.experiment_name)
        mlflow.start_run(tags=MLFLOW_CONFIG.tags)
    
    # Preparar datos
    X_train, y_train, feature_names = preparar_datos_modelo(
        df_train, 'target', features_categoricos
    )
    X_test, y_test, _ = preparar_datos_modelo(
        df_test, 'target', features_categoricos
    )
    
    # Split validación para early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=MODEL_CONFIG.random_state,
        stratify=y_train
    )
    
    if usar_mlflow:
        mlflow.log_param("n_train", len(X_tr))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_params(MODEL_CONFIG.lgbm_params)
    
    # Entrenar modelo
    model = entrenar_lightgbm(X_tr, y_tr, X_val, y_val, MODEL_CONFIG.lgbm_params)
    
    # Predicciones
    y_pred_proba_test = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Métricas
    metricas = calcular_metricas(y_test, y_pred_proba_test, MODEL_CONFIG.threshold)
    lift = calcular_lift_score(y_test, y_pred_proba_test, MODEL_CONFIG.top_k_percent)
    metricas['lift_score'] = lift
    
    if usar_mlflow:
        mlflow.log_metrics(metricas)
    
    # Feature importance
    fi_df = obtener_feature_importance(model, feature_names, top_n=20)
    
    if usar_mlflow:
        mlflow.log_text(fi_df.to_string(), "feature_importance.txt")
        mlflow.lightgbm.log_model(model, "model")
        mlflow.end_run()
    
    # Resultados
    resultados = {
        'metricas': metricas,
        'feature_importance': fi_df,
        'feature_names': feature_names,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba_test
    }
    
    return model, resultados


def generar_predicciones_nuevos_clientes(model: lgb.Booster,
                                        df_nuevos: pd.DataFrame,
                                        feature_names: List[str]) -> pd.DataFrame:
    """
    Genera predicciones para nuevos clientes.
    
    Args:
        model: Modelo entrenado
        df_nuevos: DataFrame con features de nuevos clientes
        feature_names: Nombres de features del modelo
        
    Returns:
        DataFrame con cliente_id y probabilidad predicha
    """
    X_nuevos = df_nuevos[feature_names]
    predicciones = model.predict(X_nuevos, num_iteration=model.best_iteration)
    
    resultado = pd.DataFrame({
        DATA_CONFIG.col_cliente_id: df_nuevos[DATA_CONFIG.col_cliente_id],
        'propension_digital': predicciones
    })
    
    resultado = segmentar_clientes_propension(resultado, 'propension_digital')
    
    return resultado