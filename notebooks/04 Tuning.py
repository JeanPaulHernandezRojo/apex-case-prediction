# Databricks notebook source
# MAGIC %md
# MAGIC # Hyperopt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import sys

sys.path.append('/Workspace/Users/jhernandezr@uni.pe/Apex/apex-case-prediction')

# COMMAND ----------

from src.models import (
    split_temporal,
    preparar_datos_modelo,
    calcular_metricas,
    calcular_lift_score
)
from src.features import target_encoding
from src.config import DATA_CONFIG, MODEL_CONFIG, MLFLOW_CONFIG

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import mlflow
import mlflow.lightgbm

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carga de Datos

# COMMAND ----------

df_spark = spark.sql("""
    SELECT a.* EXCEPT(ultimo_canal)
    FROM hive_metastore.fs.apex_digital_features a
""")

df = df_spark.toPandas()

# COMMAND ----------

# Split temporal
df_train, df_test = split_temporal(
    df, 
    DATA_CONFIG.col_fecha,
    MODEL_CONFIG.test_size_months
)

print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

# COMMAND ----------

# Target encoding de país
df_train, df_test = target_encoding(
    df_train, df_test, 
    DATA_CONFIG.col_pais, 
    'target',
    min_samples=50
)

# COMMAND ----------

# Preparar datos
features_categoricos = [
    'ultimo_canal',
    'penultimo_canal', 
    'transicion_canal',
    DATA_CONFIG.col_madurez,
    DATA_CONFIG.col_frecuencia
]

X_train, y_train, feature_names = preparar_datos_modelo(
    df_train, 'target', features_categoricos
)

X_test, y_test, _ = preparar_datos_modelo(
    df_test, 'target', features_categoricos
)

# COMMAND ----------

# Split validación
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=MODEL_CONFIG.random_state,
    stratify=y_train
)

print(f"Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definición del Espacio de Búsqueda

# COMMAND ----------

# MAGIC %md
# MAGIC Espacio de hiperparámetros para LightGBM

# COMMAND ----------

search_space = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': MODEL_CONFIG.random_state,
    
    # Hiperparámetros a optimizar
    'num_leaves': scope.int(hp.quniform('num_leaves', 20, 100, 5)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
    
    'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
    'bagging_freq': scope.int(hp.quniform('bagging_freq', 1, 10, 1)),
    
    'min_child_samples': scope.int(hp.quniform('min_child_samples', 10, 100, 10)),
    'min_child_weight': hp.loguniform('min_child_weight', np.log(1e-3), np.log(10)),
    
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-3), np.log(10)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-3), np.log(10)),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Función Objetivo

# COMMAND ----------

def objective(params):
    """
    Función objetivo para Hyperopt.
    Entrena LightGBM y retorna -AUC (para minimizar).
    
    Args:
        params: Diccionario de hiperparámetros
        
    Returns:
        Dict con loss y status
    """
    with mlflow.start_run(nested=True):
        # Log params
        mlflow.log_params(params)
        
        # Crear datasets
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Entrenar
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Predecir en validación
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Calcular AUC
        auc = roc_auc_score(y_val, y_pred)
        
        # Log métrica
        mlflow.log_metric('auc_val', auc)
        mlflow.log_metric('best_iteration', model.best_iteration)
        
    # Retornar negativo porque hyperopt minimiza
    return {'loss': -auc, 'status': STATUS_OK, 'auc': auc}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimización con Hyperopt

# COMMAND ----------

# MAGIC %md
# MAGIC Ejecutar optimización bayesiana con TPE

# COMMAND ----------

# Configurar MLflow
mlflow.set_experiment(MLFLOW_CONFIG.experiment_name + "_tuning")

# COMMAND ----------

# Crear objeto Trials para tracking
trials = Trials()

# COMMAND ----------

# Optimizar
with mlflow.start_run(run_name="hyperopt_tuning"):
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(MODEL_CONFIG.random_state)
    )
    
    # Convertir params a formato correcto
    best_params = space_eval(search_space, best_params)
    
    # Log mejores parámetros
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    
    # Mejor AUC
    best_auc = -min([trial['result']['loss'] for trial in trials.trials])
    mlflow.log_metric("best_auc_val", best_auc)
    
    print("=" * 60)
    print("OPTIMIZACIÓN COMPLETADA")
    print("=" * 60)
    print(f"Mejor AUC en validación: {best_auc:.4f}")
    print(f"Iteraciones realizadas: {len(trials.trials)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análisis de Resultados

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mejores Hiperparámetros

# COMMAND ----------

import json

print("Mejores hiperparámetros encontrados:")
print("=" * 60)
for key, value in sorted(best_params.items()):
    if key not in ['objective', 'metric', 'boosting_type', 'verbose', 'random_state']:
        print(f"{key:25s}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convergencia de la Optimización

# COMMAND ----------

import matplotlib.pyplot as plt

# Extraer AUC de cada trial
aucs = [-trial['result']['loss'] for trial in trials.trials]
best_auc_so_far = np.maximum.accumulate(aucs)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, len(aucs) + 1), aucs, 'o-', alpha=0.6, label='AUC por trial')
ax.plot(range(1, len(aucs) + 1), best_auc_so_far, 'r-', linewidth=2, label='Mejor AUC hasta ahora')
ax.set_xlabel('Iteración')
ax.set_ylabel('AUC-ROC')
ax.set_title('Convergencia de Hyperopt', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución de Hiperparámetros Probados

# COMMAND ----------

# Extraer hiperparámetros de todos los trials
trials_df = pd.DataFrame([
    {**trial['misc']['vals'], 'auc': -trial['result']['loss']}
    for trial in trials.trials
])

# Convertir listas a valores únicos
for col in trials_df.columns:
    if col != 'auc':
        trials_df[col] = trials_df[col].apply(lambda x: x[0] if isinstance(x, list) else x)

# COMMAND ----------

# Top 10 mejores configuraciones
trials_df.nlargest(10, 'auc')[['learning_rate', 'num_leaves', 'max_depth', 'auc']].round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entrenamiento del Modelo Final

# COMMAND ----------

# MAGIC %md
# MAGIC Entrenar modelo con mejores hiperparámetros en train completo

# COMMAND ----------

with mlflow.start_run(run_name="best_model_final"):
    # Log params
    mlflow.log_params(best_params)
    mlflow.log_param("training_set", "train_full")
    
    # Datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Entrenar con mejores parámetros
    best_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predicciones en test
    y_pred_test = best_model.predict(X_test, num_iteration=best_model.best_iteration)
    
    # Métricas en test
    metricas_test = calcular_metricas(y_test, y_pred_test, MODEL_CONFIG.threshold)
    lift_test = calcular_lift_score(y_test, y_pred_test, MODEL_CONFIG.top_k_percent)
    metricas_test['lift_score'] = lift_test
    
    # Log métricas
    mlflow.log_metrics({f"{k}_test": v for k, v in metricas_test.items() if isinstance(v, (int, float))})
    
    # Log modelo
    mlflow.lightgbm.log_model(best_model, "best_model")
    
    print("\n" + "=" * 60)
    print("MÉTRICAS EN TEST SET")
    print("=" * 60)
    print(f"ROC-AUC:           {metricas_test['roc_auc']:.4f}")
    print(f"Avg Precision:     {metricas_test['avg_precision']:.4f}")
    print(f"Precision:         {metricas_test['precision']:.4f}")
    print(f"Recall:            {metricas_test['recall']:.4f}")
    print(f"F1-Score:          {metricas_test['f1_score']:.4f}")
    print(f"Lift (Top 20%):    {metricas_test['lift_score']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparación con Modelo Baseline

# COMMAND ----------

# MAGIC %md
# MAGIC Comparar modelo optimizado vs modelo con hiperparámetros por defecto

# COMMAND ----------

# Entrenar modelo baseline (params por defecto)
baseline_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'random_state': MODEL_CONFIG.random_state,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

baseline_model = lgb.train(
    baseline_params,
    train_data,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)

y_pred_baseline = baseline_model.predict(X_test, num_iteration=baseline_model.best_iteration)

# COMMAND ----------

# Métricas baseline
metricas_baseline = calcular_metricas(y_test, y_pred_baseline, MODEL_CONFIG.threshold)
metricas_baseline['lift_score'] = calcular_lift_score(y_test, y_pred_baseline, MODEL_CONFIG.top_k_percent)

# COMMAND ----------

# Comparación
comparison = pd.DataFrame({
    'Baseline': metricas_baseline,
    'Optimizado': metricas_test,
    'Mejora': pd.Series(metricas_test) - pd.Series(metricas_baseline)
}).T

comparison = comparison[['roc_auc', 'avg_precision', 'precision', 'recall', 'f1_score', 'lift_score']]
comparison.round(4)

# COMMAND ----------

# Visualización de comparación
fig, ax = plt.subplots(figsize=(12, 6))

metrics_to_plot = ['roc_auc', 'avg_precision', 'precision', 'recall', 'f1_score']
x = np.arange(len(metrics_to_plot))
width = 0.35

baseline_vals = [metricas_baseline[m] for m in metrics_to_plot]
optimized_vals = [metricas_test[m] for m in metrics_to_plot]

ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8, color='#FF6B6B')
ax.bar(x + width/2, optimized_vals, width, label='Optimizado', alpha=0.8, color='#4ECDC4')

ax.set_xlabel('Métrica')
ax.set_ylabel('Valor')
ax.set_title('Comparación: Baseline vs Optimizado', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for i, (b, o) in enumerate(zip(baseline_vals, optimized_vals)):
    diff = o - b
    color = 'green' if diff > 0 else 'red'
    ax.text(i, max(b, o) + 0.02, f'+{diff:.3f}' if diff > 0 else f'{diff:.3f}',
            ha='center', fontsize=9, color=color, fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance del Modelo Optimizado

# COMMAND ----------

importance = best_model.feature_importance(importance_type='gain')
fi_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
fi_df.set_index('feature')['importance'].plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 20 Features - Modelo Optimizado', fontsize=14, fontweight='bold')
ax.set_xlabel('Importancia (Gain)')
plt.tight_layout()
plt.show()
