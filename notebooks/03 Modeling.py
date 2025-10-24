# Databricks notebook source
# MAGIC %md
# MAGIC # Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('/Workspace/Users/jhernandezr@uni.pe/Apex/apex-case-prediction')

# COMMAND ----------

from src.models import (
    split_temporal,
    preparar_datos_modelo,
    entrenar_lightgbm,
    calcular_metricas,
    calcular_lift_score,
    segmentar_clientes_propension,
    obtener_feature_importance,
    pipeline_entrenamiento_completo,
    generar_predicciones_nuevos_clientes
)
from src.features import target_encoding
from src.config import DATA_CONFIG, MODEL_CONFIG, MLFLOW_CONFIG

# COMMAND ----------

import mlflow
import lightgbm as lgb
from sklearn.metrics import roc_curve, precision_recall_curve

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carga de Dataset Procesado

# COMMAND ----------

# df_spark = spark.read.table("hive_metastore.fs.apex_digital_features")

# df = df_spark.toPandas()

# COMMAND ----------

df_spark = spark.sql("""
    SELECT a.* EXCEPT(ultimo_canal)
    FROM hive_metastore.fs.apex_digital_features a
""")

df = df_spark.toPandas()

# COMMAND ----------

df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split Temporal

# COMMAND ----------

# MAGIC %md
# MAGIC Validación temporal: últimos 2 meses como test

# COMMAND ----------

df_train, df_test = split_temporal(
    df, 
    DATA_CONFIG.col_fecha,
    MODEL_CONFIG.test_size_months
)

# COMMAND ----------

print(f"Train set: {len(df_train):,} clientes")
print(f"Test set: {len(df_test):,} clientes")

print(f"Train target mean: {df_train['target'].mean():.4f}")
print(f"Test target mean: {df_test['target'].mean():.4f}")

print(f"Train: {df_train['fecha_pedido_dt'].min()} - {df_train['fecha_pedido_dt'].max()}")
print(f"Test: {df_test['fecha_pedido_dt'].min()} - {df_test['fecha_pedido_dt'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Encoding de Categóricos

# COMMAND ----------

# MAGIC %md
# MAGIC Aplicar target encoding a variables de alta cardinalidad (país, agencia, etc)

# COMMAND ----------

# Target encoding de país
df_train, df_test = target_encoding(
    df_train, df_test, 
    DATA_CONFIG.col_pais, 
    'target',
    min_samples=50
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparación de Features

# COMMAND ----------

features_categoricos = [
    'ultimo_canal',
    'penultimo_canal', 
    'transicion_canal',
    DATA_CONFIG.col_madurez,
    DATA_CONFIG.col_frecuencia
]

# COMMAND ----------

X_train, y_train, feature_names = preparar_datos_modelo(
    df_train, 'target', features_categoricos
)

X_test, y_test, _ = preparar_datos_modelo(
    df_test, 'target', features_categoricos
)

# COMMAND ----------

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Número de features: {len(feature_names)}")

# COMMAND ----------

feature_names

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entrenamiento con MLflow Tracking

# COMMAND ----------

# MAGIC %md
# MAGIC Pipeline completo de entrenamiento con tracking automático

# COMMAND ----------

mlflow.set_experiment(MLFLOW_CONFIG.experiment_name)

# COMMAND ----------

model, resultados = pipeline_entrenamiento_completo(
    df_train,
    df_test,
    features_categoricos,
    usar_mlflow=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resultados y Métricas

# COMMAND ----------

# MAGIC %md
# MAGIC ### Métricas principales

# COMMAND ----------

metricas = resultados['metricas']

print("=" * 50)
print("MÉTRICAS DEL MODELO")
print("=" * 50)
print(f"ROC-AUC Score:        {metricas['roc_auc']:.4f}")
print(f"Average Precision:    {metricas['avg_precision']:.4f}")
print(f"Precision:            {metricas['precision']:.4f}")
print(f"Recall:               {metricas['recall']:.4f}")
print(f"F1-Score:             {metricas['f1_score']:.4f}")
print(f"Specificity:          {metricas['specificity']:.4f}")
print(f"Lift Score (Top 20%): {metricas['lift_score']:.4f}")
print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Matriz de confusión

# COMMAND ----------

print("\nMATRIZ DE CONFUSIÓN")
print("-" * 30)
print(f"True Positives:  {metricas['true_positives']:,}")
print(f"False Positives: {metricas['false_positives']:,}")
print(f"True Negatives:  {metricas['true_negatives']:,}")
print(f"False Negatives: {metricas['false_negatives']:,}")

# COMMAND ----------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(resultados['y_test'], 
                     (resultados['y_pred_proba'] >= MODEL_CONFIG.threshold).astype(int))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Digital', 'Digital'],
            yticklabels=['No Digital', 'Digital'])
ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
ax.set_ylabel('Real')
ax.set_xlabel('Predicho')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Curva ROC

# COMMAND ----------

fpr, tpr, thresholds_roc = roc_curve(resultados['y_test'], resultados['y_pred_proba'])

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {metricas["roc_auc"]:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Curva Precision-Recall

# COMMAND ----------

precision, recall, thresholds_pr = precision_recall_curve(
    resultados['y_test'], 
    resultados['y_pred_proba']
)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(recall, precision, linewidth=2, 
        label=f'PR (AP = {metricas["avg_precision"]:.4f})')
ax.axhline(y=resultados['y_test'].mean(), color='r', linestyle='--', 
           linewidth=2, label='Baseline')
ax.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución de Probabilidades Predichas

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(resultados['y_pred_proba'][resultados['y_test'] == 0], 
        bins=50, alpha=0.5, label='No Digital (real)', color='red', edgecolor='black')
ax.hist(resultados['y_pred_proba'][resultados['y_test'] == 1], 
        bins=50, alpha=0.5, label='Digital (real)', color='green', edgecolor='black')
ax.axvline(MODEL_CONFIG.threshold, color='blue', linestyle='--', 
           linewidth=2, label=f'Umbral = {MODEL_CONFIG.threshold}')
ax.set_title('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')
ax.set_xlabel('Probabilidad Predicha')
ax.set_ylabel('Frecuencia')
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

fi_df = resultados['feature_importance']

# COMMAND ----------

fi_df.head(20)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 10))
fi_df.head(20).set_index('feature')['importance'].plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 20 Features Más Importantes', fontsize=14, fontweight='bold')
ax.set_xlabel('Importancia (Gain)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análisis SHAP

# COMMAND ----------

# MAGIC %md
# MAGIC Valores SHAP para interpretabilidad del modelo

# COMMAND ----------

import shap

# COMMAND ----------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary Plot

# COMMAND ----------

shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance (SHAP)

# COMMAND ----------

shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                  plot_type='bar', show=False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segmentación de Clientes

# COMMAND ----------

# MAGIC %md
# MAGIC Segmentar clientes en cuartiles de propensión para acción comercial

# COMMAND ----------

df_test_pred = df_test.copy()
df_test_pred['propension_digital'] = resultados['y_pred_proba']

# COMMAND ----------

df_test_pred = segmentar_clientes_propension(df_test_pred, 'propension_digital')

# COMMAND ----------

df_test_pred['segmento_propension'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análisis por Segmento

# COMMAND ----------

segmento_analysis = df_test_pred.groupby('segmento_propension').agg({
    'target': ['count', 'sum', 'mean'],
    'propension_digital': ['min', 'max', 'mean']
}).round(4)

segmento_analysis.columns = ['_'.join(col) for col in segmento_analysis.columns]
segmento_analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución de Propensión por Segmento

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 6))

for segmento in ['Baja', 'Media', 'Alta', 'Muy Alta']:
    data = df_test_pred[df_test_pred['segmento_propension'] == segmento]['propension_digital']
    ax.hist(data, bins=30, alpha=0.5, label=segmento, edgecolor='black')

ax.set_title('Distribución de Propensión por Segmento', fontsize=14, fontweight='bold')
ax.set_xlabel('Probabilidad Predicha')
ax.set_ylabel('Cantidad de Clientes')
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tasa de Conversión Real por Segmento

# COMMAND ----------

conversion_por_segmento = df_test_pred.groupby('segmento_propension')['target'].mean() * 100

fig, ax = plt.subplots(figsize=(10, 6))
conversion_por_segmento.plot(kind='bar', ax=ax, color=['#EE964B', '#F4D35E', '#06A77D', '#0E6BA8'])
ax.set_title('Tasa de Conversión Digital Real por Segmento', fontsize=14, fontweight='bold')
ax.set_xlabel('Segmento de Propensión')
ax.set_ylabel('% Clientes con Pedido Digital')
ax.tick_params(axis='x', rotation=0)

for i, v in enumerate(conversion_por_segmento):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lift por Segmento

# COMMAND ----------

baseline_rate = df_test_pred['target'].mean()

lift_por_segmento = (conversion_por_segmento / 100) / baseline_rate

fig, ax = plt.subplots(figsize=(10, 6))
lift_por_segmento.plot(kind='bar', ax=ax, color=['#EE964B', '#F4D35E', '#06A77D', '#0E6BA8'])
ax.axhline(1, color='red', linestyle='--', linewidth=2, label='Baseline (Lift = 1)')
ax.set_title('Lift por Segmento de Propensión', fontsize=14, fontweight='bold')
ax.set_xlabel('Segmento')
ax.set_ylabel('Lift vs Baseline')
ax.tick_params(axis='x', rotation=0)
ax.legend()

for i, v in enumerate(lift_por_segmento):
    ax.text(i, v + 0.05, f'{v:.2f}x', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generación de Scoring para Toda la Base

# COMMAND ----------

# MAGIC %md
# MAGIC Aplicar modelo a todos los clientes para generar lista priorizada

# COMMAND ----------

df_all_features = spark.sql("""
    SELECT a.* EXCEPT(ultimo_canal)
    FROM hive_metastore.fs.apex_digital_features a
""").toPandas()

# COMMAND ----------

# COMMAND ----------

# Crear target dummy
df_all_features['target'] = 0

# COMMAND ----------

# PASO 1: Aplicar target encoding de país (MISMO QUE EN TRAIN)
# Calcular mapping desde df_train
pais_target_encoding_map = df_train.groupby(DATA_CONFIG.col_pais)['target'].mean().to_dict()
global_mean_pais = df_train['target'].mean()

# Aplicar a todos los datos
df_all_features['pais_cd_target_encoded'] = df_all_features[DATA_CONFIG.col_pais].map(
    pais_target_encoding_map
).fillna(global_mean_pais)

print(f"Países únicos: {df_all_features[DATA_CONFIG.col_pais].nunique()}")
print(f"Target encoding aplicado correctamente")

# COMMAND ----------

# PASO 2: Preparar datos (one-hot encoding de categóricas)
X_all, _, feature_names_all = preparar_datos_modelo(
    df_all_features, 
    'target', 
    features_categoricos
)

# COMMAND ----------

# Verificar columnas
print(f"Features esperados: {len(resultados['feature_names'])}")
print(f"Features generados: {len(feature_names_all)}")

missing = set(resultados['feature_names']) - set(feature_names_all)
if missing:
    print(f"⚠️ Columnas faltantes: {missing}")
else:
    print("✅ Todas las columnas presentes")

# COMMAND ----------

# PASO 3: Ordenar columnas
X_all = X_all[resultados['feature_names']]

# COMMAND ----------

# PASO 4: Generar predicciones
predicciones_todas = model.predict(X_all, num_iteration=model.best_iteration)

print(f"✅ Predicciones generadas para {len(predicciones_todas):,} clientes")
print(f"Rango de probabilidades: [{predicciones_todas.min():.4f}, {predicciones_todas.max():.4f}]")

# COMMAND ----------

# Continuar con df_scoring como antes...
df_scoring = pd.DataFrame({
    DATA_CONFIG.col_cliente_id: df_all_features[DATA_CONFIG.col_cliente_id],
    'propension_digital': predicciones_todas,
    'madurez_digital': df_all_features[DATA_CONFIG.col_madurez],
    'pct_digital_historico': df_all_features['pct_digital_historico']
})

# COMMAND ----------

df_scoring = segmentar_clientes_propension(df_scoring, 'propension_digital', n_segmentos=10)

# COMMAND ----------

df_scoring.head(20)

# COMMAND ----------

df_scoring['segmento_propension'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Guardar Scoring para Uso Comercial

# COMMAND ----------

output_scoring_table = "hive_metastore.inference.m_scoringDigital_ifr"

df_scoring_spark = spark.createDataFrame(df_scoring)
df_scoring_spark.write.mode('overwrite').saveAsTable(output_scoring_table)

# COMMAND ----------

# MAGIC %md
# MAGIC # Resultados

# COMMAND ----------

# MAGIC %md
# MAGIC ## Muestra

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM hive_metastore.inference.m_scoringDigital_ifr
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM hive_metastore.inference.m_scoringDigital_ifr

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   segmento_propension,
# MAGIC   madurez_digital,
# MAGIC   COUNT(cliente_id) as q_clientes,
# MAGIC   AVG(propension_digital) as prop_avg,
# MAGIC   MIN(propension_digital) as prop_min,
# MAGIC   MAX(propension_digital) as prop_max
# MAGIC FROM hive_metastore.inference.m_scoringDigital_ifr
# MAGIC GROUP BY ALL
# MAGIC ORDER BY ALL
