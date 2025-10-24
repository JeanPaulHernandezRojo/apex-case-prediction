# Databricks notebook source
# MAGIC %md
# MAGIC # Calculation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import sys

# COMMAND ----------

sys.path.append('/Workspace/Users/jhernandezr@uni.pe/Apex/apex-case-prediction')

# COMMAND ----------

from src.features import (
    calcular_features_historicos,
    calcular_features_temporales,
    calcular_features_secuenciales,
    calcular_features_categoricos,
    construir_dataset_completo
)
from src.config import DATA_CONFIG, FEATURE_CONFIG

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carga de Datos

# COMMAND ----------

df_spark = spark.sql(f"""
  SELECT
    t.*,
    NVL(DATEDIFF(DAY, LAG(fecha_pedido_dt) OVER (PARTITION BY cliente_id ORDER BY fecha_pedido_dt), fecha_pedido_dt), 600) as dias_desde_ultimo_pedido
  FROM {DATA_CONFIG.input_data_table} t
""")

df = df_spark.toPandas()

# COMMAND ----------

df[DATA_CONFIG.col_fecha] = pd.to_datetime(df[DATA_CONFIG.col_fecha])

df = df.sort_values([DATA_CONFIG.col_cliente_id, DATA_CONFIG.col_fecha])

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Features Históricos

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cálculo

# COMMAND ----------

features_hist = calcular_features_historicos(df)

# COMMAND ----------

features_hist.head(10)

# COMMAND ----------

features_hist.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución del % digital histórico

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(features_hist['pct_digital_historico'], bins=50, edgecolor='black', alpha=0.7)
ax.set_title('Distribución de % Digital Histórico por Cliente', fontsize=14, fontweight='bold')
ax.set_xlabel('% Pedidos Digitales')
ax.set_ylabel('Cantidad de Clientes')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Features Temporales

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cálculo

# COMMAND ----------

features_temp = calcular_features_temporales(df)

# COMMAND ----------

features_temp.head(10)

# COMMAND ----------

features_temp.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparación: Digital 90d vs 30d

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(features_temp['pct_digital_90d'], features_temp['pct_digital_30d'], 
           alpha=0.3, s=10)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Sin cambio')
ax.set_title('% Digital: 90 días vs 30 días', fontsize=14, fontweight='bold')
ax.set_xlabel('% Digital últimos 90 días')
ax.set_ylabel('% Digital últimos 30 días')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución de Tendencia Digital

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(features_temp['tendencia_digital'], bins=50, edgecolor='black', alpha=0.7, color='orange')
ax.set_title('Distribución de Tendencia Digital', fontsize=14, fontweight='bold')
ax.set_xlabel('Tendencia (30d - 90d)')
ax.set_ylabel('Cantidad de Clientes')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Sin cambio')
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

clientes_acelerando = (features_temp['tendencia_digital'] > 0.1).sum()
clientes_desacelerando = (features_temp['tendencia_digital'] < -0.1).sum()

print(f"Clientes acelerando uso digital: {clientes_acelerando:,}")
print(f"Clientes desacelerando uso digital: {clientes_desacelerando:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Features Secuenciales

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cálculo

# COMMAND ----------

features_seq = calcular_features_secuenciales(df)

# COMMAND ----------

features_seq.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución de último canal

# COMMAND ----------

features_seq['ultimo_canal'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transiciones más comunes

# COMMAND ----------

features_seq['transicion_canal'].value_counts().head(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución de racha actual

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(features_seq['racha_canal_actual'], bins=30, edgecolor='black', alpha=0.7, color='green')
ax.set_title('Distribución de Racha en Canal Actual', fontsize=14, fontweight='bold')
ax.set_xlabel('Número de pedidos consecutivos en mismo canal')
ax.set_ylabel('Cantidad de Clientes')
plt.tight_layout()
plt.show()

# COMMAND ----------

features_seq['racha_canal_actual'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Features Categóricos

# COMMAND ----------

features_cat = calcular_features_categoricos(df)

# COMMAND ----------

features_cat.head(10)

# COMMAND ----------

features_cat['madurez_digital_cd'].value_counts()

# COMMAND ----------

features_cat['madurez_digital_encoded'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Construcción del Dataset Completo

# COMMAND ----------

# MAGIC %md
# MAGIC Consolidar todos los features en un único dataset.

# COMMAND ----------

df_final = construir_dataset_completo(df)

# COMMAND ----------

df_final.shape

# COMMAND ----------

df_final.head()

# COMMAND ----------

df_final.columns.tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución del target en el dataset final

# COMMAND ----------

target_dist = df_final['target'].value_counts()
target_pct = (target_dist / len(df_final) * 100).round(2)

pd.DataFrame({
    'Clientes': target_dist,
    'Porcentaje': target_pct
})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificación de nulos

# COMMAND ----------

df_final.isnull().sum().sort_values(ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlación con target

# COMMAND ----------

correlaciones = df_final.select_dtypes(include=[np.number]).corr()['target'].sort_values(ascending=False)

print("Top 15 features más correlacionados con target:")
correlaciones.head(15)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 8))
correlaciones.head(20).plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 20 Features - Correlación con Target', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlación')
plt.tight_layout()
plt.show()
plt.savefig('/Workspace/Users/jhernandezr@uni.pe/Apex/apex-case-prediction/results/04_CorrelacionFeaturesTarget.png', dpi=300, bbox_inches='tight')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Guardar Dataset Procesado

# COMMAND ----------

# MAGIC %md
# MAGIC Guardar en formato parquet para el notebook de modelado

# COMMAND ----------

# Agregar fecha para split temporal (necesaria para validación)
df_ultima_fecha = df.groupby(DATA_CONFIG.col_cliente_id)[DATA_CONFIG.col_fecha].max().reset_index()
df_final = df_final.merge(df_ultima_fecha, on=DATA_CONFIG.col_cliente_id, how='left')

# COMMAND ----------

output_table = "hive_metastore.fs.apex_digital_features"

df_final_spark = spark.createDataFrame(df_final)
df_final_spark.write.mode('overwrite').saveAsTable(output_table)

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploring

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM hive_metastore.fs.apex_digital_features
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(1)
# MAGIC FROM hive_metastore.fs.apex_digital_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profile

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM hive_metastore.fs.apex_digital_features

# COMMAND ----------

# MAGIC %md
# MAGIC # Resumen

# COMMAND ----------

# MAGIC %md
# MAGIC - 29 features + 1 target
# MAGIC - Para cada cliente, excluyo su último pedido del cálculo de features
# MAGIC - El target es si ese último pedido excluido es digital
# MAGIC - Solo incluyo clientes con 2+ pedidos
