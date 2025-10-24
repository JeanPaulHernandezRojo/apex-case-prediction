# Apex Digital - Predicción de Pedidos Digitales en eB2B

## 🎯 Objetivo
Desarrollar un modelo predictivo para identificar clientes con alta probabilidad de realizar su próximo pedido a través del canal digital en la plataforma eB2B.

## 📊 Dataset
- **Registros:** 1.25M transacciones
- **Clientes únicos:** ~150K
- **Período:** Enero 2023 - Agosto 2024
- **Variables clave:** canal_pedido_cd, madurez_digital_cd, país, facturación, frecuencia

## 🚀 Enfoque de Solución

### Feature Engineering
- Comportamiento histórico por cliente (% pedidos digitales)
- Tendencias temporales (últimos 3 meses)
- Transiciones entre canales
- Características de negocio (ticket promedio, materiales)
- Encoding de variables categóricas jerárquicas

### Modelado
- **Algoritmo principal:** LightGBM
- **Validación:** Split temporal (últimos 2 meses)
- **Métricas:** ROC-AUC, Precision-Recall, Feature Importance
- **Tracking:** MLflow

### Segmentación
Clasificación de clientes en deciles de propensión para priorización comercial.

## 📂 Estructura del Proyecto
```
├── notebooks/
│   ├── 01_eda.py                    # Análisis exploratorio
│   ├── 02_feature_engineering.py    # Construcción de features
│   └── 03_modeling.py               # Entrenamiento y evaluación
├── src/
│   ├── features.py                  # Funciones de feature engineering
│   ├── models.py                    # Pipeline de modelado
│   └── config.py                    # Configuraciones
├── results/
│   ├── figures/                     # Gráficos y visualizaciones
│   └── metrics/                     # Métricas de evaluación
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Tecnologías
- **Compute:** Databricks
- **ML Framework:** scikit-learn, LightGBM
- **Tracking:** MLflow
- **Visualización:** matplotlib, seaborn, plotly

## 📈 Resultados Principales

### Hallazgos Clave
1. **Madurez Digital** es el predictor más fuerte:
   - Madurez ALTA: 75% pedidos digitales
   - Madurez MEDIA: 55% pedidos digitales
   - Madurez BAJA: 35% pedidos digitales

### Métricas del Modelo
- **ROC-AUC:** [IN PROGRESS]
- **Precision@K (Top 20%):** [IN PROGRESS]
- **Lift Score:** [IN PROGRESS]

## 💼 Recomendaciones de Negocio

1. **Segmento Alta Propensión (Top 10%):** Comunicación directa sobre beneficios digitales
2. **Segmento Media Propensión:** Incentivos y capacitación en plataforma
3. **Segmento Baja Propensión:** Mantener canales tradicionales pero con nudges digitales

## 🔄 Mejoras Futuras
- Análisis de causalidad: impacto de intervenciones
- Reentrenamiento automático mensual

## 🚀 Cómo Ejecutar

### Prerrequisitos
```bash
pip install -r requirements.txt
```

### En Databricks
1. Clonar repositorio en Databricks Repos
2. Cargar datos en DBFS: `dbfs:/mnt/data/apex_digital.parquet`
3. Ejecutar notebooks en orden: 01 → 02 → 03
4. Revisar experimentos en MLflow UI
