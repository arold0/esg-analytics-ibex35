# 📊 ESG Analytics IBEX35

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Development-orange.svg)](https://github.com/arold0/esg-analytics-ibex35)

> **Análisis integral de factores ESG (Environmental, Social, Governance) para las empresas del IBEX35**

## 📊 Estado del Proyecto

### ✅ **Fase 1, 2 & 3 COMPLETADAS:**
- **31 de 35 empresas** del IBEX35 analizadas (88.6% de éxito)
- **Datos históricos 2019-2024** recolectados y procesados
- **Pipeline completo** de análisis implementado
- **API REST v2 funcional** con base de datos SQLite y 8 endpoints optimizados
- **Reportes ejecutivos** generados automáticamente
- **Visualizaciones interactivas** en HTML

### 🔍 **Resultados Clave:**
- **9 correlaciones significativas** ESG-Financieras identificadas
- **6 sectores** analizados con ANOVA
- **4 modelos ML** evaluados (Linear, Ridge, Lasso, Random Forest)
- **API REST** con documentación Swagger integrada
- **Dashboard interactivo** disponible

### 📈 **Correlaciones Principales:**
1. **Social Score ↔ Sharpe Ratio**: -0.363 (Moderada)
2. **Governance Score ↔ Volatilidad**: 0.429 (Moderada)
3. **Governance Score ↔ ROE**: -0.350 (Moderada)
4. **E_Score ↔ Returns 1Y**: -0.375 (Moderada)
5. **ESG_Total ↔ Volatilidad**: 0.385 (Moderada)

## 🎯 Descripción del Proyecto

Este proyecto realiza un análisis completo de los factores ESG (Environmental, Social, Governance) de las empresas que componen el índice IBEX35 de la Bolsa de Madrid. El objetivo es proporcionar insights valiosos sobre el rendimiento sostenible de las principales empresas españolas y su correlación con el rendimiento financiero.

### 🌟 Características Implementadas

- 📈 **Pipeline Completo**: Recolección, procesamiento y análisis automatizado
- 🌱 **Análisis ESG**: Evaluación integral de 31 empresas IBEX35
- 📊 **Visualizaciones**: Dashboards HTML interactivos y mapas de calor
- 🤖 **Machine Learning**: 4 modelos evaluados con cross-validation
- 📋 **Reportes Automáticos**: Informes ejecutivos y técnicos
- 🌐 **API REST**: 8 endpoints con documentación Swagger
- 📊 **Análisis Estadístico**: Correlaciones, ANOVA y tests de significancia

## 🏗️ Estructura del Proyecto

```
esg-analytics-ibex35/
├── 📁 data/                    # Datos del proyecto
│   ├── raw/                   # Datos originales (CSV)
│   ├── processed/             # Análisis y resultados (YAML)
│   └── cleaned/               # Datos procesados
├── esg_analytics.db           # Base de datos SQLite
├── 📁 notebooks/              # Jupyter notebooks
│   ├── 01-data-collection.ipynb
│   └── 02-data-cleaning.ipynb
├── 📁 src/                    # Código fuente principal
│   ├── data_collection.py     # Recolección de datos
│   ├── data_processing.py     # Procesamiento y limpieza
│   ├── analysis.py            # Análisis estadístico y ML
│   ├── visualization.py       # Dashboards y gráficos
│   ├── database.py            # Modelos SQLAlchemy y BD
│   ├── api.py                 # API REST v1 (legacy)
│   └── api_v2.py              # API REST v2 con base de datos
├── 📁 scripts/                # Scripts de automatización
│   ├── download_data.py       # Descarga automática
│   ├── process_data.py        # Procesamiento batch
│   ├── run_analysis.py        # Análisis completo
│   └── generate_report.py     # Generación de reportes
├── 📁 reports/                # Reportes HTML generados
│   ├── figures/               # Visualizaciones interactivas
│   ├── detailed_analysis_report.html
│   └── executive_summary.html
├── 📁 config/                 # Configuraciones YAML
├── 📁 logs/                   # Logs de ejecución
└── 📁 tests/                  # Tests unitarios
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.11 o superior
- Conda (recomendado) o pip
- Git

### Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/arold0/esg-analytics-ibex35.git
cd esg-analytics-ibex35

# 2. Crear y activar el entorno conda
conda env create -f environment.yml
conda activate esg-analytics

# 3. Instalar dependencias adicionales
pip install -r requirements.txt
```

### Instalación Paso a Paso

```bash
# Opción 1: Todo de una vez
conda env create -f environment.yml && conda activate esg-analytics && pip install -r requirements.txt

# Opción 2: Por categorías
pip install yfinance requests beautifulsoup4 selenium  # Data collection
pip install plotly dash streamlit                      # Visualization & Web apps
pip install statsmodels scipy openpyxl                # Analysis & File handling
```

### Verificación

```bash
python -c "import pandas, yfinance, plotly, streamlit; print('✅ All packages installed successfully!')"
```

## 📖 Uso

### 1. Recolección de Datos

```bash
# Ejecutar script de recolección completa
python scripts/download_data.py

# O usar el notebook
jupyter notebook notebooks/01-data-collection.ipynb
```

### 2. Procesamiento de Datos

```bash
# Procesar y limpiar datos descargados
python scripts/process_data.py
```

### 3. Análisis Estadístico

```bash
# Ejecutar análisis estadístico completo
python scripts/run_analysis.py
```

### 4. Análisis Exploratorio

```bash
# Abrir notebook de análisis
jupyter notebook notebooks/03-exploratory-analysis.ipynb
```

### 3. Dashboard Interactivo

```bash
# Lanzar aplicación Streamlit
streamlit run src/visualization.py
```

### 4. Generación de Reportes

```bash
# Generar reportes ejecutivos
python scripts/generate_report.py

# Ver visualizaciones generadas
open reports/figures/comprehensive_dashboard.html
open reports/figures/correlation_heatmap.html
```

### 5. API REST

```bash
# Instalar dependencias de API
pip install fastapi uvicorn pydantic

# Lanzar API REST v2 (con base de datos)
uvicorn src.api_v2:app --reload --host 0.0.0.0 --port 8001

# Acceder a documentación interactiva
open http://localhost:8001

# API v1 (legacy, solo CSV)
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

## 🛠️ Tecnologías Utilizadas

### 📊 Data Science & Analytics
- **Pandas** - Manipulación y análisis de datos
- **NumPy** - Computación numérica
- **Scikit-learn** - Machine Learning
- **Statsmodels** - Análisis estadístico

### 📈 Visualización
- **Matplotlib** - Gráficos básicos
- **Seaborn** - Visualización estadística
- **Plotly** - Gráficos interactivos
- **Dash** - Aplicaciones web interactivas

### 🌐 Web Development
- **Streamlit** - Aplicaciones web rápidas
- **FastAPI** - API REST moderna y rápida
- **BeautifulSoup4** - Web scraping
- **Selenium** - Automatización web

### 📊 Financial Data
- **yfinance** - Datos de Yahoo Finance
- **OpenPyXL** - Manejo de archivos Excel

## 📊 Funcionalidades

### 🔍 Análisis ESG
- **Environmental**: Emisiones de CO2, eficiencia energética, gestión de residuos
- **Social**: Diversidad, derechos laborales, impacto comunitario
- **Governance**: Transparencia, independencia del consejo, remuneración ejecutiva

### 📈 Análisis Financiero ✅ IMPLEMENTADO
- Rendimiento bursátil histórico (2019-2024)
- Ratios financieros clave (ROE, ROA, Sharpe, Volatilidad)
- Análisis de volatilidad y drawdown
- Correlación ESG vs. Rendimiento (9 correlaciones significativas)

### 🧮 Análisis Estadístico ✅ IMPLEMENTADO
- Correlación entre métricas ESG y financieras
- Modelos de regresión (Linear, Ridge, Lasso, Random Forest)
- Tests estadísticos de significancia
- Análisis por sectores con ANOVA
- Machine Learning con cross-validation

### 📊 Visualizaciones ✅ IMPLEMENTADO
- Gráficos de evolución temporal
- Mapas de calor de correlaciones (HTML interactivos)
- Dashboards comprehensivos generados
- Distribuciones ESG por sector
- Análisis de performance financiera

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor, sigue estos pasos:

1. **Fork** el proyecto
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

### Guías de Contribución

- Sigue las convenciones de código Python (PEP 8)
- Añade tests para nuevas funcionalidades
- Actualiza la documentación según sea necesario
- Usa mensajes de commit descriptivos

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autores

- **Arold0** - *Desarrollo inicial* - [arold0](https://github.com/arold0)

## 🙏 Agradecimientos

- Datos financieros proporcionados por Yahoo Finance
- Información ESG de múltiples fuentes públicas
- Comunidad de Python y data science

## 📞 Contacto

- **GitHub**: [@arold0](https://github.com/arold0)
- **Proyecto**: [https://github.com/arold0/esg-analytics-ibex35](https://github.com/arold0/esg-analytics-ibex35)

## 📈 Roadmap

### Fase 1: MVP ✅ COMPLETADO
- [x] Estructura del proyecto
- [x] Configuración del entorno
- [x] Recolección de datos básicos (31/35 empresas, 88.6% éxito)
- [x] Análisis exploratorio inicial
- [x] Procesamiento y limpieza de datos
- [x] Análisis estadístico completo

### Fase 2: Análisis Avanzado ✅ COMPLETADO
- [x] Modelos de machine learning (Linear, Ridge, Lasso, Random Forest)
- [x] Análisis de correlaciones ESG-Financieras (9 significativas)
- [x] Tests estadísticos de significancia (Shapiro-Wilk, ANOVA)
- [x] Análisis por sectores con ANOVA (6 sectores)
- [x] Visualizaciones interactivas (HTML dashboards)
- [x] Análisis de regresión con cross-validation
- [x] Reportes ejecutivos automáticos
- [x] API REST

### Fase 3: Escalabilidad ✅ COMPLETADO
- [x] **Base de datos relacional** (SQLite)
  - [x] Esquema de datos optimizado con SQLAlchemy
  - [x] Migración de datos CSV a BD (31 empresas)
  - [x] Índices y optimización de consultas
  - [x] API v2 integrada con base de datos
- [ ] **Actualizaciones automáticas**
  - [ ] Scheduler para recolección diaria
  - [ ] Pipeline de actualización incremental
  - [ ] Notificaciones de cambios significativos
- [ ] **Múltiples índices bursátiles**
  - [x] Soporte básico para FTSE 100, DAX 30
  - [ ] Análisis comparativo entre mercados
  - [ ] Benchmarking internacional
- [ ] **Aplicación web completa**
  - [ ] Frontend React/Vue.js
  - [ ] Dashboard en tiempo real
  - [ ] Sistema de alertas y notificaciones
  - [ ] Exportación de reportes personalizados

### Fase 4: Producción 📋 PLANIFICADO
- [ ] **Containerización** (Docker)
- [ ] **CI/CD Pipeline** (GitHub Actions)
- [ ] **Monitoreo y logging** avanzado
- [ ] **Autenticación y autorización**
- [ ] **Cache y optimización** de rendimiento
- [ ] **Documentación técnica** completa

---

⭐ **Si este proyecto te resulta útil, ¡dale una estrella en GitHub!**
