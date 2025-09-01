# 📊 ESG Analytics IBEX35

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Development-orange.svg)](https://github.com/arold0/esg-analytics-ibex35)

> **Análisis integral de factores ESG (Environmental, Social, Governance) para las empresas del IBEX35**

## 📊 Resultados Actuales

### ✅ **Datos Recolectados:**
- **31 de 35 empresas** del IBEX35 (88.6% de éxito)
- **Datos históricos 2019-2024** completos
- **Métricas financieras** calculadas automáticamente
- **Scores ESG compuestos** generados
- **Análisis sectorial** completado (6 sectores)

### 🔍 **Hallazgos Estadísticos:**
- **9 correlaciones significativas** entre ESG y métricas financieras
- **6 sectores analizados** con diferencias estadísticas significativas
- **4 modelos de ML** implementados y evaluados
- **Tests de normalidad y ANOVA** completados
- **Análisis de regresión** con cross-validation

### 📈 **Correlaciones Más Fuertes:**
1. **Social Score ↔ Sharpe Ratio**: -0.363 (Moderada)
2. **Governance Score ↔ Volatilidad**: 0.429 (Moderada)
3. **Governance Score ↔ ROE**: -0.350 (Moderada)
4. **E_Score ↔ Returns 1Y**: -0.375 (Moderada)
5. **ESG_Total ↔ Volatilidad**: 0.385 (Moderada)

## 🎯 Descripción del Proyecto

Este proyecto realiza un análisis completo de los factores ESG (Environmental, Social, Governance) de las empresas que componen el índice IBEX35 de la Bolsa de Madrid. El objetivo es proporcionar insights valiosos sobre el rendimiento sostenible de las principales empresas españolas y su correlación con el rendimiento financiero.

### 🌟 Características Principales

- 📈 **Análisis de Datos Financieros**: Recopilación y análisis de datos bursátiles históricos
- 🌱 **Métricas ESG**: Evaluación de factores ambientales, sociales y de gobernanza
- 📊 **Visualizaciones Interactivas**: Dashboards y gráficos dinámicos
- 🤖 **Machine Learning**: Modelos predictivos y análisis de correlaciones
- 📋 **Reportes Ejecutivos**: Generación automática de informes
- 🌐 **Aplicación Web**: Dashboard interactivo con Streamlit

## 🏗️ Estructura del Proyecto

```
esg-analytics-ibex35/
├── 📁 data/                    # Datos del proyecto
│   ├── raw/                   # Datos sin procesar
│   ├── processed/             # Datos procesados
│   └── cleaned/               # Datos limpios
├── 📁 notebooks/              # Jupyter notebooks
│   ├── 01-data-collection.ipynb
│   ├── 02-data-cleaning.ipynb
│   ├── 03-exploratory-analysis.ipynb
│   ├── 04-correlation-analysis.ipynb
│   └── 05-visualization.ipynb
├── 📁 src/                    # Código fuente
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── analysis.py
│   └── visualization.py
├── 📁 scripts/                # Scripts de automatización
├── 📁 reports/                # Reportes generados
├── 📁 config/                 # Configuraciones
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
- [ ] API REST

### Fase 3: Escalabilidad
- [ ] Base de datos
- [ ] Actualizaciones automáticas
- [ ] Múltiples índices bursátiles
- [ ] Aplicación web completa

---

⭐ **Si este proyecto te resulta útil, ¡dale una estrella en GitHub!**
