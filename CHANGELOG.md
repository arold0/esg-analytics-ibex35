# Changelog

Todos los cambios notables del proyecto ESG Analytics IBEX35 se documentan en este archivo.

## [2.0.0] - 2025-09-01

### ✅ FASE 2 COMPLETADA - Análisis Avanzado

#### Añadido
- **API REST completa** con FastAPI
  - 8 endpoints funcionales (`/companies`, `/correlations`, `/sectors`, etc.)
  - Documentación Swagger automática en `/`
  - Validación de datos con Pydantic
  - Soporte CORS para integración web
- **Modelos de Machine Learning**
  - Linear Regression, Ridge, Lasso, Random Forest
  - Cross-validation y métricas de evaluación
  - Análisis de importancia de features
- **Análisis estadístico avanzado**
  - 9 correlaciones ESG-Financieras significativas
  - Tests de normalidad (Shapiro-Wilk)
  - ANOVA por sectores (6 sectores analizados)
- **Reportes ejecutivos automáticos**
  - HTML interactivos con visualizaciones
  - Resumen ejecutivo y análisis detallado
  - Exportación automática a `/reports/`
- **Visualizaciones interactivas**
  - Dashboards HTML con Plotly
  - Mapas de calor de correlaciones
  - Gráficos de distribución por sector

#### Mejorado
- **Pipeline de procesamiento** más robusto
- **Manejo de errores** mejorado en carga de datos
- **Documentación** actualizada con estado actual
- **Estructura de archivos** optimizada

#### Técnico
- Agregadas dependencias: `fastapi>=0.116.0`, `uvicorn>=0.35.0`, `pydantic>=2.11.0`
- Actualizado `requirements.txt` con versiones específicas
- Sincronizado `environment.yml` con dependencias actuales
- Logs estructurados para debugging

## [1.0.0] - 2025-08-30

### ✅ FASE 1 COMPLETADA - MVP

#### Añadido
- **Recolección de datos** automatizada
  - 31 de 35 empresas IBEX35 (88.6% éxito)
  - Datos históricos 2019-2024
  - Métricas financieras calculadas
- **Procesamiento de datos**
  - Limpieza y normalización
  - Cálculo de scores ESG compuestos
  - Análisis sectorial básico
- **Análisis exploratorio**
  - Correlaciones básicas
  - Estadísticas descriptivas
  - Visualizaciones preliminares

#### Infraestructura
- Configuración inicial del proyecto
- Estructura de directorios
- Entorno conda configurado
- Scripts de automatización básicos

---

## 🚀 Próximas Versiones

### [3.0.0] - Fase 3: Escalabilidad (En desarrollo)
- Base de datos relacional (PostgreSQL/SQLite)
- Actualizaciones automáticas con scheduler
- Soporte para múltiples índices bursátiles
- Aplicación web completa con frontend moderno

### [4.0.0] - Fase 4: Producción (Planificado)
- Containerización con Docker
- CI/CD Pipeline con GitHub Actions
- Monitoreo y logging avanzado
- Autenticación y autorización
- Optimización de rendimiento

---

**Formato basado en [Keep a Changelog](https://keepachangelog.com/)**
