"""
ESG Analytics IBEX35 - Statistical Analysis Module
==================================================

This module performs comprehensive statistical analysis on ESG and financial data.
It includes correlation analysis, regression models, temporal analysis, and machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ESGAnalyzer:
    """
    Clase principal para análisis estadístico de datos ESG y financieros
    """
    
    def __init__(self, config_path: str = None):
        """
        Inicializa el analizador ESG
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self.load_config(config_path)
        self.data = None
        self.results = {}
        logger.info("ESGAnalyzer initialized")
    
    def load_config(self, config_path: str = None) -> Dict:
        """Carga la configuración del proyecto"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def load_processed_data(self, data_path: str = "data/processed/ibex35_processed_data.csv") -> pd.DataFrame:
        """
        Carga los datos procesados
        
        Args:
            data_path: Ruta al archivo de datos procesados
            
        Returns:
            pd.DataFrame: Datos cargados
        """
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            raise
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """
        Análisis de correlación entre variables ESG y financieras
        
        Returns:
            Dict: Resultados del análisis de correlación
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_processed_data() first.")
        
        logger.info("Starting correlation analysis")
        
        # Identificar columnas ESG y financieras
        esg_columns = [col for col in self.data.columns if any(keyword in col.lower() 
                    for keyword in ['esg', 'environmental', 'social', 'governance', 'e_score', 's_score', 'g_score'])]
        
        financial_columns = [col for col in self.data.columns if any(keyword in col.lower() 
                           for keyword in ['roe', 'roa', 'volatility', 'sharpe', 'returns', 'price', 'beta'])]
        
        # Calcular correlaciones
        correlation_matrix = self.data[esg_columns + financial_columns].corr()
        
        # Correlaciones específicas ESG vs Financieras
        esg_financial_corr = correlation_matrix.loc[esg_columns, financial_columns]
        
        # Análisis de correlaciones significativas
        significant_correlations = []
        for esg_col in esg_columns:
            for fin_col in financial_columns:
                corr_value = correlation_matrix.loc[esg_col, fin_col]
                if abs(corr_value) > 0.3:  # Correlación moderada o fuerte
                    significant_correlations.append({
                        'esg_variable': esg_col,
                        'financial_variable': fin_col,
                        'correlation': corr_value,
                        'strength': 'Strong' if abs(corr_value) > 0.5 else 'Moderate'
                    })
        
        results = {
            'correlation_matrix': correlation_matrix,
            'esg_financial_correlations': esg_financial_corr,
            'significant_correlations': significant_correlations,
            'esg_columns': esg_columns,
            'financial_columns': financial_columns
        }
        
        self.results['correlation_analysis'] = results
        logger.info(f"Correlation analysis completed. Found {len(significant_correlations)} significant correlations")
        
        return results
    
    def regression_analysis(self, target_variable: str = 'returns_1y') -> Dict[str, Any]:
        """
        Análisis de regresión: ESG scores → Variable objetivo financiera
        
        Args:
            target_variable: Variable objetivo para la regresión
            
        Returns:
            Dict: Resultados del análisis de regresión
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_processed_data() first.")
        
        logger.info(f"Starting regression analysis with target: {target_variable}")
        
        # Preparar datos
        esg_features = ['E_Score', 'S_Score', 'G_Score', 'ESG_Total']
        available_features = [col for col in esg_features if col in self.data.columns]
        
        if target_variable not in self.data.columns:
            logger.error(f"Target variable {target_variable} not found in data")
            return {}
        
        # Filtrar datos válidos
        valid_data = self.data[available_features + [target_variable]].dropna()
        
        if len(valid_data) < 10:
            logger.warning("Insufficient data for regression analysis")
            return {}
        
        X = valid_data[available_features]
        y = valid_data[target_variable]
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Modelos de regresión
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            try:
                # Entrenar modelo
                model.fit(X_train, y_train)
                
                # Predicciones
                y_pred = model.predict(X_test)
                
                # Métricas
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                
                # Coeficientes (para modelos lineales)
                coefficients = None
                if hasattr(model, 'coef_'):
                    coefficients = dict(zip(available_features, model.coef_))
                
                results[model_name] = {
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'coefficients': coefficients,
                    'feature_importance': self._get_feature_importance(model, available_features)
                }
                
                logger.info(f"{model_name}: R² = {r2:.3f}, CV R² = {cv_scores.mean():.3f}")
                
            except Exception as e:
                logger.error(f"Error in {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.results['regression_analysis'] = results
        return results
    
    def sector_analysis(self) -> Dict[str, Any]:
        """
        Análisis comparativo por sectores
        
        Returns:
            Dict: Resultados del análisis por sectores
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_processed_data() first.")
        
        logger.info("Starting sector analysis")
        
        # Cargar información de sectores desde configuración
        companies_data = self.config['ibex35']['companies']
        sector_mapping = {company['symbol']: company['sector'] for company in companies_data}
        
        # Agregar sector al DataFrame
        self.data['sector'] = self.data['ticker'].map(sector_mapping)
        
        # Análisis por sector
        sector_stats = {}
        esg_metrics = ['E_Score', 'S_Score', 'G_Score', 'ESG_Total']
        financial_metrics = ['roe', 'roa', 'volatility_30d', 'sharpe_ratio']
        
        available_esg = [col for col in esg_metrics if col in self.data.columns]
        available_financial = [col for col in financial_metrics if col in self.data.columns]
        
        for sector in self.data['sector'].unique():
            if pd.isna(sector):
                continue
                
            sector_data = self.data[self.data['sector'] == sector]
            
            if len(sector_data) < 2:
                continue
            
            sector_stats[sector] = {
                'company_count': len(sector_data),
                'esg_metrics': sector_data[available_esg].mean().to_dict() if available_esg else {},
                'financial_metrics': sector_data[available_financial].mean().to_dict() if available_financial else {},
                'esg_std': sector_data[available_esg].std().to_dict() if available_esg else {},
                'financial_std': sector_data[available_financial].std().to_dict() if available_financial else {}
            }
        
        # Análisis de varianza (ANOVA) para diferencias entre sectores
        anova_results = {}
        for metric in available_esg + available_financial:
            if metric in self.data.columns:
                groups = [group[metric].values for name, group in self.data.groupby('sector') 
                         if len(group) >= 2 and not group[metric].isna().all()]
                
                if len(groups) >= 2:
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        anova_results[metric] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        anova_results[metric] = {'error': 'Could not compute ANOVA'}
        
        results = {
            'sector_statistics': sector_stats,
            'anova_results': anova_results,
            'sector_mapping': sector_mapping
        }
        
        self.results['sector_analysis'] = results
        logger.info(f"Sector analysis completed for {len(sector_stats)} sectors")
        
        return results
    
    def temporal_analysis(self) -> Dict[str, Any]:
        """
        Análisis temporal de las métricas ESG y financieras
        
        Returns:
            Dict: Resultados del análisis temporal
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_processed_data() first.")
        
        logger.info("Starting temporal analysis")
        
        # Por ahora, análisis básico de tendencias
        # En una implementación completa, esto requeriría datos históricos de ESG
        
        temporal_results = {
            'data_points': len(self.data),
            'date_range': 'Current snapshot (no historical ESG data available)',
            'recommendation': 'Implement historical ESG data collection for temporal analysis'
        }
        
        self.results['temporal_analysis'] = temporal_results
        logger.info("Temporal analysis completed (limited by data availability)")
        
        return temporal_results
    
    def statistical_tests(self) -> Dict[str, Any]:
        """
        Tests estadísticos de significancia
        
        Returns:
            Dict: Resultados de los tests estadísticos
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_processed_data() first.")
        
        logger.info("Starting statistical tests")
        
        tests_results = {}
        
        # Test de normalidad para variables ESG
        esg_columns = [col for col in self.data.columns if 'Score' in col and col in self.data.columns]
        
        for col in esg_columns:
            data = self.data[col].dropna()
            if len(data) > 3:
                try:
                    statistic, p_value = stats.shapiro(data)
                    tests_results[f'{col}_normality'] = {
                        'test': 'Shapiro-Wilk',
                        'statistic': statistic,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
                except:
                    tests_results[f'{col}_normality'] = {'error': 'Could not compute normality test'}
        
        # Test de correlación de Spearman vs Pearson
        if 'ESG_Total' in self.data.columns and 'returns_1y' in self.data.columns:
            valid_data = self.data[['ESG_Total', 'returns_1y']].dropna()
            
            if len(valid_data) > 3:
                # Pearson correlation
                pearson_corr, pearson_p = stats.pearsonr(valid_data['ESG_Total'], valid_data['returns_1y'])
                
                # Spearman correlation
                spearman_corr, spearman_p = stats.spearmanr(valid_data['ESG_Total'], valid_data['returns_1y'])
                
                tests_results['esg_returns_correlation'] = {
                    'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
                    'spearman': {'correlation': spearman_corr, 'p_value': spearman_p}
                }
        
        self.results['statistical_tests'] = tests_results
        logger.info(f"Statistical tests completed: {len(tests_results)} tests performed")
        
        return tests_results
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Obtiene la importancia de características del modelo
        
        Args:
            model: Modelo entrenado
            feature_names: Nombres de las características
            
        Returns:
            Dict: Importancia de características
        """
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, abs(model.coef_)))
        else:
            return {}
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Ejecuta análisis comprehensivo completo
        
        Returns:
            Dict: Todos los resultados del análisis
        """
        logger.info("Starting comprehensive statistical analysis")
        
        # Ejecutar todos los análisis
        self.correlation_analysis()
        self.regression_analysis()
        self.sector_analysis()
        self.temporal_analysis()
        self.statistical_tests()
        
        # Crear resumen ejecutivo
        executive_summary = self._create_executive_summary()
        self.results['executive_summary'] = executive_summary
        
        logger.info("Comprehensive analysis completed")
        return self.results
    
    def _create_executive_summary(self) -> Dict[str, Any]:
        """
        Crea un resumen ejecutivo de los resultados
        
        Returns:
            Dict: Resumen ejecutivo
        """
        summary = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {
                'total_companies': len(self.data) if self.data is not None else 0,
                'esg_metrics_available': len([col for col in self.data.columns if 'Score' in col]) if self.data is not None else 0,
                'financial_metrics_available': len([col for col in self.data.columns if any(keyword in col.lower() for keyword in ['roe', 'roa', 'volatility', 'returns'])]) if self.data is not None else 0
            }
        }
        
        # Agregar hallazgos clave
        key_findings = []
        
        # Correlaciones significativas
        if 'correlation_analysis' in self.results:
            sig_corr = self.results['correlation_analysis'].get('significant_correlations', [])
            if sig_corr:
                key_findings.append(f"Found {len(sig_corr)} significant correlations between ESG and financial metrics")
        
        # Mejor modelo de regresión
        if 'regression_analysis' in self.results:
            reg_results = self.results['regression_analysis']
            best_model = max(reg_results.items(), key=lambda x: x[1].get('r2', 0) if isinstance(x[1], dict) and 'r2' in x[1] else 0)
            if isinstance(best_model[1], dict) and 'r2' in best_model[1]:
                key_findings.append(f"Best regression model: {best_model[0]} (R² = {best_model[1]['r2']:.3f})")
        
        # Diferencias por sector
        if 'sector_analysis' in self.results:
            sector_stats = self.results['sector_analysis'].get('sector_statistics', {})
            if sector_stats:
                key_findings.append(f"Analyzed {len(sector_stats)} sectors for ESG performance differences")
        
        summary['key_findings'] = key_findings
        summary['recommendations'] = [
            "Consider expanding ESG data collection for more comprehensive analysis",
            "Implement regular ESG monitoring for temporal trend analysis",
            "Focus on sectors with significant ESG-financial correlations"
        ]
        
        return summary
    
    def save_results(self, output_path: str = "data/processed/analysis_results.yaml") -> None:
        """
        Guarda los resultados del análisis
        
        Args:
            output_path: Ruta de salida para los resultados
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convertir resultados a formato serializable
        serializable_results = self._make_serializable(self.results)
        
        with open(output_path, 'w') as f:
            yaml.dump(serializable_results, f, default_flow_style=False, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
    
    def _make_serializable(self, obj):
        """Convierte objetos no serializables a formato YAML compatible"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def main():
    """Función principal para ejecutar análisis estadístico"""
    logger.info("Starting ESG Analytics statistical analysis")
    
    # Inicializar analizador
    analyzer = ESGAnalyzer()
    
    # Cargar datos procesados
    analyzer.load_processed_data()
    
    # Ejecutar análisis comprehensivo
    results = analyzer.comprehensive_analysis()
    
    # Guardar resultados
    analyzer.save_results()
    
    # Mostrar resumen
    if 'executive_summary' in results:
        summary = results['executive_summary']
        print("\n" + "="*60)
        print("📊 ESG ANALYTICS - STATISTICAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"📅 Analysis Date: {summary['analysis_date']}")
        print(f"🏢 Companies Analyzed: {summary['data_summary']['total_companies']}")
        print(f"📈 ESG Metrics: {summary['data_summary']['esg_metrics_available']}")
        print(f"💰 Financial Metrics: {summary['data_summary']['financial_metrics_available']}")
        
        print("\n🔍 Key Findings:")
        for finding in summary['key_findings']:
            print(f"   • {finding}")
        
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"   • {rec}")
        
        print("="*60)
    
    logger.info("Statistical analysis completed successfully")


if __name__ == "__main__":
    main()
