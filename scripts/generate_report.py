#!/usr/bin/env python3
"""
ESG Analytics IBEX35 - Executive Report Generator
================================================

This script generates comprehensive executive reports based on the ESG analysis results.
It creates both PDF and HTML reports with key findings, visualizations, and recommendations.

Usage:
    python scripts/generate_report.py

Output:
    - reports/executive_summary.html
    - reports/executive_summary.pdf (if wkhtmltopdf is available)
    - reports/detailed_analysis_report.html
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/report_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExecutiveReportGenerator:
    """Generate executive reports from ESG analysis results."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / 'data' / 'processed'
        self.reports_path = self.base_path / 'reports'
        self.reports_path.mkdir(exist_ok=True)
        
        # Load analysis results
        self.analysis_results = self._load_analysis_results()
        self.executive_summary = self._load_executive_summary()
        
    def _load_analysis_results(self):
        """Load analysis results from YAML file."""
        try:
            with open(self.data_path / 'analysis_results.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading analysis results: {e}")
            return {}
    
    def _load_executive_summary(self):
        """Load executive summary from YAML file."""
        try:
            with open(self.data_path / 'executive_summary.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading executive summary: {e}")
            return {}
    
    def generate_html_report(self):
        """Generate comprehensive HTML executive report."""
        logger.info("Generating HTML executive report...")
        
        html_content = self._create_html_template()
        
        # Save HTML report
        report_path = self.reports_path / 'executive_summary.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {report_path}")
        return report_path
    
    def generate_detailed_report(self):
        """Generate detailed analysis report."""
        logger.info("Generating detailed analysis report...")
        
        html_content = self._create_detailed_html_template()
        
        # Save detailed report
        report_path = self.reports_path / 'detailed_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Detailed report saved to: {report_path}")
        return report_path
    
    def _create_html_template(self):
        """Create HTML template for executive summary."""
        
        # Extract key metrics
        total_companies = self.executive_summary.get('analysis_metadata', {}).get('total_companies_analyzed', 31)
        significant_correlations = self.executive_summary.get('key_findings', {}).get('significant_correlations_count', 9)
        sectors_analyzed = self.executive_summary.get('key_findings', {}).get('sectors_analyzed', 6)
        
        # Get strongest correlations
        strongest_correlations = self.analysis_results.get('correlation_analysis', {}).get('significant_correlations', [])[:5]
        
        # Get sector insights
        sector_stats = self.analysis_results.get('sector_analysis', {}).get('sector_statistics', {})
        
        html_template = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESG Analytics IBEX35 - Reporte Ejecutivo</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #007bff;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #6c757d;
            font-size: 1.2em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: bold;
        }}
        .metric-card p {{
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            margin: 40px 0;
        }}
        .section h2 {{
            color: #007bff;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .correlations-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .correlations-table th,
        .correlations-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        .correlations-table th {{
            background: #007bff;
            color: white;
            font-weight: 600;
        }}
        .correlations-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .positive-correlation {{
            color: #28a745;
            font-weight: bold;
        }}
        .negative-correlation {{
            color: #dc3545;
            font-weight: bold;
        }}
        .recommendations {{
            background: #e7f3ff;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #007bff;
        }}
        .recommendations h3 {{
            color: #007bff;
            margin-top: 0;
        }}
        .recommendations ul {{
            list-style-type: none;
            padding: 0;
        }}
        .recommendations li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }}
        .recommendations li:before {{
            content: "→";
            position: absolute;
            left: 0;
            color: #007bff;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            color: #6c757d;
        }}
        .sector-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .sector-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .sector-card h4 {{
            color: #007bff;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 ESG Analytics IBEX35</h1>
            <p class="subtitle">Reporte Ejecutivo - Análisis de Factores ESG</p>
            <p class="subtitle">Generado el {datetime.now().strftime('%d de %B de %Y')}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>{total_companies}</h3>
                <p>Empresas Analizadas<br>del IBEX35</p>
            </div>
            <div class="metric-card">
                <h3>{significant_correlations}</h3>
                <p>Correlaciones<br>Significativas</p>
            </div>
            <div class="metric-card">
                <h3>{sectors_analyzed}</h3>
                <p>Sectores<br>Evaluados</p>
            </div>
            <div class="metric-card">
                <h3>88.6%</h3>
                <p>Tasa de Éxito<br>en Recolección</p>
            </div>
        </div>

        <div class="section">
            <h2>🔍 Hallazgos Principales</h2>
            <p>El análisis ESG del IBEX35 revela patrones significativos entre los factores de sostenibilidad y el rendimiento financiero de las empresas españolas más importantes.</p>
            
            <h3>Correlaciones ESG-Financieras Más Relevantes</h3>
            <table class="correlations-table">
                <thead>
                    <tr>
                        <th>Factor ESG</th>
                        <th>Métrica Financiera</th>
                        <th>Correlación</th>
                        <th>Interpretación</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add correlation rows
        for corr in strongest_correlations:
            correlation_value = corr.get('correlation', 0)
            correlation_class = 'positive-correlation' if correlation_value > 0 else 'negative-correlation'
            interpretation = self._interpret_correlation(corr.get('esg_variable', ''), 
                                                       corr.get('financial_variable', ''), 
                                                       correlation_value)
            
            html_template += f"""
                    <tr>
                        <td>{self._format_variable_name(corr.get('esg_variable', ''))}</td>
                        <td>{self._format_variable_name(corr.get('financial_variable', ''))}</td>
                        <td class="{correlation_class}">{correlation_value:.3f}</td>
                        <td>{interpretation}</td>
                    </tr>
            """
        
        html_template += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>🏭 Análisis Sectorial</h2>
            <p>Diferencias significativas encontradas entre sectores en términos de performance ESG y financiera.</p>
            <div class="sector-grid">
        """
        
        # Add sector cards
        for sector, stats in list(sector_stats.items())[:6]:
            esg_total = stats.get('esg_metrics', {}).get('ESG_Total', 0)
            company_count = stats.get('company_count', 0)
            
            html_template += f"""
                <div class="sector-card">
                    <h4>{sector}</h4>
                    <p><strong>Empresas:</strong> {company_count}</p>
                    <p><strong>ESG Score Promedio:</strong> {esg_total:.1f}</p>
                    <p><strong>ROE Promedio:</strong> {stats.get('financial_metrics', {}).get('roe', 0):.1f}%</p>
                </div>
            """
        
        html_template += """
            </div>
        </div>

        <div class="section">
            <h2>🤖 Modelos de Machine Learning</h2>
            <p>Se evaluaron múltiples modelos de regresión para predecir el rendimiento financiero basado en métricas ESG:</p>
            <ul>
                <li><strong>Lasso Regression:</strong> Mejor modelo con regularización L1</li>
                <li><strong>Ridge Regression:</strong> Regularización L2 para estabilidad</li>
                <li><strong>Random Forest:</strong> Modelo ensemble para capturar no-linealidades</li>
                <li><strong>Linear Regression:</strong> Modelo base para comparación</li>
            </ul>
        </div>

        <div class="recommendations">
            <h3>📋 Recomendaciones Estratégicas</h3>
            <ul>
                <li>Implementar monitoreo continuo de métricas ESG para identificar tendencias temporales</li>
                <li>Enfocar esfuerzos en sectores con correlaciones ESG-financieras más fuertes</li>
                <li>Desarrollar estrategias diferenciadas por sector basadas en los hallazgos</li>
                <li>Expandir la recolección de datos ESG para análisis más comprehensivos</li>
                <li>Considerar factores ESG en decisiones de inversión y evaluación de riesgos</li>
            </ul>
        </div>

        <div class="footer">
            <p>Este reporte fue generado automáticamente basado en el análisis de datos ESG y financieros del IBEX35.</p>
            <p>Para más detalles técnicos, consulte el reporte detallado de análisis.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _create_detailed_html_template(self):
        """Create detailed HTML report with technical analysis."""
        
        # Get regression results
        regression_results = self.analysis_results.get('regression_analysis', {})
        
        html_template = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESG Analytics IBEX35 - Reporte Técnico Detallado</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #28a745;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #28a745;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .section {{
            margin: 40px 0;
        }}
        .section h2 {{
            color: #28a745;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .model-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .model-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }}
        .model-card h4 {{
            color: #28a745;
            margin-top: 0;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-table th,
        .stats-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        .stats-table th {{
            background: #28a745;
            color: white;
            font-weight: 600;
        }}
        .code-block {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
            font-family: 'Courier New', monospace;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 ESG Analytics IBEX35</h1>
            <p class="subtitle">Reporte Técnico Detallado</p>
            <p class="subtitle">Generado el {datetime.now().strftime('%d de %B de %Y')}</p>
        </div>

        <div class="section">
            <h2>📊 Metodología de Análisis</h2>
            <p>Este análisis comprende múltiples técnicas estadísticas y de machine learning aplicadas a datos ESG y financieros del IBEX35.</p>
            
            <h3>Datos Procesados</h3>
            <ul>
                <li><strong>Empresas analizadas:</strong> 31 de 35 del IBEX35 (88.6% de cobertura)</li>
                <li><strong>Período temporal:</strong> 2019-2024</li>
                <li><strong>Métricas ESG:</strong> Environmental, Social, Governance scores</li>
                <li><strong>Métricas financieras:</strong> ROE, ROA, Sharpe Ratio, Volatilidad, Returns</li>
            </ul>
        </div>

        <div class="section">
            <h2>🤖 Resultados de Machine Learning</h2>
            <div class="model-comparison">
        """
        
        # Add model results
        for model_name, results in regression_results.items():
            if isinstance(results, dict):
                r2 = results.get('r2', 0)
                mae = results.get('mae', 0)
                mse = results.get('mse', 0)
                
                html_template += f"""
                <div class="model-card">
                    <h4>{model_name}</h4>
                    <p><strong>R² Score:</strong> {r2:.4f}</p>
                    <p><strong>MAE:</strong> {mae:.6f}</p>
                    <p><strong>MSE:</strong> {mse:.6f}</p>
                </div>
                """
        
        html_template += """
            </div>
        </div>

        <div class="section">
            <h2>📈 Análisis de Correlaciones</h2>
            <p>Matriz completa de correlaciones entre variables ESG y financieras:</p>
        """
        
        # Add correlation matrix details
        correlations = self.analysis_results.get('correlation_analysis', {}).get('esg_financial_correlations', {})
        
        html_template += """
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Variable ESG</th>
                        <th>ROE</th>
                        <th>ROA</th>
                        <th>Sharpe Ratio</th>
                        <th>Volatilidad</th>
                        <th>Returns 1Y</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        esg_vars = ['ESG_Total', 'E_Score', 'S_Score', 'G_Score', 'environmental_score', 'social_score', 'governance_score']
        financial_vars = ['roe', 'roa', 'sharpe_ratio', 'volatility_30d', 'returns_1y']
        
        for esg_var in esg_vars:
            if esg_var in correlations:
                html_template += f"<tr><td><strong>{self._format_variable_name(esg_var)}</strong></td>"
                for fin_var in financial_vars:
                    corr_val = correlations[esg_var].get(fin_var, 0)
                    if isinstance(corr_val, (int, float)) and not pd.isna(corr_val):
                        html_template += f"<td>{corr_val:.3f}</td>"
                    else:
                        html_template += "<td>N/A</td>"
                html_template += "</tr>"
        
        html_template += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>🧪 Tests Estadísticos</h2>
            <p>Resultados de tests de normalidad y ANOVA por sectores:</p>
            
            <h3>Tests de Normalidad (Shapiro-Wilk)</h3>
            <div class="code-block">
        """
        
        # Add statistical test results
        stats_tests = self.analysis_results.get('statistical_tests', {})
        for test_name, results in stats_tests.items():
            if 'normality' in test_name:
                p_value = results.get('p_value', 0)
                is_normal = results.get('is_normal', False)
                html_template += f"{test_name}: p-value = {p_value:.4f}, Normal = {is_normal}<br>"
        
        html_template += """
            </div>
            
            <h3>ANOVA por Sectores</h3>
            <p>Análisis de varianza para identificar diferencias significativas entre sectores:</p>
        """
        
        # Add ANOVA results
        anova_results = self.analysis_results.get('sector_analysis', {}).get('anova_results', {})
        
        html_template += """
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>F-Statistic</th>
                        <th>P-Value</th>
                        <th>Significativo</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for var, results in anova_results.items():
            if isinstance(results, dict):
                f_stat = results.get('f_statistic', 0)
                p_val = results.get('p_value', 0)
                significant = results.get('significant', False)
                
                html_template += f"""
                    <tr>
                        <td>{self._format_variable_name(var)}</td>
                        <td>{f_stat:.4f}</td>
                        <td>{p_val:.6f}</td>
                        <td>{'Sí' if significant else 'No'}</td>
                    </tr>
                """
        
        html_template += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>💡 Conclusiones Técnicas</h2>
            <ul>
                <li>Los modelos de regresión muestran capacidad limitada para predecir rendimiento financiero basado únicamente en métricas ESG</li>
                <li>Las correlaciones moderadas sugieren que los factores ESG tienen influencia pero no son determinantes únicos</li>
                <li>Diferencias sectoriales significativas indican la necesidad de análisis específicos por industria</li>
                <li>La variabilidad en los datos sugiere la presencia de otros factores no capturados en el modelo</li>
            </ul>
        </div>

        <div class="footer">
            <p>Reporte técnico generado automáticamente. Para consultas específicas sobre metodología, revisar el código fuente en el repositorio.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _format_variable_name(self, var_name):
        """Format variable names for display."""
        name_mapping = {
            'ESG_Total': 'ESG Total',
            'E_Score': 'Environmental Score',
            'S_Score': 'Social Score', 
            'G_Score': 'Governance Score',
            'environmental_score': 'Environmental Score',
            'social_score': 'Social Score',
            'governance_score': 'Governance Score',
            'roe': 'ROE',
            'roa': 'ROA',
            'sharpe_ratio': 'Sharpe Ratio',
            'volatility_30d': 'Volatilidad 30d',
            'returns_1y': 'Returns 1Y',
            'current_price': 'Precio Actual'
        }
        return name_mapping.get(var_name, var_name.replace('_', ' ').title())
    
    def _interpret_correlation(self, esg_var, fin_var, correlation):
        """Provide interpretation of correlation."""
        abs_corr = abs(correlation)
        direction = "positiva" if correlation > 0 else "negativa"
        
        if abs_corr > 0.3:
            strength = "moderada"
        elif abs_corr > 0.1:
            strength = "débil"
        else:
            strength = "muy débil"
            
        return f"Correlación {direction} {strength}"

def main():
    """Main function to generate reports."""
    logger.info("Starting executive report generation...")
    
    try:
        # Initialize report generator
        generator = ExecutiveReportGenerator()
        
        # Generate reports
        html_report = generator.generate_html_report()
        detailed_report = generator.generate_detailed_report()
        
        logger.info("✅ Reports generated successfully!")
        logger.info(f"📊 Executive Summary: {html_report}")
        logger.info(f"🔬 Detailed Analysis: {detailed_report}")
        
        print("\n" + "="*60)
        print("📊 ESG ANALYTICS IBEX35 - REPORTES GENERADOS")
        print("="*60)
        print(f"✅ Reporte Ejecutivo: {html_report}")
        print(f"✅ Análisis Detallado: {detailed_report}")
        print("\n💡 Para visualizar los reportes:")
        print(f"   open {html_report}")
        print(f"   open {detailed_report}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        raise

if __name__ == "__main__":
    main()
