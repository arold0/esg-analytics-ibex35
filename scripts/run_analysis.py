#!/usr/bin/env python3
"""
ESG Analytics IBEX35 - Statistical Analysis Script
==================================================

This script runs comprehensive statistical analysis on the processed IBEX35 data.
It includes correlation analysis, regression models, sector analysis, and more.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis import ESGAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/statistical_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_comprehensive_analysis():
    """
    Ejecuta análisis estadístico comprehensivo
    """
    logger.info("🚀 Starting comprehensive statistical analysis")
    
    # Crear directorios necesarios
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        # 1. Inicializar analizador
        logger.info("📊 Step 1: Initializing ESG Analyzer")
        analyzer = ESGAnalyzer()
        
        # 2. Cargar datos procesados
        logger.info("📋 Step 2: Loading processed data")
        analyzer.load_processed_data()
        
        # 3. Ejecutar análisis comprehensivo
        logger.info("🔬 Step 3: Running comprehensive analysis")
        results = analyzer.comprehensive_analysis()
        
        # 4. Guardar resultados
        logger.info("💾 Step 4: Saving analysis results")
        analyzer.save_results()
        
        # 5. Crear resumen ejecutivo
        logger.info("📈 Step 5: Creating executive summary")
        create_executive_summary(results)
        
        logger.info("🎉 Comprehensive statistical analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False


def create_executive_summary(results):
    """
    Crea un resumen ejecutivo detallado
    
    Args:
        results: Resultados del análisis
    """
    import yaml
    
    # Extraer información clave
    summary = results.get('executive_summary', {})
    
    # Estadísticas de correlación
    correlation_info = results.get('correlation_analysis', {})
    significant_correlations = correlation_info.get('significant_correlations', [])
    
    # Estadísticas de regresión
    regression_info = results.get('regression_analysis', {})
    best_model = None
    best_r2 = 0
    
    for model_name, model_results in regression_info.items():
        if isinstance(model_results, dict) and 'r2' in model_results:
            if model_results['r2'] > best_r2:
                best_r2 = model_results['r2']
                best_model = model_name
    
    # Estadísticas por sector
    sector_info = results.get('sector_analysis', {})
    sector_stats = sector_info.get('sector_statistics', {})
    
    # Crear resumen ejecutivo detallado
    executive_summary = {
        'analysis_metadata': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_companies_analyzed': summary.get('data_summary', {}).get('total_companies', 0),
            'esg_metrics_available': summary.get('data_summary', {}).get('esg_metrics_available', 0),
            'financial_metrics_available': summary.get('data_summary', {}).get('financial_metrics_available', 0)
        },
        'key_findings': {
            'significant_correlations_count': len(significant_correlations),
            'best_regression_model': best_model,
            'best_model_r2': best_r2,
            'sectors_analyzed': len(sector_stats),
            'strongest_correlations': significant_correlations[:5] if significant_correlations else []
        },
        'sector_insights': {
            'sector_count': len(sector_stats),
            'sectors_with_data': list(sector_stats.keys())
        },
        'statistical_tests': {
            'tests_performed': len(results.get('statistical_tests', {})),
            'normality_tests': len([k for k in results.get('statistical_tests', {}).keys() if 'normality' in k])
        },
        'recommendations': summary.get('recommendations', [])
    }
    
    # Guardar resumen ejecutivo
    summary_file = 'data/processed/executive_summary.yaml'
    with open(summary_file, 'w') as f:
        yaml.dump(executive_summary, f, default_flow_style=False, indent=2)
    
    # Mostrar resumen en consola
    print("\n" + "="*70)
    print("📊 ESG ANALYTICS IBEX35 - EXECUTIVE SUMMARY")
    print("="*70)
    print(f"📅 Analysis Date: {executive_summary['analysis_metadata']['date']}")
    print(f"🏢 Companies Analyzed: {executive_summary['analysis_metadata']['total_companies_analyzed']}")
    print(f"📈 ESG Metrics: {executive_summary['analysis_metadata']['esg_metrics_available']}")
    print(f"💰 Financial Metrics: {executive_summary['analysis_metadata']['financial_metrics_available']}")
    
    print(f"\n🔍 Key Statistical Findings:")
    print(f"   • Significant Correlations: {executive_summary['key_findings']['significant_correlations_count']}")
    print(f"   • Best Regression Model: {executive_summary['key_findings']['best_regression_model']}")
    print(f"   • Model R² Score: {executive_summary['key_findings']['best_model_r2']:.3f}")
    print(f"   • Sectors Analyzed: {executive_summary['key_findings']['sectors_analyzed']}")
    
    if executive_summary['key_findings']['strongest_correlations']:
        print(f"\n📈 Strongest ESG-Financial Correlations:")
        for i, corr in enumerate(executive_summary['key_findings']['strongest_correlations'][:3], 1):
            print(f"   {i}. {corr['esg_variable']} ↔ {corr['financial_variable']}: {corr['correlation']:.3f} ({corr['strength']})")
    
    print(f"\n🏭 Sector Analysis:")
    print(f"   • Sectors with Data: {executive_summary['sector_insights']['sector_count']}")
    print(f"   • Sectors: {', '.join(executive_summary['sector_insights']['sectors_with_data'][:5])}")
    
    print(f"\n🧪 Statistical Tests:")
    print(f"   • Tests Performed: {executive_summary['statistical_tests']['tests_performed']}")
    print(f"   • Normality Tests: {executive_summary['statistical_tests']['normality_tests']}")
    
    print(f"\n💡 Key Recommendations:")
    for i, rec in enumerate(executive_summary['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n💾 Files Generated:")
    print(f"   📊 Analysis Results: data/processed/analysis_results.yaml")
    print(f"   📈 Executive Summary: {summary_file}")
    print(f"   📝 Analysis Logs: logs/statistical_analysis.log")
    
    print("\n🎉 Statistical analysis completed successfully!")
    print("="*70)
    
    logger.info(f"Executive summary created: {summary_file}")


def main():
    """Función principal del script"""
    try:
        logger.info("Starting ESG Analytics statistical analysis")
        
        success = run_comprehensive_analysis()
        
        if success:
            logger.info("✅ Statistical analysis completed successfully")
            return 0
        else:
            logger.error("❌ Statistical analysis failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Analysis failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
