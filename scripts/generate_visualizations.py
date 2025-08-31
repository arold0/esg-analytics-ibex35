#!/usr/bin/env python3
"""
ESG Analytics IBEX35 - Visualization Generation Script
====================================================

This script generates comprehensive visualizations for the ESG analytics project.
It creates interactive plots, dashboards, and reports for data analysis.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from visualization import ESGVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/visualization_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_comprehensive_visualizations():
    """
    Genera todas las visualizaciones del proyecto
    """
    logger.info("🚀 Starting comprehensive visualization generation")
    
    # Crear directorios necesarios
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        # 1. Inicializar visualizador
        logger.info("📊 Step 1: Initializing ESG Visualizer")
        visualizer = ESGVisualizer()
        
        # 2. Cargar datos procesados
        logger.info("📋 Step 2: Loading processed data")
        visualizer.load_data()
        
        # 3. Cargar resultados de análisis
        logger.info("📈 Step 3: Loading analysis results")
        visualizer.load_analysis_results()
        
        # 4. Generar todas las visualizaciones
        logger.info("🎨 Step 4: Generating all visualizations")
        figures = visualizer.generate_all_visualizations()
        
        # 5. Crear reporte HTML
        logger.info("📄 Step 5: Creating visualization report")
        report_path = visualizer.create_visualization_report()
        
        # 6. Crear resumen de visualizaciones
        logger.info("📊 Step 6: Creating visualization summary")
        create_visualization_summary(figures, report_path)
        
        logger.info("🎉 Comprehensive visualization generation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {e}")
        return False


def create_visualization_summary(figures, report_path):
    """
    Crea un resumen de las visualizaciones generadas
    
    Args:
        figures: Diccionario con las figuras generadas
        report_path: Ruta al reporte HTML
    """
    import yaml
    
    # Crear resumen
    summary = {
        'visualization_metadata': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_visualizations': len(figures),
            'output_directory': 'reports/figures',
            'report_file': report_path
        },
        'generated_charts': {
            'correlation_heatmap': 'Mapa de calor de correlaciones ESG-Financieras',
            'esg_distributions': 'Distribuciones de métricas ESG',
            'sector_comparison': 'Comparación por sectores',
            'scatter_plots': 'Gráficos de dispersión ESG vs Financieras',
            'top_performers': 'Top y Bottom ESG Performers',
            'comprehensive_dashboard': 'Dashboard comprehensivo'
        },
        'file_formats': {
            'html': 'Interactive plots for web viewing',
            'png': 'Static images for reports and presentations'
        },
        'technical_details': {
            'library': 'Plotly',
            'interactivity': 'Full interactive plots with hover information',
            'responsive': 'Adaptable to different screen sizes',
            'exportable': 'Can be exported to various formats'
        }
    }
    
    # Guardar resumen
    summary_file = 'reports/figures/visualization_summary.yaml'
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    # Mostrar resumen en consola
    print("\n" + "="*70)
    print("📊 ESG ANALYTICS IBEX35 - VISUALIZATION SUMMARY")
    print("="*70)
    print(f"📅 Generation Date: {summary['visualization_metadata']['date']}")
    print(f"📈 Total Visualizations: {summary['visualization_metadata']['total_visualizations']}")
    print(f"📁 Output Directory: {summary['visualization_metadata']['output_directory']}")
    print(f"📄 Report File: {summary['visualization_metadata']['report_file']}")
    
    print(f"\n🎨 Generated Charts:")
    for chart_name, description in summary['generated_charts'].items():
        print(f"   • {chart_name.replace('_', ' ').title()}: {description}")
    
    print(f"\n💾 File Formats:")
    for format_type, description in summary['file_formats'].items():
        print(f"   • {format_type.upper()}: {description}")
    
    print(f"\n🔧 Technical Details:")
    for detail, value in summary['technical_details'].items():
        print(f"   • {detail.replace('_', ' ').title()}: {value}")
    
    print(f"\n📂 Files Generated:")
    print(f"   📊 Interactive HTML: reports/figures/*.html")
    print(f"   🖼️  Static Images: reports/figures/*.png")
    print(f"   📄 Summary Report: {summary_file}")
    print(f"   🌐 Main Report: {report_path}")
    
    print("\n🎉 Visualization generation completed successfully!")
    print("="*70)
    
    logger.info(f"Visualization summary created: {summary_file}")


def main():
    """Función principal del script"""
    try:
        logger.info("Starting ESG Analytics visualization generation")
        
        success = generate_comprehensive_visualizations()
        
        if success:
            logger.info("✅ Visualization generation completed successfully")
            return 0
        else:
            logger.error("❌ Visualization generation failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Visualization generation failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
