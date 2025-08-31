#!/usr/bin/env python3
"""
ESG Analytics IBEX35 - Data Processing Script
============================================

This script processes and cleans the downloaded IBEX35 data.
It includes data cleaning, feature engineering, and preparation for analysis.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing import ESGDataProcessor
from data_collection import get_ibex35_companies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_financial_data_from_files(data_dir: str = "data/raw/ibex35_complete/") -> dict:
    """
    Carga los datos financieros desde archivos CSV
    
    Args:
        data_dir: Directorio con los archivos de datos
        
    Returns:
        dict: Diccionario con datos financieros por empresa
    """
    logger.info(f"Loading financial data from {data_dir}")
    
    financial_data = {}
    files_loaded = 0
    
    # Cargar archivos de precios
    for file in os.listdir(data_dir):
        if file.endswith('_prices.csv'):
            ticker = file.replace('_prices.csv', '')
            
            try:
                file_path = os.path.join(data_dir, file)
                prices_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Cargar información de la empresa si existe
                info_file = os.path.join(data_dir, f"{ticker}_info.csv")
                info_dict = {}
                if os.path.exists(info_file):
                    info_df = pd.read_csv(info_file)
                    if not info_df.empty:
                        info_dict = info_df.iloc[0].to_dict()
                
                # Cargar datos financieros si existen
                financials_file = os.path.join(data_dir, f"{ticker}_financials.csv")
                financials_df = pd.DataFrame()
                if os.path.exists(financials_file):
                    financials_df = pd.read_csv(financials_file, index_col=0)
                
                financial_data[ticker] = {
                    'prices': prices_df,
                    'info': info_dict,
                    'financials': financials_df
                }
                
                files_loaded += 1
                logger.debug(f"Loaded data for {ticker}")
                
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
                continue
    
    logger.info(f"Successfully loaded data for {files_loaded} companies")
    return financial_data


def create_sample_esg_data(companies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea datos ESG de muestra para demostración
    
    Args:
        companies_df: DataFrame con información de empresas
        
    Returns:
        pd.DataFrame: Datos ESG de muestra
    """
    logger.info("Creating sample ESG data for demonstration")
    
    # Crear datos ESG de muestra
    esg_data = []
    
    for _, company in companies_df.iterrows():
        ticker = company['symbol']  # Usar 'symbol' de la configuración
        sector = company['sector']
        
        # Generar datos ESG de muestra basados en el sector
        if 'Financial' in sector:
            # Bancos suelen tener mejor governance
            esg_record = {
                'ticker': company['symbol'],  # Usar 'symbol' de la configuración
                'environmental_score': np.random.normal(60, 15),
                'social_score': np.random.normal(70, 10),
                'governance_score': np.random.normal(80, 8),
                'carbon_emissions': np.random.normal(50, 20),
                'energy_efficiency': np.random.normal(65, 15),
                'gender_diversity': np.random.normal(75, 10),
                'board_independence': np.random.normal(85, 8)
            }
        elif 'Energy' in sector or 'Utilities' in sector:
            # Sector energético suele tener mejor environmental
            esg_record = {
                'ticker': ticker,
                'environmental_score': np.random.normal(75, 12),
                'social_score': np.random.normal(65, 15),
                'governance_score': np.random.normal(70, 12),
                'carbon_emissions': np.random.normal(30, 15),
                'energy_efficiency': np.random.normal(80, 10),
                'gender_diversity': np.random.normal(60, 15),
                'board_independence': np.random.normal(75, 10)
            }
        else:
            # Otros sectores
            esg_record = {
                'ticker': ticker,
                'environmental_score': np.random.normal(65, 15),
                'social_score': np.random.normal(68, 12),
                'governance_score': np.random.normal(72, 10),
                'carbon_emissions': np.random.normal(45, 18),
                'energy_efficiency': np.random.normal(70, 12),
                'gender_diversity': np.random.normal(65, 12),
                'board_independence': np.random.normal(78, 8)
            }
        
        # Asegurar que los valores estén en rangos razonables
        for key, value in esg_record.items():
            if key != 'ticker':
                esg_record[key] = max(0, min(100, value))
        
        esg_data.append(esg_record)
    
    esg_df = pd.DataFrame(esg_data)
    logger.info(f"Created sample ESG data for {len(esg_df)} companies")
    return esg_df


def process_complete_dataset():
    """
    Procesa el dataset completo del IBEX35
    """
    logger.info("🚀 Starting complete IBEX35 data processing")
    
    # Crear directorios necesarios
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/cleaned', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. Cargar datos financieros
    logger.info("📊 Step 1: Loading financial data from files")
    financial_data = load_financial_data_from_files()
    
    if not financial_data:
        logger.error("No financial data found. Please run data download first.")
        return False
    
    # 2. Cargar información de empresas desde configuración
    logger.info("📋 Step 2: Loading companies information from config")
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    companies_data = config['ibex35']['companies']
    companies_df = pd.DataFrame(companies_data)
    
    # 3. Crear datos ESG de muestra
    logger.info("🌱 Step 3: Creating sample ESG data")
    esg_data = create_sample_esg_data(companies_df)
    
    # 4. Inicializar procesador
    logger.info("🔧 Step 4: Initializing data processor")
    processor = ESGDataProcessor()
    
    # 5. Limpiar datos financieros
    logger.info("🧹 Step 5: Cleaning financial data")
    cleaned_financial = processor.clean_financial_data(financial_data)
    
    # 6. Crear scores ESG compuestos
    logger.info("📈 Step 6: Creating ESG composite scores")
    esg_with_scores = processor.create_esg_composite_scores(esg_data)
    
    # 7. Crear dataset procesado
    logger.info("🔗 Step 7: Creating processed dataset")
    processed_dataset = processor.create_processed_dataset(cleaned_financial, esg_with_scores)
    
    # 8. Guardar datos procesados
    logger.info("💾 Step 8: Saving processed data")
    processor.save_processed_data(processed_dataset)
    
    # 9. Crear resumen final
    logger.info("📊 Step 9: Creating final summary")
    create_processing_summary(processed_dataset, cleaned_financial, esg_with_scores)
    
    logger.info("🎉 Complete data processing finished successfully")
    return True


def create_processing_summary(processed_dataset: pd.DataFrame, 
                            cleaned_financial: dict, 
                            esg_data: pd.DataFrame):
    """
    Crea un resumen final del procesamiento
    
    Args:
        processed_dataset: Dataset procesado final
        cleaned_financial: Datos financieros limpios
        esg_data: Datos ESG procesados
    """
    import yaml
    
    # Estadísticas del procesamiento
    summary = {
        'processing_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_companies_processed': len(processed_dataset),
            'financial_companies_cleaned': len(cleaned_financial),
            'esg_companies_processed': len(esg_data) if not esg_data.empty else 0
        },
        'data_quality': {
            'missing_values_percentage': (processed_dataset.isnull().sum() / len(processed_dataset) * 100).to_dict(),
            'columns_with_data': list(processed_dataset.columns),
            'numeric_columns': list(processed_dataset.select_dtypes(include=[np.number]).columns)
        },
        'esg_scores_summary': {
            'avg_environmental_score': esg_data['E_Score'].mean() if 'E_Score' in esg_data.columns else None,
            'avg_social_score': esg_data['S_Score'].mean() if 'S_Score' in esg_data.columns else None,
            'avg_governance_score': esg_data['G_Score'].mean() if 'G_Score' in esg_data.columns else None,
            'avg_esg_total': esg_data['ESG_Total'].mean() if 'ESG_Total' in esg_data.columns else None
        },
        'financial_metrics_summary': {
            'avg_roe': processed_dataset['roe'].mean() if 'roe' in processed_dataset.columns else None,
            'avg_roa': processed_dataset['roa'].mean() if 'roa' in processed_dataset.columns else None,
            'avg_volatility': processed_dataset['volatility_30d'].mean() if 'volatility_30d' in processed_dataset.columns else None,
            'avg_sharpe_ratio': processed_dataset['sharpe_ratio'].mean() if 'sharpe_ratio' in processed_dataset.columns else None
        }
    }
    
    # Guardar resumen
    summary_file = 'data/processed/processing_summary_final.yaml'
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    # Mostrar resumen en consola
    print("\n" + "="*60)
    print("📊 IBEX35 DATA PROCESSING SUMMARY")
    print("="*60)
    print(f"✅ Companies processed: {summary['processing_info']['total_companies_processed']}")
    print(f"✅ Financial data cleaned: {summary['processing_info']['financial_companies_cleaned']}")
    print(f"✅ ESG data processed: {summary['processing_info']['esg_companies_processed']}")
    
    print("\n📈 ESG Scores Summary:")
    if summary['esg_scores_summary']['avg_environmental_score']:
        print(f"   Environmental: {summary['esg_scores_summary']['avg_environmental_score']:.1f}")
    if summary['esg_scores_summary']['avg_social_score']:
        print(f"   Social: {summary['esg_scores_summary']['avg_social_score']:.1f}")
    if summary['esg_scores_summary']['avg_governance_score']:
        print(f"   Governance: {summary['esg_scores_summary']['avg_governance_score']:.1f}")
    if summary['esg_scores_summary']['avg_esg_total']:
        print(f"   Total ESG: {summary['esg_scores_summary']['avg_esg_total']:.1f}")
    
    print("\n💰 Financial Metrics Summary:")
    if summary['financial_metrics_summary']['avg_roe']:
        print(f"   Average ROE: {summary['financial_metrics_summary']['avg_roe']:.2f}%")
    if summary['financial_metrics_summary']['avg_roa']:
        print(f"   Average ROA: {summary['financial_metrics_summary']['avg_roa']:.2f}%")
    if summary['financial_metrics_summary']['avg_volatility']:
        print(f"   Average Volatility: {summary['financial_metrics_summary']['avg_volatility']:.2f}%")
    if summary['financial_metrics_summary']['avg_sharpe_ratio']:
        print(f"   Average Sharpe Ratio: {summary['financial_metrics_summary']['avg_sharpe_ratio']:.2f}")
    
    print(f"\n💾 Files saved:")
    print(f"   📄 Processed data: data/processed/ibex35_processed_data.csv")
    print(f"   📊 Processing summary: data/processed/processing_summary.yaml")
    print(f"   📈 Final summary: {summary_file}")
    
    print("\n🎉 Data processing completed successfully!")
    print("="*60)
    
    logger.info(f"Final processing summary created: {summary_file}")


def main():
    """Función principal del script"""
    try:
        logger.info("Starting IBEX35 data processing")
        
        success = process_complete_dataset()
        
        if success:
            logger.info("✅ Data processing completed successfully")
            return 0
        else:
            logger.error("❌ Data processing failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Processing failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
