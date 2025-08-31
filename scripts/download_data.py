#!/usr/bin/env python3
"""
ESG Analytics IBEX35 - Data Download Script
===========================================

This script downloads financial and ESG data for IBEX35 companies.
It uses both Yahoo Finance API and web scraping for comprehensive data collection.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_collection import (
    get_ibex35_companies,
    download_financial_data,
    calculate_financial_metrics,
    save_data_to_files
)

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Web scraping imports
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_driver():
    """
    Configura el driver de Selenium
    
    Returns:
        webdriver.Chrome: Driver configurado
    """
    logger.info("Setting up Chrome driver")
    
    options = Options()
    options.add_argument('--headless')  # Ejecutar sin interfaz gráfica
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
    
    try:
        # Try to use webdriver_manager for automatic ChromeDriver installation
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        logger.info("Chrome driver setup successful")
        return driver
    except ImportError:
        logger.warning("webdriver_manager not available, trying system ChromeDriver")
        try:
            driver = webdriver.Chrome(options=options)
            logger.info("Chrome driver setup successful")
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            return None


def scrape_esg_data_sustainalytics(companies):
    """
    Web scraping de datos ESG básicos de Sustainalytics
    
    Args:
        companies: Lista de empresas para buscar
    
    Returns:
        pd.DataFrame: Datos ESG extraídos
    """
    driver = setup_driver()
    if not driver:
        logger.error("Could not setup Chrome driver")
        return pd.DataFrame()
    
    esg_data = []
    base_url = "https://www.sustainalytics.com/esg-rating"
    
    logger.info(f"Starting ESG data scraping for {len(companies)} companies")
    
    for i, company in enumerate(companies):
        try:
            logger.info(f"Scraping ESG data for {company} ({i+1}/{len(companies)})")
            
            # Buscar datos ESG públicos
            search_url = f"{base_url}/{company.lower().replace(' ', '-')}"
            driver.get(search_url)
            time.sleep(3)
            
            # Extraer datos disponibles públicamente
            # Nota: Implementar lógica específica según la estructura del sitio
            
            # Por ahora, creamos datos de ejemplo
            esg_data.append({
                'company': company,
                'esg_rating': None,
                'environmental_score': None,
                'social_score': None,
                'governance_score': None,
                'source': 'sustainalytics',
                'date_scraped': pd.Timestamp.now()
            })
            
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error scraping {company}: {e}")
            esg_data.append({
                'company': company,
                'esg_rating': None,
                'environmental_score': None,
                'social_score': None,
                'governance_score': None,
                'source': 'sustainalytics',
                'date_scraped': pd.Timestamp.now(),
                'error': str(e)
            })
    
    driver.quit()
    return pd.DataFrame(esg_data)


def scrape_company_sustainability_reports(companies):
    """
    Scraping de informes de sostenibilidad de sitios web corporativos
    
    Args:
        companies: Lista de empresas
    
    Returns:
        pd.DataFrame: Datos ESG extraídos
    """
    logger.info("Starting sustainability report scraping")
    
    # URLs de informes de sostenibilidad conocidas
    sustainability_urls = {
        'Iberdrola': 'https://www.iberdrola.com/sustainability',
        'Repsol': 'https://www.repsol.com/en/sustainability/index.cshtml',
        'Telefonica': 'https://www.telefonica.com/en/sustainability/',
        'Banco Santander': 'https://www.santander.com/en/sustainability',
        'BBVA': 'https://www.bbva.com/en/sustainability/',
        'Inditex': 'https://www.inditex.com/en/sustainability',
        'Ferrovial': 'https://www.ferrovial.com/en/sustainability/',
        'Endesa': 'https://www.endesa.com/en/sustainability',
        'Mapfre': 'https://www.mapfre.com/en/sustainability/',
        'Amadeus': 'https://www.amadeus.com/en/sustainability'
    }
    
    esg_data = []
    
    for company in companies:
        try:
            if company in sustainability_urls:
                url = sustainability_urls[company]
                logger.info(f"Scraping sustainability data for {company}")
                
                # Hacer request a la página
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parsear con BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extraer información ESG (ejemplo básico)
                # TODO: Implementar extracción específica según cada sitio
                
                esg_data.append({
                    'company': company,
                    'url': url,
                    'content_length': len(response.content),
                    'date_scraped': pd.Timestamp.now()
                })
                
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error scraping sustainability report for {company}: {e}")
    
    return pd.DataFrame(esg_data)


def download_all_data():
    """
    Función principal para descargar todos los datos del IBEX35 completo
    """
    logger.info("🚀 Starting complete IBEX35 data download")
    
    # Crear directorios necesarios
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/cleaned', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. Obtener lista completa de empresas IBEX35
    logger.info("📋 Step 1: Getting complete IBEX35 companies list")
    companies_df = get_ibex35_companies()
    companies_df.to_csv('data/raw/ibex35_companies_complete.csv', index=False)
    logger.info(f"✅ Loaded {len(companies_df)} companies")
    
    # 2. Descargar datos financieros completos (2019-2024)
    logger.info("📈 Step 2: Downloading complete financial data (2019-2024)")
    logger.info("⏱️  This may take several minutes due to rate limiting...")
    tickers = companies_df['ticker'].tolist()
    financial_data = download_financial_data(
        tickers, 
        start_date='2019-01-01', 
        end_date='2024-12-31',
        delay=1.0  # Delay de 1 segundo para respetar rate limits
    )
    
    # 3. Calcular métricas financieras completas
    logger.info("🧮 Step 3: Calculating comprehensive financial metrics")
    financial_metrics = calculate_financial_metrics(financial_data)
    financial_metrics.to_csv('data/raw/ibex35_financial_metrics_complete.csv', index=False)
    
    # 4. Guardar todos los datos financieros
    logger.info("💾 Step 4: Saving all financial data")
    save_data_to_files(financial_data, 'data/raw/ibex35_complete')
    
    # 5. Scraping de datos ESG (opcional - solo primeras 10 empresas para evitar bloqueos)
    logger.info("🌱 Step 5: Scraping ESG data (first 10 companies)")
    company_names = companies_df['company_name'].tolist()
    
    # Scraping de Sustainalytics
    sustainalytics_data = scrape_esg_data_sustainalytics(company_names[:10])
    if not sustainalytics_data.empty:
        sustainalytics_data.to_csv('data/raw/sustainalytics_esg_data.csv', index=False)
    
    # Scraping de informes de sostenibilidad
    sustainability_data = scrape_company_sustainability_reports(company_names[:10])
    if not sustainability_data.empty:
        sustainability_data.to_csv('data/raw/sustainability_reports_data.csv', index=False)
    
    # 6. Crear resumen completo de datos descargados
    logger.info("📊 Step 6: Creating comprehensive data summary")
    create_complete_data_summary(financial_data, financial_metrics, companies_df)
    
    logger.info("🎉 Complete IBEX35 data download finished successfully")


def create_complete_data_summary(financial_data, financial_metrics, companies_df):
    """
    Crea un resumen completo de los datos descargados
    
    Args:
        financial_data: Datos financieros descargados
        financial_metrics: Métricas financieras calculadas
        companies_df: DataFrame con información de empresas
    """
    from datetime import datetime
    
    # Estadísticas de descarga
    companies_with_data = len([k for k, v in financial_data.items() if not v['prices'].empty])
    companies_without_data = len(financial_data) - companies_with_data
    
    # Análisis por sector
    sector_analysis = {}
    for _, company in companies_df.iterrows():
        ticker = company['ticker']
        sector = company['sector']
        
        if ticker in financial_data and not financial_data[ticker]['prices'].empty:
            if sector not in sector_analysis:
                sector_analysis[sector] = {'with_data': 0, 'total': 0}
            sector_analysis[sector]['with_data'] += 1
        sector_analysis[sector]['total'] = sector_analysis.get(sector, {}).get('total', 0) + 1
    
    # Métricas financieras promedio
    avg_metrics = {}
    if not financial_metrics.empty:
        numeric_columns = financial_metrics.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if col != 'ticker':
                avg_metrics[f'avg_{col}'] = financial_metrics[col].mean()
    
    summary = {
        'download_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_companies': len(financial_data),
            'companies_with_data': companies_with_data,
            'companies_without_data': companies_without_data,
            'success_rate': f"{(companies_with_data/len(financial_data)*100):.1f}%",
            'data_period': '2019-2024'
        },
        'sector_analysis': sector_analysis,
        'average_metrics': avg_metrics,
        'data_files': {
            'companies_list': 'data/raw/ibex35_companies_complete.csv',
            'financial_metrics': 'data/raw/ibex35_financial_metrics_complete.csv',
            'price_data': 'data/raw/ibex35_complete/',
            'logs': 'logs/data_download.log'
        }
    }
    
    # Guardar resumen
    with open('data/raw/download_summary_complete.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    # Mostrar resumen en consola
    print("\n" + "="*60)
    print("📊 COMPLETE IBEX35 DATA DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Total companies: {summary['download_info']['total_companies']}")
    print(f"✅ Companies with data: {summary['download_info']['companies_with_data']}")
    print(f"✅ Success rate: {summary['download_info']['success_rate']}")
    print(f"✅ Data period: {summary['download_info']['data_period']}")
    
    print("\n📈 Sector Analysis:")
    for sector, stats in sector_analysis.items():
        success_rate = (stats['with_data'] / stats['total']) * 100
        print(f"   {sector}: {stats['with_data']}/{stats['total']} ({success_rate:.1f}%)")
    
    if avg_metrics:
        print("\n📊 Average Financial Metrics:")
        for metric, value in list(avg_metrics.items())[:5]:  # Mostrar solo las primeras 5
            print(f"   {metric}: {value:.2f}")
    
    print(f"\n💾 Files saved:")
    print(f"   📄 Companies list: {summary['data_files']['companies_list']}")
    print(f"   📊 Financial metrics: {summary['data_files']['financial_metrics']}")
    print(f"   📈 Price data: {summary['data_files']['price_data']}")
    print(f"   📝 Logs: {summary['data_files']['logs']}")
    
    print("\n🎉 Download completed successfully!")
    print("="*60)
    
    logger.info(f"Complete data summary created: {summary['download_info']}")


def create_data_summary(financial_data, financial_metrics):
    """
    Crea un resumen básico de los datos descargados (para compatibilidad)
    
    Args:
        financial_data: Datos financieros descargados
        financial_metrics: Métricas financieras calculadas
    """
    summary = {
        'download_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_companies': len(financial_data),
        'companies_with_data': len([k for k, v in financial_data.items() if not v['prices'].empty]),
        'data_period': '2019-2024',
        'financial_metrics_available': len(financial_metrics),
        'esg_data_sources': ['sustainalytics', 'sustainability_reports']
    }
    
    # Guardar resumen
    with open('data/raw/download_summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    logger.info(f"Data summary created: {summary}")


def main():
    """Función principal del script"""
    try:
        download_all_data()
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
