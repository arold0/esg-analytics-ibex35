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
    Función principal para descargar todos los datos
    """
    logger.info("Starting comprehensive data download")
    
    # Crear directorios necesarios
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. Obtener lista de empresas IBEX35
    logger.info("Step 1: Getting IBEX35 companies list")
    companies_df = get_ibex35_companies()
    companies_df.to_csv('data/raw/ibex35_companies.csv', index=False)
    
    # 2. Descargar datos financieros
    logger.info("Step 2: Downloading financial data")
    tickers = companies_df['ticker'].tolist()
    financial_data = download_financial_data(tickers)
    
    # 3. Calcular métricas financieras
    logger.info("Step 3: Calculating financial metrics")
    financial_metrics = calculate_financial_metrics(financial_data)
    financial_metrics.to_csv('data/raw/financial_metrics.csv', index=False)
    
    # 4. Guardar datos financieros
    logger.info("Step 4: Saving financial data")
    save_data_to_files(financial_data)
    
    # 5. Scraping de datos ESG (opcional)
    logger.info("Step 5: Scraping ESG data")
    company_names = companies_df['company_name'].tolist()
    
    # Scraping de Sustainalytics
    sustainalytics_data = scrape_esg_data_sustainalytics(company_names[:5])  # Solo primeras 5 para prueba
    if not sustainalytics_data.empty:
        sustainalytics_data.to_csv('data/raw/sustainalytics_esg_data.csv', index=False)
    
    # Scraping de informes de sostenibilidad
    sustainability_data = scrape_company_sustainability_reports(company_names[:5])  # Solo primeras 5 para prueba
    if not sustainability_data.empty:
        sustainability_data.to_csv('data/raw/sustainability_reports_data.csv', index=False)
    
    # 6. Crear resumen de datos descargados
    logger.info("Step 6: Creating data summary")
    create_data_summary(financial_data, financial_metrics)
    
    logger.info("Data download completed successfully")


def create_data_summary(financial_data, financial_metrics):
    """
    Crea un resumen de los datos descargados
    
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
