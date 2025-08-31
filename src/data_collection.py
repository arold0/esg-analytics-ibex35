"""
ESG Analytics IBEX35 - Data Collection Module
=============================================

This module handles the collection of financial and ESG data for IBEX35 companies.
It includes functions for downloading financial data from Yahoo Finance and
collecting ESG data from various sources.
"""

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """Load configuration from config.yaml file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_ibex35_companies() -> pd.DataFrame:
    """
    Obtiene la lista actualizada de empresas del IBEX 35
    
    Returns:
        pd.DataFrame: DataFrame con tickers y nombres de empresas
    """
    # Lista manual actualizada de empresas IBEX 35
    ibex35_tickers = {
        'ACS.MC': 'ACS Actividades de Construccion y Servicios SA',
        'AENA.MC': 'Aena SME SA',
        'AMS.MC': 'Amadeus IT Group SA',
        'ANA.MC': 'Acciona SA',
        'BBVA.MC': 'Banco Bilbao Vizcaya Argentaria SA',
        'BKT.MC': 'Bankinter SA',
        'CAB.MC': 'Caixabank SA',
        'CIE.MC': 'Cie Automotive SA',
        'COL.MC': 'Inmobiliaria Colonial Socimi SA',
        'ELE.MC': 'Endesa SA',
        'ENG.MC': 'Enagas SA',
        'FDR.MC': 'Fluidra SA',
        'FER.MC': 'Ferrovial SA',
        'GRF.MC': 'Grifols SA',
        'IAG.MC': 'International Consolidated Airlines Group SA',
        'IBE.MC': 'Iberdrola SA',
        'IDR.MC': 'Indra Sistemas SA',
        'ITX.MC': 'Industria de Diseno Textil SA',
        'LOG.MC': 'Logista SA',
        'MAP.MC': 'Mapfre SA',
        'MEL.MC': 'Melia Hotels International SA',
        'MTS.MC': 'ArcelorMittal SA',
        'NTGY.MC': 'Naturgy Energy Group SA',
        'PHM.MC': 'Pharma Mar SA',
        'REE.MC': 'Red Electrica Corp SA',
        'REP.MC': 'Repsol SA',
        'ROL.MC': 'Rolls-Royce SMR',
        'SAN.MC': 'Banco Santander SA',
        'SAB.MC': 'Banco Sabadell SA',
        'SCYR.MC': 'Sacyr SA',
        'SLR.MC': 'Solaria Energia y Medio Ambiente SA',
        'TEF.MC': 'Telefonica SA',
        'UNI.MC': 'Unicaja Banco SA',
        'VIS.MC': 'Viscofan SA',
        'VLA.MC': 'Inmobiliaria del Sur SA'
    }
    
    df = pd.DataFrame(list(ibex35_tickers.items()), 
                     columns=['ticker', 'company_name'])
    
    logger.info(f"Loaded {len(df)} IBEX35 companies")
    return df


def download_financial_data(tickers: List[str], 
                           start_date: str = '2019-01-01', 
                           end_date: str = '2024-12-31',
                           delay: float = 1.0) -> Dict:
    """
    Descarga datos financieros históricos de Yahoo Finance
    
    Args:
        tickers: Lista de tickers de empresas
        start_date: Fecha de inicio para datos históricos
        end_date: Fecha de fin para datos históricos
        delay: Delay entre requests para evitar rate limiting
    
    Returns:
        Dict: Diccionario con datos financieros por empresa
    """
    config = load_config()
    data = {}
    
    logger.info(f"Starting financial data download for {len(tickers)} companies")
    
    for i, ticker in enumerate(tickers):
        try:
            logger.info(f"Downloading data for {ticker} ({i+1}/{len(tickers)})")
            
            stock = yf.Ticker(ticker)
            
            # Datos históricos de precios
            hist_data = stock.history(start=start_date, end=end_date)
            
            # Información financiera básica
            info = stock.info
            
            # Estados financieros (últimos 4 años)
            try:
                financials = stock.financials
            except:
                financials = pd.DataFrame()
            
            # Balance sheet
            try:
                balance_sheet = stock.balance_sheet
            except:
                balance_sheet = pd.DataFrame()
            
            # Cash flow
            try:
                cash_flow = stock.cashflow
            except:
                cash_flow = pd.DataFrame()
            
            data[ticker] = {
                'prices': hist_data,
                'info': info,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
            
            logger.info(f"Successfully downloaded data for {ticker}")
            
            # Rate limiting
            if i < len(tickers) - 1:  # Don't delay after last request
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            data[ticker] = {
                'prices': pd.DataFrame(),
                'info': {},
                'financials': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame()
            }
    
    logger.info(f"Financial data download completed. {len([k for k, v in data.items() if not v['prices'].empty])} companies with data")
    return data


def collect_esg_data() -> Dict:
    """
    Recolecta datos ESG de múltiples fuentes
    
    Returns:
        Dict: Estructura para almacenar datos ESG
    """
    # Estructura para almacenar datos ESG
    esg_data = {
        'environmental': {},
        'social': {},
        'governance': {}
    }
    
    logger.info("Starting ESG data collection")
    
    # Fuentes de datos ESG gratuitas
    # 1. Informes de sostenibilidad corporativa (web scraping)
    # 2. Datos públicos de emisiones
    # 3. Información de diversidad y gobierno corporativo
    
    # TODO: Implementar scraping de datos ESG
    # Por ahora retornamos estructura vacía
    logger.warning("ESG data collection not yet implemented")
    
    return esg_data


def scrape_sustainability_reports(companies: List[str]) -> pd.DataFrame:
    """
    Web scraping de informes de sostenibilidad
    
    Args:
        companies: Lista de nombres de empresas
    
    Returns:
        pd.DataFrame: Datos ESG extraídos
    """
    logger.info(f"Starting sustainability report scraping for {len(companies)} companies")
    
    # TODO: Implementar scraping de informes ESG
    # Por ahora retornamos DataFrame vacío
    logger.warning("Sustainability report scraping not yet implemented")
    
    return pd.DataFrame()


def get_company_sector_mapping() -> Dict[str, str]:
    """
    Obtiene el mapeo de empresas por sector
    
    Returns:
        Dict: Mapeo de ticker a sector
    """
    config = load_config()
    companies = config['ibex35']['companies']
    
    sector_mapping = {}
    for company in companies:
        sector_mapping[company['symbol']] = company['sector']
    
    return sector_mapping


def calculate_financial_metrics(financial_data: Dict) -> pd.DataFrame:
    """
    Calcula métricas financieras adicionales
    
    Args:
        financial_data: Datos financieros descargados
    
    Returns:
        pd.DataFrame: Métricas financieras calculadas
    """
    metrics = []
    
    for ticker, data in financial_data.items():
        if data['prices'].empty:
            continue
            
        prices = data['prices']
        info = data['info']
        
        # Métricas básicas
        current_price = prices['Close'].iloc[-1] if len(prices) > 0 else None
        price_change_1y = ((current_price / prices['Close'].iloc[-252]) - 1) * 100 if len(prices) >= 252 else None
        
        # Volatilidad (30 días)
        returns = prices['Close'].pct_change().dropna()
        volatility_30d = returns.tail(30).std() * (252 ** 0.5) * 100 if len(returns) >= 30 else None
        
        # Métricas de la información de la empresa
        market_cap = info.get('marketCap', None)
        pe_ratio = info.get('trailingPE', None)
        pb_ratio = info.get('priceToBook', None)
        dividend_yield = info.get('dividendYield', None)
        
        metrics.append({
            'ticker': ticker,
            'current_price': current_price,
            'price_change_1y': price_change_1y,
            'volatility_30d': volatility_30d,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'dividend_yield': dividend_yield
        })
    
    return pd.DataFrame(metrics)


def save_data_to_files(data: Dict, output_dir: str = "data/raw") -> None:
    """
    Guarda los datos descargados en archivos
    
    Args:
        data: Datos a guardar
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar datos financieros
    for ticker, company_data in data.items():
        if not company_data['prices'].empty:
            # Guardar precios históricos
            prices_file = os.path.join(output_dir, f"{ticker}_prices.csv")
            company_data['prices'].to_csv(prices_file)
            
            # Guardar información de la empresa
            info_file = os.path.join(output_dir, f"{ticker}_info.csv")
            pd.DataFrame([company_data['info']]).to_csv(info_file, index=False)
            
            # Guardar estados financieros
            if not company_data['financials'].empty:
                financials_file = os.path.join(output_dir, f"{ticker}_financials.csv")
                company_data['financials'].to_csv(financials_file)
    
    logger.info(f"Data saved to {output_dir}")


def main():
    """Función principal para ejecutar la recolección de datos"""
    logger.info("Starting ESG Analytics IBEX35 data collection")
    
    # Obtener lista de empresas
    companies_df = get_ibex35_companies()
    tickers = companies_df['ticker'].tolist()
    
    # Descargar datos financieros
    financial_data = download_financial_data(tickers)
    
    # Calcular métricas financieras
    financial_metrics = calculate_financial_metrics(financial_data)
    
    # Guardar datos
    save_data_to_files(financial_data)
    
    # Guardar métricas calculadas
    financial_metrics.to_csv("data/raw/financial_metrics.csv", index=False)
    
    logger.info("Data collection completed successfully")


if __name__ == "__main__":
    main()
