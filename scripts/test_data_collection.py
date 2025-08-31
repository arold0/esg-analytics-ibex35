#!/usr/bin/env python3
"""
ESG Analytics IBEX35 - Test Data Collection Script
==================================================

This script tests the data collection functionality without web scraping.
It focuses on financial data download to verify the system works correctly.
"""

import sys
import os
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_collection():
    """
    Función de prueba para verificar la recolección de datos
    """
    logger.info("Starting test data collection")
    
    # Crear directorios necesarios
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. Obtener lista de empresas IBEX35
    logger.info("Step 1: Getting IBEX35 companies list")
    companies_df = get_ibex35_companies()
    companies_df.to_csv('data/raw/ibex35_companies.csv', index=False)
    logger.info(f"✅ Loaded {len(companies_df)} companies")
    
    # 2. Probar con solo 5 empresas para verificar funcionamiento
    test_tickers = ['SAN.MC', 'BBVA.MC', 'TEF.MC', 'IBE.MC', 'REP.MC']
    logger.info(f"Step 2: Testing with {len(test_tickers)} companies")
    
    # 3. Descargar datos financieros (solo 2024 para prueba rápida)
    logger.info("Step 3: Downloading financial data")
    financial_data = download_financial_data(
        test_tickers, 
        start_date='2024-01-01', 
        end_date='2024-12-31',
        delay=0.5  # Delay más corto para pruebas
    )
    
    # 4. Calcular métricas financieras
    logger.info("Step 4: Calculating financial metrics")
    financial_metrics = calculate_financial_metrics(financial_data)
    financial_metrics.to_csv('data/raw/test_financial_metrics.csv', index=False)
    
    # 5. Guardar datos
    logger.info("Step 5: Saving data")
    save_data_to_files(financial_data, 'data/raw/test_data')
    
    # 6. Mostrar resumen
    logger.info("Step 6: Creating summary")
    companies_with_data = len([k for k, v in financial_data.items() if not v['prices'].empty])
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    print(f"✅ Companies tested: {len(test_tickers)}")
    print(f"✅ Companies with data: {companies_with_data}")
    print(f"✅ Data period: 2024-01-01 to 2024-12-31")
    print(f"✅ Files saved in: data/raw/test_data/")
    print(f"✅ Metrics saved: data/raw/test_financial_metrics.csv")
    
    if companies_with_data > 0:
        print("\n📈 Sample Financial Metrics:")
        print(financial_metrics[['ticker', 'current_price', 'volatility_30d']].to_string(index=False))
    
    print("\n🎉 Test completed successfully!")
    return True


def main():
    """Función principal del script de prueba"""
    try:
        success = test_data_collection()
        if success:
            logger.info("✅ All tests passed!")
            return 0
        else:
            logger.error("❌ Some tests failed!")
            return 1
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
