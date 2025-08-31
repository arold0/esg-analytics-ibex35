"""
ESG Analytics IBEX35 - Data Processing Module
============================================

This module handles data cleaning, preprocessing, and preparation for analysis.
It includes functions for cleaning financial data, creating ESG composite scores,
and handling missing values.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESGDataProcessor:
    """
    Clase para procesar y limpiar datos ESG y financieros
    """
    
    def __init__(self, config_path: str = None):
        """
        Inicializa el procesador de datos
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.config = self.load_config(config_path)
        logger.info("ESGDataProcessor initialized")
    
    def load_config(self, config_path: str = None) -> Dict:
        """Carga la configuración del proyecto"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def clean_financial_data(self, financial_data: Dict) -> Dict:
        """
        Limpia y normaliza datos financieros
        
        Args:
            financial_data: Diccionario con datos financieros por empresa
            
        Returns:
            Dict: Datos financieros limpios
        """
        logger.info("Starting financial data cleaning")
        cleaned_data = {}
        
        for ticker, data in financial_data.items():
            try:
                # Procesar datos históricos
                prices = data['prices'].copy()
                
                if prices.empty:
                    logger.warning(f"No price data available for {ticker}")
                    continue
                
                # Calcular métricas financieras adicionales
                prices['Returns'] = prices['Close'].pct_change()
                prices['Volatility'] = prices['Returns'].rolling(30).std()
                prices['ROE'] = self.calculate_roe(data)
                prices['ROA'] = self.calculate_roa(data)
                
                # Calcular métricas adicionales
                prices['Sharpe_Ratio'] = self.calculate_sharpe_ratio(prices)
                prices['Max_Drawdown'] = self.calculate_max_drawdown(prices)
                prices['Beta'] = self.calculate_beta(prices)
                
                # Manejar valores faltantes
                prices_cleaned = self.handle_missing_values(prices)
                
                cleaned_data[ticker] = prices_cleaned
                logger.info(f"Cleaned financial data for {ticker}")
                
            except Exception as e:
                logger.error(f"Error cleaning data for {ticker}: {e}")
                continue
        
        logger.info(f"Financial data cleaning completed. {len(cleaned_data)} companies processed")
        return cleaned_data
    
    def create_esg_composite_scores(self, esg_data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea scores compuestos ESG
        
        Args:
            esg_data: DataFrame con datos ESG
            
        Returns:
            pd.DataFrame: Datos ESG con scores compuestos
        """
        if esg_data.empty:
            logger.warning("No ESG data available for composite score creation")
            return pd.DataFrame()
        
        logger.info("Creating ESG composite scores")
        
        # Normalizar métricas ESG (0-100)
        esg_normalized = esg_data.copy()
        
        # Identificar columnas numéricas para normalizar
        numeric_columns = esg_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column not in ['ticker', 'company']:  # Excluir columnas de identificación
                try:
                    # Normalizar a escala 0-100
                    esg_normalized[column] = MinMaxScaler().fit_transform(
                        esg_data[[column]]
                    ) * 100
                except Exception as e:
                    logger.warning(f"Could not normalize column {column}: {e}")
        
        # Crear scores compuestos por categoría ESG
        environmental_cols = [col for col in esg_normalized.columns 
                            if any(keyword in col.lower() for keyword in 
                                  ['environmental', 'carbon', 'energy', 'waste', 'emission'])]
        
        social_cols = [col for col in esg_normalized.columns 
                      if any(keyword in col.lower() for keyword in 
                            ['social', 'diversity', 'employee', 'community', 'labor'])]
        
        governance_cols = [col for col in esg_normalized.columns 
                          if any(keyword in col.lower() for keyword in 
                                ['governance', 'board', 'transparency', 'corruption', 'executive'])]
        
        # Calcular scores compuestos
        if environmental_cols:
            esg_normalized['E_Score'] = esg_normalized[environmental_cols].mean(axis=1)
        else:
            esg_normalized['E_Score'] = 50  # Valor neutral si no hay datos
        
        if social_cols:
            esg_normalized['S_Score'] = esg_normalized[social_cols].mean(axis=1)
        else:
            esg_normalized['S_Score'] = 50  # Valor neutral si no hay datos
        
        if governance_cols:
            esg_normalized['G_Score'] = esg_normalized[governance_cols].mean(axis=1)
        else:
            esg_normalized['G_Score'] = 50  # Valor neutral si no hay datos
        
        # Score ESG total (ponderado)
        esg_normalized['ESG_Total'] = (
            esg_normalized['E_Score'] * 0.33 +
            esg_normalized['S_Score'] * 0.33 +
            esg_normalized['G_Score'] * 0.34
        )
        
        logger.info("ESG composite scores created successfully")
        return esg_normalized
    
    def calculate_roe(self, company_data: Dict) -> float:
        """
        Calcula Return on Equity
        
        Args:
            company_data: Datos de la empresa
            
        Returns:
            float: ROE calculado
        """
        try:
            info = company_data['info']
            roe = info.get('returnOnEquity', 0)
            return roe * 100 if roe is not None else np.nan
        except:
            return np.nan
    
    def calculate_roa(self, company_data: Dict) -> float:
        """
        Calcula Return on Assets
        
        Args:
            company_data: Datos de la empresa
            
        Returns:
            float: ROA calculado
        """
        try:
            info = company_data['info']
            roa = info.get('returnOnAssets', 0)
            return roa * 100 if roa is not None else np.nan
        except:
            return np.nan
    
    def calculate_sharpe_ratio(self, prices: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """
        Calcula el ratio de Sharpe
        
        Args:
            prices: DataFrame con precios
            risk_free_rate: Tasa libre de riesgo (default 2%)
            
        Returns:
            float: Ratio de Sharpe
        """
        try:
            returns = prices['Close'].pct_change().dropna()
            excess_returns = returns - risk_free_rate/252  # Diario
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        except:
            return np.nan
    
    def calculate_max_drawdown(self, prices: pd.DataFrame) -> float:
        """
        Calcula el máximo drawdown
        
        Args:
            prices: DataFrame con precios
            
        Returns:
            float: Máximo drawdown
        """
        try:
            cumulative = prices['Close'].cummax()
            drawdown = (prices['Close'] - cumulative) / cumulative
            return drawdown.min() * 100
        except:
            return np.nan
    
    def calculate_beta(self, prices: pd.DataFrame, market_returns: pd.Series = None) -> float:
        """
        Calcula el beta de la empresa
        
        Args:
            prices: DataFrame con precios
            market_returns: Retornos del mercado (opcional)
            
        Returns:
            float: Beta calculado
        """
        try:
            stock_returns = prices['Close'].pct_change().dropna()
            
            if market_returns is None:
                # Usar un proxy simple del mercado (promedio de retornos)
                return 1.0  # Valor neutral si no hay datos del mercado
            
            # Calcular beta
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            return covariance / market_variance
        except:
            return np.nan
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja valores faltantes en el DataFrame
        
        Args:
            df: DataFrame a procesar
            
        Returns:
            pd.DataFrame: DataFrame con valores faltantes manejados
        """
        logger.info(f"Handling missing values in DataFrame with shape {df.shape}")
        
        # Identificar columnas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            logger.warning("No numeric columns found for missing value handling")
            return df
        
        # Crear copia para no modificar el original
        df_cleaned = df.copy()
        
        # Imputación para variables numéricas
        try:
            df_cleaned[numeric_columns] = self.imputer.fit_transform(df_cleaned[numeric_columns])
            logger.info(f"Missing values imputed for {len(numeric_columns)} numeric columns")
        except Exception as e:
            logger.error(f"Error in missing value imputation: {e}")
            # Fallback: usar forward fill y backward fill
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        return df_cleaned
    
    def create_processed_dataset(self, financial_data: Dict, esg_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Crea un dataset procesado combinando datos financieros y ESG
        
        Args:
            financial_data: Datos financieros limpios
            esg_data: Datos ESG (opcional)
            
        Returns:
            pd.DataFrame: Dataset procesado combinado
        """
        logger.info("Creating processed dataset")
        
        # Procesar datos financieros
        processed_financial = []
        
        for ticker, data in financial_data.items():
            if data.empty:
                continue
            
            # Obtener métricas más recientes
            latest_data = data.iloc[-1]
            
            # Crear registro para la empresa
            company_record = {
                'ticker': ticker,
                'current_price': latest_data.get('Close', np.nan),
                'returns_1y': latest_data.get('Returns', np.nan),
                'volatility_30d': latest_data.get('Volatility', np.nan),
                'roe': latest_data.get('ROE', np.nan),
                'roa': latest_data.get('ROA', np.nan),
                'sharpe_ratio': latest_data.get('Sharpe_Ratio', np.nan),
                'max_drawdown': latest_data.get('Max_Drawdown', np.nan),
                'beta': latest_data.get('Beta', np.nan)
            }
            
            processed_financial.append(company_record)
        
        # Crear DataFrame financiero
        financial_df = pd.DataFrame(processed_financial)
        
        # Combinar con datos ESG si están disponibles
        if esg_data is not None and not esg_data.empty:
            # Merge por ticker
            combined_df = financial_df.merge(esg_data, on='ticker', how='left')
        else:
            combined_df = financial_df
        
        logger.info(f"Processed dataset created with {len(combined_df)} companies")
        return combined_df
    
    def save_processed_data(self, processed_data: pd.DataFrame, output_path: str = "data/processed/") -> None:
        """
        Guarda los datos procesados
        
        Args:
            processed_data: DataFrame procesado
            output_path: Ruta de salida
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Guardar datos procesados
        processed_file = os.path.join(output_path, "ibex35_processed_data.csv")
        processed_data.to_csv(processed_file, index=False)
        
        # Crear resumen de procesamiento
        summary = {
            'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_companies': len(processed_data),
            'columns_processed': list(processed_data.columns),
            'missing_values_summary': processed_data.isnull().sum().to_dict(),
            'output_file': processed_file
        }
        
        summary_file = os.path.join(output_path, "processing_summary.yaml")
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        logger.info(f"Processed data saved to {processed_file}")
        logger.info(f"Processing summary saved to {summary_file}")


def main():
    """Función principal para procesar datos"""
    logger.info("Starting data processing pipeline")
    
    # Inicializar procesador
    processor = ESGDataProcessor()
    
    # Cargar datos financieros (ejemplo)
    # En la práctica, esto vendría de los archivos descargados
    logger.info("Data processing pipeline completed")


if __name__ == "__main__":
    main()
