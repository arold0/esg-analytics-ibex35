"""
ESG Analytics IBEX35 - Visualization Module
==========================================

This module creates comprehensive visualizations for ESG and financial data analysis.
It includes interactive plots, dashboards, and static charts for reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Optional, Tuple, Any
import yaml
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure Plotly for offline use (commented out for script execution)
# pyo.init_notebook_mode(connected=True)


class ESGVisualizer:
    """
    Clase principal para crear visualizaciones ESG y financieras
    """
    
    def __init__(self, config_path: str = None):
        """
        Inicializa el visualizador ESG
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self.load_config(config_path)
        self.data = None
        self.analysis_results = None
        self.output_dir = "reports/figures"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("ESGVisualizer initialized")
    
    def load_config(self, config_path: str = None) -> Dict:
        """Carga la configuración del proyecto"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def load_data(self, data_path: str = "data/processed/ibex35_processed_data.csv") -> pd.DataFrame:
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
    
    def load_analysis_results(self, results_path: str = "data/processed/analysis_results.yaml") -> Dict:
        """
        Carga los resultados del análisis estadístico
        
        Args:
            results_path: Ruta al archivo de resultados
            
        Returns:
            Dict: Resultados del análisis
        """
        try:
            with open(results_path, 'r') as f:
                self.analysis_results = yaml.safe_load(f)
            logger.info("Analysis results loaded successfully")
            return self.analysis_results
        except FileNotFoundError:
            logger.warning(f"Analysis results not found: {results_path}")
            return {}
        except Exception as e:
            logger.warning(f"Could not load analysis results: {e}")
            return {}
    
    def create_correlation_heatmap(self, save_plot: bool = True) -> go.Figure:
        """
        Crea un mapa de calor de correlaciones ESG vs Financieras
        
        Args:
            save_plot: Si guardar el gráfico
            
        Returns:
            go.Figure: Gráfico de Plotly
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Creating correlation heatmap")
        
        # Identificar columnas ESG y financieras
        esg_columns = [col for col in self.data.columns if any(keyword in col.lower() 
                    for keyword in ['esg', 'environmental', 'social', 'governance', 'e_score', 's_score', 'g_score'])]
        
        financial_columns = [col for col in self.data.columns if any(keyword in col.lower() 
                           for keyword in ['roe', 'roa', 'volatility', 'sharpe', 'returns', 'price', 'beta'])]
        
        # Calcular correlaciones
        correlation_data = self.data[esg_columns + financial_columns].corr()
        
        # Crear mapa de calor
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(correlation_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlación ESG vs Métricas Financieras - IBEX35",
            xaxis_title="Métricas Financieras",
            yaxis_title="Métricas ESG",
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        if save_plot:
            fig.write_html(f"{self.output_dir}/correlation_heatmap.html")
            # fig.write_image(f"{self.output_dir}/correlation_heatmap.png")  # Commented out for now
        
        return fig
    
    def create_esg_distribution_plots(self, save_plot: bool = True) -> go.Figure:
        """
        Crea gráficos de distribución de métricas ESG
        
        Args:
            save_plot: Si guardar el gráfico
            
        Returns:
            go.Figure: Gráfico de Plotly
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Creating ESG distribution plots")
        
        # Métricas ESG
        esg_metrics = ['E_Score', 'S_Score', 'G_Score', 'ESG_Total']
        available_metrics = [col for col in esg_metrics if col in self.data.columns]
        
        if not available_metrics:
            logger.warning("No ESG metrics found in data")
            return go.Figure()
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=available_metrics,
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(available_metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=self.data[metric].dropna(),
                    name=metric,
                    nbinsx=20,
                    marker_color=colors[i],
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Distribución de Métricas ESG - IBEX35",
            showlegend=False,
            height=600,
            font=dict(size=12)
        )
        
        if save_plot:
            fig.write_html(f"{self.output_dir}/esg_distributions.html")
            # fig.write_image(f"{self.output_dir}/esg_distributions.png")  # Commented out for now
        
        return fig
    
    def create_sector_comparison(self, save_plot: bool = True) -> go.Figure:
        """
        Crea gráfico comparativo por sectores
        
        Args:
            save_plot: Si guardar el gráfico
            
        Returns:
            go.Figure: Gráfico de Plotly
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Creating sector comparison plot")
        
        # Cargar información de sectores
        companies_data = self.config['ibex35']['companies']
        sector_mapping = {company['symbol']: company['sector'] for company in companies_data}
        
        # Agregar sector al DataFrame
        self.data['sector'] = self.data['ticker'].map(sector_mapping)
        
        # Calcular promedios por sector
        sector_stats = self.data.groupby('sector').agg({
            'E_Score': 'mean',
            'S_Score': 'mean', 
            'G_Score': 'mean',
            'ESG_Total': 'mean'
        }).reset_index()
        
        # Crear gráfico de barras
        fig = go.Figure()
        
        metrics = ['E_Score', 'S_Score', 'G_Score', 'ESG_Total']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            if metric in sector_stats.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=sector_stats['sector'],
                    y=sector_stats[metric],
                    marker_color=colors[i],
                    opacity=0.8
                ))
        
        fig.update_layout(
            title="Promedio de Métricas ESG por Sector - IBEX35",
            xaxis_title="Sector",
            yaxis_title="Score Promedio",
            barmode='group',
            height=500,
            font=dict(size=12)
        )
        
        if save_plot:
            fig.write_html(f"{self.output_dir}/sector_comparison.html")
            # fig.write_image(f"{self.output_dir}/sector_comparison.png")  # Commented out for now
        
        return fig
    
    def create_scatter_plots(self, save_plot: bool = True) -> go.Figure:
        """
        Crea gráficos de dispersión ESG vs Financieras
        
        Args:
            save_plot: Si guardar el gráfico
            
        Returns:
            go.Figure: Gráfico de Plotly
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Creating ESG vs Financial scatter plots")
        
        # Pares de variables para scatter plots
        scatter_pairs = [
            ('ESG_Total', 'roe'),
            ('ESG_Total', 'roa'),
            ('ESG_Total', 'volatility_30d'),
            ('ESG_Total', 'sharpe_ratio')
        ]
        
        # Filtrar pares disponibles
        available_pairs = []
        for esg_var, fin_var in scatter_pairs:
            if esg_var in self.data.columns and fin_var in self.data.columns:
                available_pairs.append((esg_var, fin_var))
        
        if not available_pairs:
            logger.warning("No suitable variable pairs found for scatter plots")
            return go.Figure()
        
        # Crear subplots
        n_plots = len(available_pairs)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{esg} vs {fin}" for esg, fin in available_pairs],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (esg_var, fin_var) in enumerate(available_pairs):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Datos válidos
            valid_data = self.data[[esg_var, fin_var, 'ticker']].dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=valid_data[esg_var],
                    y=valid_data[fin_var],
                    mode='markers+text',
                    text=valid_data['ticker'],
                    textposition="top center",
                    name=f"{esg_var} vs {fin_var}",
                    marker=dict(
                        size=8,
                        color=colors[i],
                        opacity=0.7
                    )
                ),
                row=row, col=col
            )
            
            # Agregar línea de tendencia
            z = np.polyfit(valid_data[esg_var], valid_data[fin_var], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=valid_data[esg_var],
                    y=p(valid_data[esg_var]),
                    mode='lines',
                    name=f'Trend {esg_var} vs {fin_var}',
                    line=dict(color=colors[i], dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="ESG vs Métricas Financieras - IBEX35",
            height=600,
            font=dict(size=12)
        )
        
        if save_plot:
            fig.write_html(f"{self.output_dir}/esg_financial_scatter.html")
            # fig.write_image(f"{self.output_dir}/esg_financial_scatter.png")  # Commented out for now
        
        return fig
    
    def create_top_performers_chart(self, save_plot: bool = True) -> go.Figure:
        """
        Crea gráfico de mejores y peores performers ESG
        
        Args:
            save_plot: Si guardar el gráfico
            
        Returns:
            go.Figure: Gráfico de Plotly
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Creating top performers chart")
        
        if 'ESG_Total' not in self.data.columns:
            logger.warning("ESG_Total not found in data")
            return go.Figure()
        
        # Top 10 mejores y peores performers
        top_10 = self.data.nlargest(10, 'ESG_Total')[['ticker', 'ESG_Total']]
        bottom_10 = self.data.nsmallest(10, 'ESG_Total')[['ticker', 'ESG_Total']]
        
        # Crear gráfico
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Top 10 ESG Performers", "Bottom 10 ESG Performers"],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Top performers
        fig.add_trace(
            go.Bar(
                x=top_10['ticker'],
                y=top_10['ESG_Total'],
                name="Top 10",
                marker_color='green',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Bottom performers
        fig.add_trace(
            go.Bar(
                x=bottom_10['ticker'],
                y=bottom_10['ESG_Total'],
                name="Bottom 10",
                marker_color='red',
                opacity=0.8
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Top y Bottom ESG Performers - IBEX35",
            height=500,
            font=dict(size=12)
        )
        
        if save_plot:
            fig.write_html(f"{self.output_dir}/top_performers.html")
            # fig.write_image(f"{self.output_dir}/top_performers.png")  # Commented out for now
        
        return fig
    
    def create_comprehensive_dashboard(self, save_plot: bool = True) -> go.Figure:
        """
        Crea un dashboard comprehensivo con múltiples visualizaciones
        
        Args:
            save_plot: Si guardar el gráfico
            
        Returns:
            go.Figure: Dashboard de Plotly
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Creating comprehensive dashboard")
        
        # Crear dashboard con múltiples gráficos
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Distribución ESG Total",
                "ESG vs ROE",
                "Promedio por Sector",
                "ESG vs Volatilidad",
                "Top Performers",
                "Correlación ESG-Financiera"
            ],
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ]
        )
        
        # 1. Distribución ESG Total
        if 'ESG_Total' in self.data.columns:
            fig.add_trace(
                go.Histogram(
                    x=self.data['ESG_Total'].dropna(),
                    name="ESG Total",
                    marker_color='blue',
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # 2. ESG vs ROE
        if 'ESG_Total' in self.data.columns and 'roe' in self.data.columns:
            valid_data = self.data[['ESG_Total', 'roe', 'ticker']].dropna()
            fig.add_trace(
                go.Scatter(
                    x=valid_data['ESG_Total'],
                    y=valid_data['roe'],
                    mode='markers+text',
                    text=valid_data['ticker'],
                    textposition="top center",
                    name="ESG vs ROE",
                    marker=dict(size=8, color='red', opacity=0.7)
                ),
                row=1, col=2
            )
        
        # 3. Promedio por Sector
        if 'sector' in self.data.columns and 'ESG_Total' in self.data.columns:
            sector_avg = self.data.groupby('sector')['ESG_Total'].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=sector_avg['sector'],
                    y=sector_avg['ESG_Total'],
                    name="Sector Avg",
                    marker_color='green',
                    opacity=0.8
                ),
                row=2, col=1
            )
        
        # 4. ESG vs Volatilidad
        if 'ESG_Total' in self.data.columns and 'volatility_30d' in self.data.columns:
            valid_data = self.data[['ESG_Total', 'volatility_30d', 'ticker']].dropna()
            fig.add_trace(
                go.Scatter(
                    x=valid_data['ESG_Total'],
                    y=valid_data['volatility_30d'],
                    mode='markers+text',
                    text=valid_data['ticker'],
                    textposition="top center",
                    name="ESG vs Volatility",
                    marker=dict(size=8, color='orange', opacity=0.7)
                ),
                row=2, col=2
            )
        
        # 5. Top Performers
        if 'ESG_Total' in self.data.columns:
            top_5 = self.data.nlargest(5, 'ESG_Total')[['ticker', 'ESG_Total']]
            fig.add_trace(
                go.Bar(
                    x=top_5['ticker'],
                    y=top_5['ESG_Total'],
                    name="Top 5",
                    marker_color='purple',
                    opacity=0.8
                ),
                row=3, col=1
            )
        
        # 6. Correlación simplificada
        if 'ESG_Total' in self.data.columns:
            corr_cols = ['ESG_Total', 'roe', 'roa', 'volatility_30d', 'sharpe_ratio']
            available_cols = [col for col in corr_cols if col in self.data.columns]
            
            if len(available_cols) > 1:
                corr_matrix = self.data[available_cols].corr()
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu_r',
                        zmid=0,
                        name="Correlation"
                    ),
                    row=3, col=2
                )
        
        fig.update_layout(
            title="Dashboard ESG Analytics IBEX35 - Vista General",
            height=1200,
            font=dict(size=10),
            showlegend=False
        )
        
        if save_plot:
            fig.write_html(f"{self.output_dir}/comprehensive_dashboard.html")
            # fig.write_image(f"{self.output_dir}/comprehensive_dashboard.png")  # Commented out for now
        
        return fig
    
    def generate_all_visualizations(self) -> Dict[str, go.Figure]:
        """
        Genera todas las visualizaciones disponibles
        
        Returns:
            Dict: Diccionario con todas las figuras generadas
        """
        logger.info("Generating all visualizations")
        
        figures = {}
        
        try:
            figures['correlation_heatmap'] = self.create_correlation_heatmap()
            figures['esg_distributions'] = self.create_esg_distribution_plots()
            figures['sector_comparison'] = self.create_sector_comparison()
            figures['scatter_plots'] = self.create_scatter_plots()
            figures['top_performers'] = self.create_top_performers_chart()
            figures['comprehensive_dashboard'] = self.create_comprehensive_dashboard()
            
            logger.info(f"Generated {len(figures)} visualizations successfully")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return figures
    
    def create_visualization_report(self) -> str:
        """
        Crea un reporte HTML con todas las visualizaciones
        
        Returns:
            str: Ruta al archivo HTML generado
        """
        logger.info("Creating visualization report")
        
        # Generar todas las visualizaciones
        figures = self.generate_all_visualizations()
        
        # Crear reporte HTML
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ESG Analytics IBEX35 - Reporte de Visualizaciones</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin: 30px 0; }
                .chart { margin: 20px 0; text-align: center; }
                iframe { border: none; width: 100%; height: 600px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 ESG Analytics IBEX35</h1>
                <h2>Reporte de Visualizaciones</h2>
                <p>Análisis de factores ESG y su correlación con métricas financieras</p>
            </div>
        """
        
        # Agregar cada visualización al reporte
        for name, fig in figures.items():
            html_content += f"""
            <div class="section">
                <h3>{name.replace('_', ' ').title()}</h3>
                <div class="chart">
                    <iframe src="{self.output_dir}/{name}.html"></iframe>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Guardar reporte
        report_path = f"{self.output_dir}/visualization_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Visualization report created: {report_path}")
        return report_path


def main():
    """Función principal para generar visualizaciones"""
    logger.info("Starting ESG Analytics visualization generation")
    
    # Inicializar visualizador
    visualizer = ESGVisualizer()
    
    # Cargar datos
    visualizer.load_data()
    
    # Cargar resultados de análisis (opcional)
    visualizer.load_analysis_results()
    
    # Generar todas las visualizaciones
    figures = visualizer.generate_all_visualizations()
    
    # Crear reporte
    report_path = visualizer.create_visualization_report()
    
    print("\n" + "="*60)
    print("📊 ESG ANALYTICS - VISUALIZATIONS GENERATED")
    print("="*60)
    print(f"✅ Visualizations created: {len(figures)}")
    print(f"📁 Output directory: {visualizer.output_dir}")
    print(f"📄 Report file: {report_path}")
    
    print("\n📈 Generated Charts:")
    for name in figures.keys():
        print(f"   • {name.replace('_', ' ').title()}")
    
    print("\n🎉 Visualization generation completed successfully!")
    print("="*60)
    
    logger.info("Visualization generation completed successfully")


if __name__ == "__main__":
    main()
