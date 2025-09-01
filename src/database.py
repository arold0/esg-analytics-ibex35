"""
Database models and configuration for ESG Analytics - Phase 3
============================================================
Optimized schema for storing ESG data, financial metrics, and analysis results.
Supports multiple stock indices and historical data tracking.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import os

Base = declarative_base()

# ============================================================================
# CORE ENTITIES
# ============================================================================

class StockIndex(Base):
    """Stock market indices (IBEX35, FTSE100, DAX30, etc.)"""
    __tablename__ = "stock_indices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String(10), unique=True, nullable=False, index=True)  # IBEX35, FTSE100
    name = Column(String(100), nullable=False)  # "IBEX 35", "FTSE 100"
    country = Column(String(50), nullable=False, index=True)  # Spain, UK, Germany
    currency = Column(String(3), nullable=False)  # EUR, GBP, USD
    description = Column(Text)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    companies = relationship("Company", back_populates="stock_index", cascade="all, delete-orphan")

class Company(Base):
    """Company information with multi-index support"""
    __tablename__ = "companies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20), nullable=False, index=True)  # Ticker symbol
    name = Column(String(255), nullable=False)
    sector = Column(String(100), nullable=False, index=True)
    industry = Column(String(150))
    market_cap = Column(Float)
    employees = Column(Integer)
    founded_year = Column(Integer)
    headquarters = Column(String(100))
    website = Column(String(255))
    
    # Foreign Keys
    stock_index_id = Column(UUID(as_uuid=True), ForeignKey("stock_indices.id"), nullable=False)
    
    # Metadata
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stock_index = relationship("StockIndex", back_populates="companies")
    esg_scores = relationship("ESGScore", back_populates="company", cascade="all, delete-orphan")
    financial_data = relationship("FinancialData", back_populates="company", cascade="all, delete-orphan")
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_company_ticker_index', 'ticker', 'stock_index_id'),
        Index('idx_company_sector_active', 'sector', 'is_active'),
    )

# ============================================================================
# ESG DATA
# ============================================================================

class ESGScore(Base):
    """ESG scores with historical tracking"""
    __tablename__ = "esg_scores"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    
    # ESG Scores (0-100 scale)
    esg_score = Column(Float)
    environmental_score = Column(Float)
    social_score = Column(Float)
    governance_score = Column(Float)
    
    # ESG Ratings (A+ to D-)
    esg_rating = Column(String(5))
    environmental_rating = Column(String(5))
    social_rating = Column(String(5))
    governance_rating = Column(String(5))
    
    # Detailed ESG Metrics
    carbon_emissions = Column(Float)  # CO2 equivalent tons
    energy_efficiency = Column(Float)  # Energy use per revenue
    water_usage = Column(Float)  # Water consumption
    waste_management = Column(Float)  # Waste recycling rate
    employee_satisfaction = Column(Float)  # Employee satisfaction score
    diversity_ratio = Column(Float)  # Gender/ethnic diversity
    board_independence = Column(Float)  # Independent board members %
    executive_compensation = Column(Float)  # CEO pay ratio
    
    # Data Quality & Metadata
    data_source = Column(String(50), default="yahoo_finance", index=True)
    data_quality_score = Column(Float)  # 0-1 confidence score
    reporting_period = Column(String(20))  # Q1 2024, FY 2023
    collection_date = Column(DateTime, default=datetime.utcnow, index=True)
    is_latest = Column(Boolean, default=True, index=True)  # Flag for latest record
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="esg_scores")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_esg_company_latest', 'company_id', 'is_latest'),
        Index('idx_esg_collection_date', 'collection_date'),
        Index('idx_esg_data_source', 'data_source', 'collection_date'),
    )

# ============================================================================
# FINANCIAL DATA
# ============================================================================

class FinancialData(Base):
    """Financial metrics with historical tracking"""
    __tablename__ = "financial_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    
    # Financial Performance
    revenue = Column(Float)  # Total revenue
    net_income = Column(Float)  # Net profit
    total_assets = Column(Float)  # Balance sheet assets
    total_equity = Column(Float)  # Shareholders equity
    total_debt = Column(Float)  # Total debt
    
    # Financial Ratios
    profit_margin = Column(Float)  # Net margin %
    roe = Column(Float)  # Return on Equity
    roa = Column(Float)  # Return on Assets
    debt_to_equity = Column(Float)  # Debt/Equity ratio
    current_ratio = Column(Float)  # Current assets/liabilities
    quick_ratio = Column(Float)  # Quick assets/liabilities
    
    # Market Metrics
    stock_price = Column(Float)  # Current stock price
    market_cap = Column(Float)  # Market capitalization
    price_to_earnings = Column(Float)  # P/E ratio
    price_to_book = Column(Float)  # P/B ratio
    price_to_sales = Column(Float)  # P/S ratio
    dividend_yield = Column(Float)  # Dividend yield %
    
    # Risk & Performance
    beta = Column(Float)  # Stock beta
    volatility = Column(Float)  # Price volatility
    sharpe_ratio = Column(Float)  # Risk-adjusted return
    returns_1y = Column(Float)  # 1-year return %
    returns_3y = Column(Float)  # 3-year return %
    returns_5y = Column(Float)  # 5-year return %
    max_drawdown = Column(Float)  # Maximum drawdown %
    
    # Volume & Trading
    avg_volume = Column(Float)  # Average trading volume
    volume_ratio = Column(Float)  # Current vs average volume
    
    # Data Quality & Metadata
    reporting_period = Column(String(20), index=True)  # Q1 2024, FY 2023
    fiscal_year = Column(Integer, index=True)  # 2024, 2023
    quarter = Column(Integer)  # 1, 2, 3, 4
    data_source = Column(String(50), default="yahoo_finance", index=True)
    collection_date = Column(DateTime, default=datetime.utcnow, index=True)
    is_latest = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="financial_data")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_financial_company_latest', 'company_id', 'is_latest'),
        Index('idx_financial_period', 'fiscal_year', 'quarter'),
        Index('idx_financial_collection', 'collection_date'),
    )

# ============================================================================
# ANALYSIS RESULTS
# ============================================================================

class AnalysisResult(Base):
    """Store analysis results and correlations"""
    __tablename__ = "analysis_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_type = Column(String(50), nullable=False, index=True)  # correlation, regression, sector_analysis
    analysis_name = Column(String(100), nullable=False)
    stock_index_id = Column(UUID(as_uuid=True), ForeignKey("stock_indices.id"))
    
    # Statistical Results
    correlation_coefficient = Column(Float)
    p_value = Column(Float)
    r_squared = Column(Float)
    statistical_significance = Column(Boolean, index=True)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    
    # Model Performance (for ML results)
    model_type = Column(String(50))  # linear, ridge, lasso, random_forest
    train_score = Column(Float)
    test_score = Column(Float)
    cross_val_score = Column(Float)
    feature_importance = Column(Text)  # JSON string
    
    # Analysis Metadata
    sample_size = Column(Integer)
    analysis_period_start = Column(DateTime)
    analysis_period_end = Column(DateTime)
    detailed_results = Column(Text)  # JSON string for complex results
    
    # Execution Info
    execution_time = Column(Float)  # Seconds
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_by = Column(String(100))  # Script or user identifier
    
    # Relationships
    stock_index = relationship("StockIndex")
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_type_date', 'analysis_type', 'created_at'),
        Index('idx_analysis_significance', 'statistical_significance', 'analysis_type'),
    )

# ============================================================================
# DATA COLLECTION TRACKING
# ============================================================================

class DataCollectionLog(Base):
    """Track data collection runs and status"""
    __tablename__ = "data_collection_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_index_id = Column(UUID(as_uuid=True), ForeignKey("stock_indices.id"))
    
    # Collection Details
    collection_type = Column(String(50), nullable=False)  # full, incremental, esg_only, financial_only
    status = Column(String(20), nullable=False, index=True)  # running, completed, failed, partial
    companies_total = Column(Integer)
    companies_successful = Column(Integer)
    companies_failed = Column(Integer)
    success_rate = Column(Float)  # Percentage
    
    # Timing
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    
    # Error Tracking
    error_message = Column(Text)
    failed_companies = Column(Text)  # JSON list of failed tickers
    
    # Data Quality
    data_quality_issues = Column(Text)  # JSON list of issues
    records_created = Column(Integer)
    records_updated = Column(Integer)
    
    # Relationships
    stock_index = relationship("StockIndex")
    
    # Indexes
    __table_args__ = (
        Index('idx_collection_status_date', 'status', 'started_at'),
        Index('idx_collection_index_date', 'stock_index_id', 'started_at'),
    )

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

class DatabaseConfig:
    """Database configuration and connection management"""
    
    @staticmethod
    def get_database_url():
        """Get database URL from environment or default to SQLite"""
        return os.getenv(
            'DATABASE_URL',
            'sqlite:///./esg_analytics.db'
        )
    
    @staticmethod
    def create_engine_and_session():
        """Create database engine and session"""
        database_url = DatabaseConfig.get_database_url()
        
        # Configure engine based on database type
        if database_url.startswith('sqlite'):
            engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                echo=False
            )
        else:
            engine = create_engine(database_url, echo=False)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return engine, SessionLocal
    
    @staticmethod
    def create_tables(engine):
        """Create all tables"""
        Base.metadata.create_all(bind=engine)
    
    @staticmethod
    def get_session():
        """Get database session (dependency injection for FastAPI)"""
        engine, SessionLocal = DatabaseConfig.create_engine_and_session()
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def init_database():
    """Initialize database with tables and default data"""
    engine, SessionLocal = DatabaseConfig.create_engine_and_session()
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Add default stock indices
    session = SessionLocal()
    try:
        # Check if IBEX35 exists
        ibex35 = session.query(StockIndex).filter(StockIndex.code == "IBEX35").first()
        if not ibex35:
            indices = [
                StockIndex(
                    code="IBEX35",
                    name="IBEX 35",
                    country="Spain",
                    currency="EUR",
                    description="Spanish stock market index"
                ),
                StockIndex(
                    code="FTSE100",
                    name="FTSE 100",
                    country="United Kingdom",
                    currency="GBP",
                    description="UK stock market index",
                    is_active=False  # Planned for future
                ),
                StockIndex(
                    code="DAX30",
                    name="DAX 30",
                    country="Germany",
                    currency="EUR",
                    description="German stock market index",
                    is_active=False  # Planned for future
                )
            ]
            session.add_all(indices)
            session.commit()
            print("✅ Default stock indices created")
        else:
            print("✅ Database already initialized")
            
    except Exception as e:
        session.rollback()
        print(f"❌ Error initializing database: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("🗄️ Database schema ready for ESG Analytics Phase 3")
