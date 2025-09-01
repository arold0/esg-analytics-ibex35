"""
ESG Analytics API v2 - Database-powered
======================================
Updated API using SQLAlchemy database instead of CSV files.
Supports multiple stock indices and improved performance.
"""
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# Import database models and config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from database import (
    DatabaseConfig, StockIndex, Company, ESGScore, FinancialData, 
    AnalysisResult, DataCollectionLog
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ESG Analytics API v2",
    description="Advanced ESG Analytics for Stock Market Indices - Database Edition",
    version="2.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CompanyResponse(BaseModel):
    id: str
    ticker: str
    name: str
    sector: str
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    stock_index: str
    is_active: bool
    
    class Config:
        from_attributes = True

class ESGScoreResponse(BaseModel):
    esg_score: Optional[float] = None
    environmental_score: Optional[float] = None
    social_score: Optional[float] = None
    governance_score: Optional[float] = None
    esg_rating: Optional[str] = None
    data_source: str
    reporting_period: Optional[str] = None
    collection_date: datetime
    
    class Config:
        from_attributes = True

class FinancialDataResponse(BaseModel):
    revenue: Optional[float] = None
    profit_margin: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    price_to_earnings: Optional[float] = None
    beta: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    returns_1y: Optional[float] = None
    market_cap: Optional[float] = None
    reporting_period: Optional[str] = None
    fiscal_year: Optional[int] = None
    
    class Config:
        from_attributes = True

class CompanyDetailResponse(BaseModel):
    company: CompanyResponse
    esg_data: Optional[ESGScoreResponse] = None
    financial_data: Optional[FinancialDataResponse] = None

class CorrelationResponse(BaseModel):
    analysis_name: str
    correlation_coefficient: Optional[float] = None
    p_value: Optional[float] = None
    statistical_significance: Optional[bool] = None
    sample_size: Optional[int] = None
    created_at: datetime

class SectorAnalysisResponse(BaseModel):
    sector: str
    company_count: int
    avg_esg_score: Optional[float] = None
    avg_financial_performance: Optional[float] = None
    top_companies: List[str] = []

class StockIndexResponse(BaseModel):
    code: str
    name: str
    country: str
    currency: str
    company_count: int
    is_active: bool
    
    class Config:
        from_attributes = True

# ============================================================================
# DATABASE DEPENDENCY
# ============================================================================

def get_db() -> Session:
    """Database dependency for FastAPI"""
    engine, SessionLocal = DatabaseConfig.create_engine_and_session()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "database": "connected",
        "timestamp": datetime.utcnow()
    }

@app.get("/indices", response_model=List[StockIndexResponse])
async def get_stock_indices(
    active_only: bool = Query(True, description="Return only active indices"),
    db: Session = Depends(get_db)
):
    """Get all stock indices"""
    query = db.query(StockIndex)
    if active_only:
        query = query.filter(StockIndex.is_active == True)
    
    indices = query.all()
    
    # Add company count to each index
    result = []
    for index in indices:
        company_count = db.query(Company).filter(
            Company.stock_index_id == index.id,
            Company.is_active == True
        ).count()
        
        result.append(StockIndexResponse(
            code=index.code,
            name=index.name,
            country=index.country,
            currency=index.currency,
            company_count=company_count,
            is_active=index.is_active
        ))
    
    return result

@app.get("/companies", response_model=List[CompanyResponse])
async def get_companies(
    index_code: str = Query("IBEX35", description="Stock index code"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    limit: int = Query(50, ge=1, le=100, description="Number of companies to return"),
    db: Session = Depends(get_db)
):
    """Get companies from a stock index"""
    
    # Get stock index
    stock_index = db.query(StockIndex).filter(StockIndex.code == index_code).first()
    if not stock_index:
        raise HTTPException(status_code=404, detail=f"Stock index {index_code} not found")
    
    # Build query
    query = db.query(Company).filter(
        Company.stock_index_id == stock_index.id,
        Company.is_active == True
    )
    
    if sector:
        query = query.filter(Company.sector.ilike(f"%{sector}%"))
    
    companies = query.limit(limit).all()
    
    # Format response
    result = []
    for company in companies:
        result.append(CompanyResponse(
            id=str(company.id),
            ticker=company.ticker,
            name=company.name,
            sector=company.sector,
            industry=company.industry,
            market_cap=company.market_cap,
            stock_index=stock_index.code,
            is_active=company.is_active
        ))
    
    logger.info(f"Retrieved {len(result)} companies from {index_code}")
    return result

@app.get("/companies/{ticker}", response_model=CompanyDetailResponse)
async def get_company_detail(
    ticker: str,
    index_code: str = Query("IBEX35", description="Stock index code"),
    db: Session = Depends(get_db)
):
    """Get detailed company information including ESG and financial data"""
    
    # Get stock index
    stock_index = db.query(StockIndex).filter(StockIndex.code == index_code).first()
    if not stock_index:
        raise HTTPException(status_code=404, detail=f"Stock index {index_code} not found")
    
    # Get company
    company = db.query(Company).filter(
        Company.ticker == ticker.upper(),
        Company.stock_index_id == stock_index.id
    ).first()
    
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found in {index_code}")
    
    # Get latest ESG data
    esg_data = db.query(ESGScore).filter(
        ESGScore.company_id == company.id,
        ESGScore.is_latest == True
    ).first()
    
    # Get latest financial data
    financial_data = db.query(FinancialData).filter(
        FinancialData.company_id == company.id,
        FinancialData.is_latest == True
    ).first()
    
    # Format response
    company_response = CompanyResponse(
        id=str(company.id),
        ticker=company.ticker,
        name=company.name,
        sector=company.sector,
        industry=company.industry,
        market_cap=company.market_cap,
        stock_index=stock_index.code,
        is_active=company.is_active
    )
    
    esg_response = None
    if esg_data:
        esg_response = ESGScoreResponse(
            esg_score=esg_data.esg_score,
            environmental_score=esg_data.environmental_score,
            social_score=esg_data.social_score,
            governance_score=esg_data.governance_score,
            esg_rating=esg_data.esg_rating,
            data_source=esg_data.data_source,
            reporting_period=esg_data.reporting_period,
            collection_date=esg_data.collection_date
        )
    
    financial_response = None
    if financial_data:
        financial_response = FinancialDataResponse(
            revenue=financial_data.revenue,
            profit_margin=financial_data.profit_margin,
            roe=financial_data.roe,
            debt_to_equity=financial_data.debt_to_equity,
            price_to_earnings=financial_data.price_to_earnings,
            beta=financial_data.beta,
            volatility=financial_data.volatility,
            sharpe_ratio=financial_data.sharpe_ratio,
            returns_1y=financial_data.returns_1y,
            market_cap=financial_data.market_cap,
            reporting_period=financial_data.reporting_period,
            fiscal_year=financial_data.fiscal_year
        )
    
    return CompanyDetailResponse(
        company=company_response,
        esg_data=esg_response,
        financial_data=financial_response
    )

@app.get("/correlations", response_model=List[CorrelationResponse])
async def get_correlations(
    index_code: str = Query("IBEX35", description="Stock index code"),
    significant_only: bool = Query(True, description="Return only statistically significant correlations"),
    db: Session = Depends(get_db)
):
    """Get ESG-Financial correlations"""
    
    # Get stock index
    stock_index = db.query(StockIndex).filter(StockIndex.code == index_code).first()
    if not stock_index:
        raise HTTPException(status_code=404, detail=f"Stock index {index_code} not found")
    
    # Build query
    query = db.query(AnalysisResult).filter(
        AnalysisResult.analysis_type == "correlation",
        AnalysisResult.stock_index_id == stock_index.id
    )
    
    if significant_only:
        query = query.filter(AnalysisResult.statistical_significance == True)
    
    correlations = query.order_by(desc(AnalysisResult.created_at)).all()
    
    # Format response
    result = []
    for corr in correlations:
        result.append(CorrelationResponse(
            analysis_name=corr.analysis_name,
            correlation_coefficient=corr.correlation_coefficient,
            p_value=corr.p_value,
            statistical_significance=corr.statistical_significance,
            sample_size=corr.sample_size,
            created_at=corr.created_at
        ))
    
    logger.info(f"Retrieved {len(result)} correlations for {index_code}")
    return result

@app.get("/sectors", response_model=List[SectorAnalysisResponse])
async def get_sector_analysis(
    index_code: str = Query("IBEX35", description="Stock index code"),
    db: Session = Depends(get_db)
):
    """Get sector-wise analysis"""
    
    # Get stock index
    stock_index = db.query(StockIndex).filter(StockIndex.code == index_code).first()
    if not stock_index:
        raise HTTPException(status_code=404, detail=f"Stock index {index_code} not found")
    
    # Get sector statistics
    sector_stats = db.query(
        Company.sector,
        func.count(Company.id).label('company_count'),
        func.avg(ESGScore.esg_score).label('avg_esg_score'),
        func.avg(FinancialData.roe).label('avg_roe')
    ).join(
        ESGScore, Company.id == ESGScore.company_id, isouter=True
    ).join(
        FinancialData, Company.id == FinancialData.company_id, isouter=True
    ).filter(
        Company.stock_index_id == stock_index.id,
        Company.is_active == True
    ).group_by(Company.sector).all()
    
    # Format response
    result = []
    for stat in sector_stats:
        # Get top companies in sector
        top_companies = db.query(Company.name).filter(
            Company.sector == stat.sector,
            Company.stock_index_id == stock_index.id,
            Company.is_active == True
        ).limit(3).all()
        
        result.append(SectorAnalysisResponse(
            sector=stat.sector,
            company_count=stat.company_count,
            avg_esg_score=round(stat.avg_esg_score, 2) if stat.avg_esg_score else None,
            avg_financial_performance=round(stat.avg_roe, 2) if stat.avg_roe else None,
            top_companies=[company.name for company in top_companies]
        ))
    
    return result

@app.get("/ml-models", response_model=List[Dict[str, Any]])
async def get_ml_models(
    index_code: str = Query("IBEX35", description="Stock index code"),
    db: Session = Depends(get_db)
):
    """Get machine learning model results"""
    
    # Get stock index
    stock_index = db.query(StockIndex).filter(StockIndex.code == index_code).first()
    if not stock_index:
        raise HTTPException(status_code=404, detail=f"Stock index {index_code} not found")
    
    # Get ML model results
    ml_results = db.query(AnalysisResult).filter(
        AnalysisResult.analysis_type == "ml_model",
        AnalysisResult.stock_index_id == stock_index.id
    ).order_by(desc(AnalysisResult.created_at)).all()
    
    # Format response
    result = []
    for model in ml_results:
        result.append({
            "model_name": model.analysis_name,
            "model_type": model.model_type,
            "train_score": model.train_score,
            "test_score": model.test_score,
            "cross_val_score": model.cross_val_score,
            "r_squared": model.r_squared,
            "sample_size": model.sample_size,
            "created_at": model.created_at
        })
    
    return result

@app.get("/summary", response_model=Dict[str, Any])
async def get_executive_summary(
    index_code: str = Query("IBEX35", description="Stock index code"),
    db: Session = Depends(get_db)
):
    """Get executive summary with key metrics"""
    
    # Get stock index
    stock_index = db.query(StockIndex).filter(StockIndex.code == index_code).first()
    if not stock_index:
        raise HTTPException(status_code=404, detail=f"Stock index {index_code} not found")
    
    # Get basic statistics
    total_companies = db.query(Company).filter(
        Company.stock_index_id == stock_index.id,
        Company.is_active == True
    ).count()
    
    # Get ESG statistics
    esg_stats = db.query(
        func.avg(ESGScore.esg_score).label('avg_esg'),
        func.count(ESGScore.id).label('esg_count')
    ).join(Company).filter(
        Company.stock_index_id == stock_index.id,
        ESGScore.is_latest == True
    ).first()
    
    # Get correlation count
    significant_correlations = db.query(AnalysisResult).filter(
        AnalysisResult.analysis_type == "correlation",
        AnalysisResult.stock_index_id == stock_index.id,
        AnalysisResult.statistical_significance == True
    ).count()
    
    # Get sector count
    sector_count = db.query(func.count(func.distinct(Company.sector))).filter(
        Company.stock_index_id == stock_index.id,
        Company.is_active == True
    ).scalar()
    
    # Get latest data collection info
    latest_collection = db.query(DataCollectionLog).filter(
        DataCollectionLog.stock_index_id == stock_index.id
    ).order_by(desc(DataCollectionLog.started_at)).first()
    
    return {
        "index": {
            "code": stock_index.code,
            "name": stock_index.name,
            "country": stock_index.country
        },
        "statistics": {
            "total_companies": total_companies,
            "companies_with_esg_data": esg_stats.esg_count if esg_stats else 0,
            "average_esg_score": round(esg_stats.avg_esg, 2) if esg_stats and esg_stats.avg_esg else None,
            "significant_correlations": significant_correlations,
            "sectors_analyzed": sector_count
        },
        "data_quality": {
            "last_update": latest_collection.completed_at if latest_collection else None,
            "success_rate": latest_collection.success_rate if latest_collection else None,
            "data_coverage": round((esg_stats.esg_count / total_companies * 100), 1) if esg_stats and total_companies > 0 else 0
        },
        "generated_at": datetime.utcnow()
    }

@app.get("/stats")
async def get_api_stats(db: Session = Depends(get_db)):
    """Get API and database statistics"""
    
    # Count records by type
    total_companies = db.query(Company).count()
    total_esg_records = db.query(ESGScore).count()
    total_financial_records = db.query(FinancialData).count()
    total_analysis_results = db.query(AnalysisResult).count()
    
    # Get indices info
    indices = db.query(StockIndex).all()
    
    return {
        "api_version": "2.0.0",
        "database_stats": {
            "total_companies": total_companies,
            "total_esg_records": total_esg_records,
            "total_financial_records": total_financial_records,
            "total_analysis_results": total_analysis_results
        },
        "supported_indices": [
            {
                "code": idx.code,
                "name": idx.name,
                "active": idx.is_active
            } for idx in indices
        ],
        "features": [
            "Multi-index support",
            "Real-time database queries",
            "Historical data tracking",
            "Statistical significance testing",
            "Machine learning model results"
        ]
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        # Initialize database tables
        engine, _ = DatabaseConfig.create_engine_and_session()
        from database import Base
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Database initialized successfully")
        logger.info("🚀 ESG Analytics API v2 started - Database Edition")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
