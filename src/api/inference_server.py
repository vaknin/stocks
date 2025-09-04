"""
Real-Time Inference Server for Phase 7 Production System.
Provides FastAPI-based REST API for real-time stock prediction and trading recommendations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import uvicorn
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import threading
import logging
from pathlib import Path
import json
import os
from contextlib import asynccontextmanager

# Import our Phase 7 components
from ..models.efficient_prediction_pipeline import EfficientPredictionPipeline, PredictionRequest
from ..models.ensemble import MetaLearningEnsemble
from ..models.memory_optimization import get_memory_optimizer
from ..trading.recommendation_engine import TradingRecommendationEngine
from ..config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Global state
app_state = {
    'ensemble_model': None,
    'prediction_pipeline': None,
    'recommendation_engine': None,
    'memory_optimizer': None,
    'startup_time': None,
    'request_count': 0,
    'health_status': 'starting'
}


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    ticker: str = Field(..., description="Stock ticker symbol")
    data: Optional[Dict[str, List[float]]] = Field(None, description="OHLCV data as dict")
    horizons: List[int] = Field([1, 5, 20], description="Prediction horizons in days")
    include_confidence: bool = Field(True, description="Include confidence scores")
    include_intervals: bool = Field(True, description="Include prediction intervals")
    use_cache: bool = Field(True, description="Use cached results if available")


class MultiPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    tickers: List[str] = Field(..., description="List of stock ticker symbols")
    data: Optional[Dict[str, Dict[str, List[float]]]] = Field(None, description="Multi-ticker OHLCV data")
    horizons: List[int] = Field([1, 5, 20], description="Prediction horizons in days")
    include_confidence: bool = Field(True, description="Include confidence scores")
    include_intervals: bool = Field(True, description="Include prediction intervals")
    max_parallel: int = Field(4, description="Maximum parallel requests")


class RecommendationRequest(BaseModel):
    """Request model for trading recommendations."""
    tickers: List[str] = Field(..., description="List of stock ticker symbols")
    portfolio_value: float = Field(100000.0, description="Current portfolio value")
    timeframes: List[str] = Field(['daily'], description="Analysis timeframes")
    max_positions: int = Field(5, description="Maximum number of positions")
    risk_tolerance: str = Field('moderate', description="Risk tolerance level")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    models_loaded: bool
    memory_usage_mb: float
    request_count: int
    performance_stats: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    ticker: str
    predictions: Dict[str, Any]
    confidence_scores: Optional[Dict[str, float]] = None
    prediction_intervals: Optional[Dict[str, List[float]]] = None
    processing_time_ms: float
    cache_hit: bool
    timestamp: datetime
    model_version: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None


# Startup and shutdown handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting Phase 7 Inference Server...")
    
    try:
        await initialize_models()
        app_state['startup_time'] = datetime.now()
        app_state['health_status'] = 'healthy'
        logger.info("Inference server startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app_state['health_status'] = 'unhealthy'
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down inference server...")
    await cleanup_resources()
    logger.info("Shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Phase 7 Stock Prediction API",
    description="Real-time stock prediction and trading recommendations using advanced ensemble models",
    version="7.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Authentication (simple bearer token)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - replace with proper auth in production."""
    if credentials is None:
        return None  # Allow unauthenticated access for demo
    
    # In production, validate JWT token or API key here
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    
    # Dummy validation - replace with real authentication
    if credentials.credentials != "demo-token-123":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"user_id": "demo_user"}


# Utility functions
async def initialize_models():
    """Initialize all models and components."""
    logger.info("Initializing ensemble model...")
    
    # Initialize ensemble model
    app_state['ensemble_model'] = MetaLearningEnsemble(
        horizon_len=[1, 5, 20],
        enable_neural_meta_learning=True,
        enable_online_learning=True,
        enable_multi_resolution=True,
        device="cpu"  # Use CPU for API server
    )
    
    # Initialize prediction pipeline
    app_state['prediction_pipeline'] = EfficientPredictionPipeline(
        ensemble_model=app_state['ensemble_model'],
        cache_config={
            'max_size': 2000,
            'ttl_seconds': 300,  # 5 minutes
            'memory_limit_mb': 512
        },
        parallel_config={
            'max_workers': 4,
            'use_process_pool': False
        }
    )
    
    # Initialize recommendation engine
    app_state['recommendation_engine'] = TradingRecommendationEngine()
    
    # Initialize memory optimizer
    app_state['memory_optimizer'] = get_memory_optimizer(enable_monitoring=True)
    
    logger.info("All models initialized successfully")


async def cleanup_resources():
    """Clean up resources during shutdown."""
    if app_state['prediction_pipeline']:
        app_state['prediction_pipeline'].shutdown()
    
    if app_state['memory_optimizer']:
        app_state['memory_optimizer'].shutdown()


def increment_request_count():
    """Increment request counter."""
    app_state['request_count'] += 1


def validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker symbol."""
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Invalid ticker symbol")
    
    ticker = ticker.upper().strip()
    
    # Basic ticker validation
    if len(ticker) > 10 or not ticker.isalpha():
        raise ValueError(f"Invalid ticker format: {ticker}")
    
    return ticker


def create_sample_data(ticker: str, days: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    np.random.seed(hash(ticker) % 2147483647)
    
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
    base_price = np.random.uniform(50, 500)
    
    prices = [base_price]
    for _ in range(days - 1):
        change = np.random.normal(0.001, 0.02)  # 0.1% drift, 2% volatility
        prices.append(prices[-1] * (1 + change))
    
    return pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, days)
    }, index=dates)


# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Phase 7 Stock Prediction API",
        "version": "7.0.0",
        "status": app_state['health_status'],
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    current_time = datetime.now()
    uptime = (current_time - app_state['startup_time']).total_seconds() if app_state['startup_time'] else 0
    
    # Get memory usage
    memory_usage = 0
    if app_state['memory_optimizer']:
        stats = app_state['memory_optimizer'].get_comprehensive_stats()
        memory_usage = stats.get('system', {}).get('process_memory_mb', 0)
    
    # Get performance statistics
    performance_stats = {}
    if app_state['prediction_pipeline']:
        performance_stats = app_state['prediction_pipeline'].get_performance_stats()
    
    return HealthCheckResponse(
        status=app_state['health_status'],
        timestamp=current_time,
        uptime_seconds=uptime,
        version="7.0.0",
        models_loaded=app_state['ensemble_model'] is not None,
        memory_usage_mb=memory_usage,
        request_count=app_state['request_count'],
        performance_stats=performance_stats
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Single stock prediction endpoint."""
    increment_request_count()
    
    try:
        # Validate ticker
        ticker = validate_ticker(request.ticker)
        
        # Get or create price data
        if request.data:
            # Convert provided data to DataFrame
            price_data = pd.DataFrame(request.data)
        else:
            # Generate sample data for demonstration
            price_data = create_sample_data(ticker)
        
        # Make prediction
        start_time = time.time()
        
        if app_state['prediction_pipeline'] is None:
            raise HTTPException(status_code=503, detail="Prediction pipeline not available")
        
        result = app_state['prediction_pipeline'].predict_single(
            ticker=ticker,
            price_data=price_data,
            horizons=request.horizons,
            return_confidence=request.include_confidence,
            use_cache=request.use_cache
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Extract confidence scores and intervals if requested
        confidence_scores = None
        prediction_intervals = None
        
        if request.include_confidence and 'error' not in result.predictions:
            confidence_scores = {}
            prediction_intervals = {}
            
            for horizon in request.horizons:
                horizon_key = f'horizon_{horizon}'
                if horizon_key in result.predictions:
                    horizon_data = result.predictions[horizon_key]
                    if 'confidence' in horizon_data:
                        confidence_scores[horizon_key] = horizon_data['confidence']
                    if 'prediction_interval' in horizon_data:
                        prediction_intervals[horizon_key] = horizon_data['prediction_interval']
        
        # Schedule background memory optimization
        background_tasks.add_task(optimize_memory_if_needed)
        
        return PredictionResponse(
            ticker=ticker,
            predictions=result.predictions,
            confidence_scores=confidence_scores,
            prediction_intervals=prediction_intervals,
            processing_time_ms=processing_time,
            cache_hit=result.cache_hit,
            timestamp=datetime.now(),
            model_version="7.0.0"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    request: MultiPredictionRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Batch prediction endpoint for multiple stocks."""
    increment_request_count()
    
    try:
        if len(request.tickers) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 50)")
        
        # Validate all tickers
        validated_tickers = [validate_ticker(ticker) for ticker in request.tickers]
        
        # Prepare prediction requests
        prediction_requests = []
        
        for ticker in validated_tickers:
            # Get price data
            if request.data and ticker in request.data:
                price_data = pd.DataFrame(request.data[ticker])
            else:
                price_data = create_sample_data(ticker)
            
            pred_req = PredictionRequest(
                ticker=ticker,
                price_data=price_data,
                multi_asset_data=None,
                horizons=request.horizons,
                return_confidence=request.include_confidence,
                priority=1
            )
            prediction_requests.append(pred_req)
        
        # Process predictions in parallel
        start_time = time.time()
        
        if app_state['prediction_pipeline'] is None:
            raise HTTPException(status_code=503, detail="Prediction pipeline not available")
        
        results = app_state['prediction_pipeline'].predict_parallel(
            requests=prediction_requests,
            max_parallel=min(request.max_parallel, 8)
        )
        
        total_time = (time.time() - start_time) * 1000
        
        # Convert results to response format
        responses = []
        
        for result in results:
            # Extract confidence scores and intervals
            confidence_scores = None
            prediction_intervals = None
            
            if request.include_confidence and 'error' not in result.predictions:
                confidence_scores = {}
                prediction_intervals = {}
                
                for horizon in request.horizons:
                    horizon_key = f'horizon_{horizon}'
                    if horizon_key in result.predictions:
                        horizon_data = result.predictions[horizon_key]
                        if 'confidence' in horizon_data:
                            confidence_scores[horizon_key] = horizon_data['confidence']
                        if 'prediction_interval' in horizon_data:
                            prediction_intervals[horizon_key] = horizon_data['prediction_interval']
            
            responses.append(PredictionResponse(
                ticker=result.ticker,
                predictions=result.predictions,
                confidence_scores=confidence_scores,
                prediction_intervals=prediction_intervals,
                processing_time_ms=result.processing_time_ms,
                cache_hit=result.cache_hit,
                timestamp=datetime.now(),
                model_version="7.0.0"
            ))
        
        # Schedule background optimization
        background_tasks.add_task(optimize_memory_if_needed)
        
        logger.info(f"Batch prediction completed: {len(results)} stocks in {total_time:.1f}ms")
        
        return responses
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/recommendations")
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Get trading recommendations based on predictions."""
    increment_request_count()
    
    try:
        if app_state['recommendation_engine'] is None:
            raise HTTPException(status_code=503, detail="Recommendation engine not available")
        
        # Generate recommendations
        start_time = time.time()
        
        recommendations = app_state['recommendation_engine'].generate_recommendations(
            timeframes=request.timeframes
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Filter recommendations to requested tickers if specified
        if request.tickers:
            filtered_recommendations = []
            for rec in recommendations.recommendations:
                if rec.symbol in request.tickers:
                    filtered_recommendations.append(rec)
            recommendations.recommendations = filtered_recommendations
        
        # Convert to JSON-serializable format
        result = {
            'analysis_date': recommendations.analysis_date,
            'current_portfolio_value': recommendations.current_portfolio_value,
            'recommended_allocation': recommendations.recommended_allocation,
            'cash_reserve': recommendations.cash_reserve,
            'market_outlook': recommendations.market_outlook,
            'risk_assessment': recommendations.risk_assessment,
            'recommendations': [
                {
                    'action': rec.action,
                    'symbol': rec.symbol,
                    'recommended_shares': rec.recommended_shares,
                    'dollar_amount': rec.dollar_amount,
                    'entry_price': rec.entry_price,
                    'stop_loss': rec.stop_loss,
                    'target_price': rec.target_price,
                    'confidence': rec.confidence,
                    'reasoning': rec.reasoning,
                    'priority': rec.priority,
                    'timeframe': rec.timeframe
                }
                for rec in recommendations.recommendations
            ],
            'processing_time_ms': processing_time,
            'timestamp': datetime.now()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@app.get("/models/status")
async def get_model_status(user=Depends(get_current_user)):
    """Get detailed model status and performance metrics."""
    try:
        status = {}
        
        if app_state['ensemble_model']:
            status['ensemble'] = app_state['ensemble_model'].get_ensemble_status()
        
        if app_state['prediction_pipeline']:
            status['prediction_pipeline'] = app_state['prediction_pipeline'].get_performance_stats()
        
        if app_state['memory_optimizer']:
            status['memory'] = app_state['memory_optimizer'].get_comprehensive_stats()
        
        status['request_statistics'] = {
            'total_requests': app_state['request_count'],
            'uptime_seconds': (datetime.now() - app_state['startup_time']).total_seconds() if app_state['startup_time'] else 0,
            'requests_per_minute': app_state['request_count'] / max(1, (datetime.now() - app_state['startup_time']).total_seconds() / 60) if app_state['startup_time'] else 0
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")


@app.post("/admin/optimize-memory")
async def optimize_memory_endpoint(user=Depends(get_current_user)):
    """Manually trigger memory optimization."""
    try:
        if app_state['memory_optimizer'] is None:
            raise HTTPException(status_code=503, detail="Memory optimizer not available")
        
        result = app_state['memory_optimizer'].optimize_for_inference()
        
        return {
            'status': 'success',
            'optimization_result': result,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Memory optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Memory optimization failed: {str(e)}")


@app.post("/admin/clear-cache")
async def clear_cache_endpoint(user=Depends(get_current_user)):
    """Clear prediction cache."""
    try:
        if app_state['prediction_pipeline'] is None:
            raise HTTPException(status_code=503, detail="Prediction pipeline not available")
        
        # Get cache stats before clearing
        before_stats = app_state['prediction_pipeline'].cache.get_stats()
        
        # Clear caches
        app_state['prediction_pipeline'].cache.l1_cache.clear()
        app_state['prediction_pipeline'].cache.l2_cache.clear()
        app_state['prediction_pipeline'].cache.access_times.clear()
        app_state['prediction_pipeline'].cache.creation_times.clear()
        app_state['prediction_pipeline'].cache.hit_counts.clear()
        app_state['prediction_pipeline'].cache.memory_usage.clear()
        
        after_stats = app_state['prediction_pipeline'].cache.get_stats()
        
        return {
            'status': 'success',
            'before_stats': before_stats,
            'after_stats': after_stats,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Cache clearing error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")


# Background tasks
async def optimize_memory_if_needed():
    """Background task to optimize memory if usage is high."""
    try:
        if app_state['memory_optimizer']:
            current_usage = app_state['memory_optimizer'].monitor.get_current_usage()
            
            if current_usage.percent_used > 85:  # High memory usage
                app_state['memory_optimizer'].perform_light_cleanup()
                logger.info("Background memory optimization performed")
    
    except Exception as e:
        logger.debug(f"Background memory optimization error: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now(),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc) if os.getenv("DEBUG") == "1" else None,
            timestamp=datetime.now(),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


# Server configuration
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the inference server."""
    uvicorn.run(
        "src.api.inference_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    run_server()