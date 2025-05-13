from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

app = FastAPI(
    title="Stock Price Predictor API",
    description="API for predicting stock prices using LSTM model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the router
app.include_router(router, prefix="/api/v1", tags=["predictions"])

@app.get("/")
async def root():
    """
    Root endpoint that redirects to the API documentation.
    """
    return {
        "message": "Welcome to Stock Price Predictor API",
        "documentation": "/docs"
    } 