from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ui.simple_ui import launch_gradio
from ui.schemas import router as agent_router
from ui.schemas import router as backtest_router
from ui.portfolio_api import router as portfolio_router
from ui.database_api import router as database_router
from ui.screener_api import router as screener_router
from ui.data_sync_api import router as data_sync_router
import logging
import sys

# Add parent directory to path to import from data module
sys.path.append('/app')

# Import data modules
from data.startup import initialize as initialize_data

# Import backtest patch to fix market_value errors
try:
    from backtest.backtest_patch import apply_patches
    logging.info("Imported backtest patch module successfully")
except ImportError as e:
    logging.error(f"Could not import backtest patch: {str(e)}")

app = FastAPI()

# Allow CORS for Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with appropriate prefixes
app.include_router(agent_router, prefix="")
app.include_router(backtest_router, prefix="/backtest")
app.include_router(portfolio_router, prefix="")
app.include_router(database_router, prefix="")
app.include_router(screener_router, prefix="")
app.include_router(data_sync_router, prefix="")

# We don't need to launch Gradio from here anymore
# The Gradio UI is now launched independently in its own container

# Initialize database and data synchronization
try:
    initialize_data()
except Exception as e:
    logging.error(f"Error initializing data components: {str(e)}")
    # Continue even if initialization fails

# Apply backtesting patches
try:
    apply_patches()
    logging.info("Successfully applied backtesting patches")
except Exception as e:
    logging.error(f"Error applying backtesting patches: {str(e)}")
    # Continue even if patching fails

@app.get("/ping")
def ping():
    return {"status": "ok"}
