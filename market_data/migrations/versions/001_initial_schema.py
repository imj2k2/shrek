"""Initial schema creation

Revision ID: 001
Revises: 
Create Date: 2025-05-07

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create all tables from the models
    op.create_table(
        'symbols',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('symbol', sa.String(20), unique=True, nullable=False, index=True),
        sa.Column('name', sa.String(255)),
        sa.Column('description', sa.Text),
        sa.Column('sector', sa.String(100)),
        sa.Column('industry', sa.String(100)),
        sa.Column('exchange', sa.String(20)),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('is_etf', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
    )
    
    op.create_table(
        'market_data',
        sa.Column('id', UUID(as_uuid=True), nullable=False),
        sa.Column('symbol_id', UUID(as_uuid=True), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('open', sa.Float),
        sa.Column('high', sa.Float),
        sa.Column('low', sa.Float),
        sa.Column('close', sa.Float),
        sa.Column('volume', sa.Integer),
        sa.Column('vwap', sa.Float),
        sa.Column('source', sa.String(50)),
        sa.Column('timeframe', sa.String(10)),
        sa.Column('sma_50', sa.Float),
        sa.Column('sma_200', sa.Float),
        sa.Column('ema_12', sa.Float),
        sa.Column('ema_26', sa.Float),
        sa.Column('macd', sa.Float),
        sa.Column('macd_signal', sa.Float),
        sa.Column('macd_hist', sa.Float),
        sa.Column('rsi', sa.Float)
    )
    
    op.create_table(
        'fundamentals',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('symbol_id', UUID(as_uuid=True), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column('date', sa.DateTime, nullable=False),
        sa.Column('pe_ratio', sa.Float),
        sa.Column('eps', sa.Float),
        sa.Column('dividend_yield', sa.Float),
        sa.Column('market_cap', sa.Float),
        sa.Column('price_to_book', sa.Float),
        sa.Column('price_to_sales', sa.Float),
        sa.Column('debt_to_equity', sa.Float),
        sa.Column('profit_margin', sa.Float),
        sa.Column('return_on_equity', sa.Float),
        sa.Column('beta', sa.Float)
    )
    
    op.create_table(
        'backtest_results',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('start_date', sa.DateTime, nullable=False),
        sa.Column('end_date', sa.DateTime, nullable=False),
        sa.Column('strategy', sa.String(100)),
        sa.Column('symbols', JSONB),
        sa.Column('initial_capital', sa.Float),
        sa.Column('final_capital', sa.Float),
        sa.Column('total_return', sa.Float),
        sa.Column('annual_return', sa.Float),
        sa.Column('sharpe_ratio', sa.Float),
        sa.Column('max_drawdown', sa.Float),
        sa.Column('win_rate', sa.Float),
        sa.Column('profit_factor', sa.Float),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('config', JSONB)
    )
    
    op.create_table(
        'backtest_trades',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('backtest_id', UUID(as_uuid=True), sa.ForeignKey("backtest_results.id"), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('action', sa.String(10), nullable=False),
        sa.Column('position_type', sa.String(10), nullable=False, server_default='long'),
        sa.Column('price', sa.Float, nullable=False),
        sa.Column('quantity', sa.Integer, nullable=False),
        sa.Column('commission', sa.Float, server_default='0'),
        sa.Column('profit_loss', sa.Float)
    )
    
    op.create_table(
        'backtest_equity_curve',
        sa.Column('id', UUID(as_uuid=True), nullable=False),
        sa.Column('backtest_id', UUID(as_uuid=True), sa.ForeignKey("backtest_results.id"), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('equity', sa.Float, nullable=False),
        sa.Column('cash', sa.Float, nullable=False),
        sa.Column('holdings_value', sa.Float, nullable=False),
        sa.Column('daily_return', sa.Float),
        sa.Column('drawdown', sa.Float, server_default='0')
    )
    
    op.create_table(
        'screener_criteria',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('criteria', JSONB, nullable=False),
        sa.Column('position_type', sa.String(10), server_default='long'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('last_used', sa.DateTime)
    )
    
    op.create_table(
        'screener_results',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('criteria_id', UUID(as_uuid=True), sa.ForeignKey("screener_criteria.id"), nullable=False),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('results', JSONB),
        sa.Column('count', sa.Integer)
    )
    
    # Create indexes for performance
    op.create_index('idx_market_data_symbol_timestamp', 'market_data', ['symbol_id', 'timestamp'])
    
    # Add composite primary keys including timestamp for TimescaleDB
    op.create_primary_key('pk_market_data', 'market_data', ['id', 'timestamp'])
    op.create_primary_key('pk_backtest_equity_curve', 'backtest_equity_curve', ['id', 'timestamp'])
    op.create_index('idx_fundamentals_symbol_date', 'fundamentals', ['symbol_id', 'date'])
    op.create_index('idx_backtest_trades_backtest_id', 'backtest_trades', ['backtest_id'])
    op.create_index('idx_backtest_equity_backtest_id', 'backtest_equity_curve', ['backtest_id'])
    
    # Execute raw SQL to convert tables to TimescaleDB hypertables
    try:
        # Use simpler syntax for creating hypertables
        op.execute("SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);")
        op.execute("SELECT create_hypertable('backtest_equity_curve', 'timestamp', if_not_exists => TRUE);")
    except Exception as e:
        print(f"Warning: Could not create hypertables: {e}")
        # Try alternative syntax if the first attempt fails
        try:
            op.execute("SELECT create_hypertable('market_data', 'timestamp');")
            op.execute("SELECT create_hypertable('backtest_equity_curve', 'timestamp');")
        except Exception as e2:
            print(f"Warning: Alternative hypertable creation failed: {e2}")
            print("Continuing with regular tables - TimescaleDB features may be limited.")


def downgrade() -> None:
    # Drop all tables in reverse order
    op.drop_table('screener_results')
    op.drop_table('screener_criteria')
    op.drop_table('backtest_equity_curve')
    op.drop_table('backtest_trades')
    op.drop_table('backtest_results')
    op.drop_table('fundamentals')
    op.drop_table('market_data')
    op.drop_table('symbols')
