# Discord bot for trading platform notifications and commands
import discord
from discord.ext import commands, tasks
import os
import requests
import json
import logging
import datetime
from discord import app_commands

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('discord_bot.log')
    ]
)

# Get API URL from environment or use default
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Get Discord token from environment
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    logging.error("DISCORD_TOKEN environment variable not set")
    raise ValueError("DISCORD_TOKEN environment variable must be set")

# Setup intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize bot
bot = commands.Bot(command_prefix="/", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    daily_report.start()
    weekly_report.start()

@bot.command()
async def positions(ctx):
    try:
        resp = requests.get(f"{API_BASE_URL}/portfolio")
        if resp.ok:
            portfolio = resp.json().get('portfolio', {})
            positions = portfolio.get('positions', {})
            
            if not positions:
                await ctx.send("No active positions found.")
                return
                
            # Format positions nicely
            position_list = []
            for symbol, details in positions.items():
                qty = details.get('quantity', 0)
                entry = details.get('avg_price', 0)
                current = details.get('current_price', entry)
                pnl = (current - entry) * qty
                pnl_pct = ((current / entry) - 1) * 100 if entry > 0 else 0
                
                position_list.append(
                    f"**{symbol}**: {qty} shares @ ${entry:.2f} | Current: ${current:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)"
                )
            
            # Create embed
            embed = discord.Embed(
                title="Current Positions",
                description="\n".join(position_list),
                color=discord.Color.blue()
            )
            embed.set_footer(text=f"As of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"Error fetching positions: {resp.status_code}")
    except Exception as e:
        logging.error(f"Error in positions command: {str(e)}")
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command()
async def portfolio(ctx):
    try:
        resp = requests.get(f"{API_BASE_URL}/portfolio")
        if resp.ok:
            portfolio = resp.json().get('portfolio', {})
            
            # Format portfolio data
            equity = portfolio.get('equity', 0)
            cash = portfolio.get('cash', 0)
            invested = portfolio.get('invested', 0)
            day_pnl = portfolio.get('day_pnl', 0)
            total_pnl = portfolio.get('total_pnl', 0)
            total_pnl_pct = portfolio.get('total_pnl_pct', 0) * 100
            
            # Create embed
            embed = discord.Embed(
                title="Portfolio Summary",
                color=discord.Color.green() if total_pnl >= 0 else discord.Color.red()
            )
            
            embed.add_field(name="Total Equity", value=f"${equity:,.2f}", inline=True)
            embed.add_field(name="Cash", value=f"${cash:,.2f}", inline=True)
            embed.add_field(name="Invested", value=f"${invested:,.2f}", inline=True)
            embed.add_field(name="Day P&L", value=f"${day_pnl:,.2f}", inline=True)
            embed.add_field(name="Total P&L", value=f"${total_pnl:,.2f}", inline=True)
            embed.add_field(name="Total P&L %", value=f"{total_pnl_pct:.2f}%", inline=True)
            
            embed.set_footer(text=f"As of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"Error fetching portfolio: {resp.status_code}")
    except Exception as e:
        logging.error(f"Error in portfolio command: {str(e)}")
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command()
async def recommendations(ctx):
    try:
        resp = requests.get(f"{API_BASE_URL}/agents/signal", params={"lookback": 5})
        if resp.ok:
            signals = resp.json().get('signals', {})
            
            if not signals:
                await ctx.send("No active recommendations available.")
                return
                
            # Format recommendations
            embed = discord.Embed(
                title="Trading Recommendations",
                description="AI-generated trading signals based on current market conditions",
                color=discord.Color.gold()
            )
            
            for agent_name, signal in signals.items():
                if isinstance(signal, dict) and 'action' in signal:
                    action = signal.get('action', 'unknown')
                    symbol = signal.get('symbol', 'N/A')
                    confidence = signal.get('confidence', 0)
                    price = signal.get('price', 0)
                    strategy = signal.get('strategy', 'N/A')
                    
                    # Color code by action
                    action_emoji = "🔄"
                    if action == "buy":
                        action_emoji = "🟢"
                    elif action == "sell":
                        action_emoji = "🔴"
                    
                    embed.add_field(
                        name=f"{agent_name.capitalize()} Agent",
                        value=f"{action_emoji} **{action.upper()}** {symbol} @ ${price:.2f}\n" + 
                              f"Strategy: {strategy}\nConfidence: {confidence:.2f}",
                        inline=False
                    )
            
            embed.set_footer(text=f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"Error fetching recommendations: {resp.status_code}")
    except Exception as e:
        logging.error(f"Error in recommendations command: {str(e)}")
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command()
async def market_sentiment(ctx):
    try:
        resp = requests.get(f"{API_BASE_URL}/market/sentiment")
        if resp.ok:
            sentiment = resp.json().get('sentiment', {})
            
            # Create embed
            embed = discord.Embed(
                title="Market Sentiment Analysis",
                description="AI-generated market sentiment based on multiple data sources",
                color=discord.Color.purple()
            )
            
            # Add overall sentiment
            overall = sentiment.get('overall', 0)
            sentiment_text = "Neutral 😐"
            if overall > 0.5:
                sentiment_text = "Bullish 🐂"
            elif overall < -0.5:
                sentiment_text = "Bearish 🐻"
            elif overall > 0:
                sentiment_text = "Slightly Bullish 📈"
            elif overall < 0:
                sentiment_text = "Slightly Bearish 📉"
                
            embed.add_field(name="Overall Market Sentiment", value=sentiment_text, inline=False)
            
            # Add sector sentiments if available
            sectors = sentiment.get('sectors', {})
            if sectors:
                sectors_text = ""
                for sector, value in sectors.items():
                    emoji = "😐"
                    if value > 0.5:
                        emoji = "🔥"
                    elif value < -0.5:
                        emoji = "❄️"
                    elif value > 0:
                        emoji = "📈"
                    elif value < 0:
                        emoji = "📉"
                    sectors_text += f"{sector}: {emoji} {value:.2f}\n"
                
                embed.add_field(name="Sector Sentiment", value=sectors_text, inline=False)
            
            # Add market indicators
            indicators = sentiment.get('indicators', {})
            if indicators:
                indicators_text = ""
                for indicator, value in indicators.items():
                    indicators_text += f"{indicator}: {value}\n"
                
                embed.add_field(name="Market Indicators", value=indicators_text, inline=False)
            
            embed.set_footer(text=f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await ctx.send(embed=embed)
        else:
            # If endpoint doesn't exist yet, show placeholder
            embed = discord.Embed(
                title="Market Sentiment Analysis",
                description="AI-generated market sentiment based on multiple data sources",
                color=discord.Color.purple()
            )
            
            embed.add_field(
                name="Overall Market Sentiment", 
                value="Slightly Bullish 📈", 
                inline=False
            )
            
            embed.add_field(
                name="Sector Sentiment", 
                value="Technology: 🔥 0.75\nFinancials: 📈 0.32\nHealthcare: 😐 0.05\nEnergy: 📉 -0.18", 
                inline=False
            )
            
            embed.add_field(
                name="Market Indicators", 
                value="VIX: 18.5\nS&P 500 P/E: 21.2\nYield Curve: Slightly inverted", 
                inline=False
            )
            
            embed.set_footer(text="This is sample data. Market sentiment endpoint not available yet.")
            await ctx.send(embed=embed)
    except Exception as e:
        logging.error(f"Error in market_sentiment command: {str(e)}")
        await ctx.send(f"An error occurred: {str(e)}")

async def notify_signal(signal):
    try:
        channel = discord.utils.get(bot.get_all_channels(), name="trading-alerts")
        if not channel:
            logging.warning("Trading alerts channel not found")
            return
            
        # Format signal as embed
        embed = discord.Embed(
            title="🔔 New Trading Signal",
            color=discord.Color.blue()
        )
        
        if isinstance(signal, dict):
            action = signal.get('action', 'unknown')
            symbol = signal.get('symbol', 'N/A')
            price = signal.get('price', 0)
            strategy = signal.get('strategy', 'N/A')
            agent = signal.get('agent', 'system')
            confidence = signal.get('confidence', 0)
            
            # Color code by action
            if action == "buy":
                embed.color = discord.Color.green()
            elif action == "sell":
                embed.color = discord.Color.red()
                
            embed.add_field(name="Action", value=action.upper(), inline=True)
            embed.add_field(name="Symbol", value=symbol, inline=True)
            embed.add_field(name="Price", value=f"${price:.2f}", inline=True)
            embed.add_field(name="Strategy", value=strategy, inline=True)
            embed.add_field(name="Agent", value=agent, inline=True)
            embed.add_field(name="Confidence", value=f"{confidence:.2f}", inline=True)
        else:
            embed.description = f"Signal details: {signal}"
            
        embed.set_footer(text=f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        await channel.send(embed=embed)
    except Exception as e:
        logging.error(f"Error in notify_signal: {str(e)}")

async def notify_execution(result):
    try:
        channel = discord.utils.get(bot.get_all_channels(), name="trading-alerts")
        if not channel:
            logging.warning("Trading alerts channel not found")
            return
            
        # Format execution result as embed
        embed = discord.Embed(
            title="✅ Trade Executed",
            color=discord.Color.green()
        )
        
        if isinstance(result, dict):
            action = result.get('action', 'unknown')
            symbol = result.get('symbol', 'N/A')
            quantity = result.get('quantity', 0)
            price = result.get('price', 0)
            status = result.get('status', 'completed')
            
            # Color code by status
            if status != "completed":
                embed.color = discord.Color.gold()
                
            embed.add_field(name="Action", value=action.upper(), inline=True)
            embed.add_field(name="Symbol", value=symbol, inline=True)
            embed.add_field(name="Quantity", value=quantity, inline=True)
            embed.add_field(name="Price", value=f"${price:.2f}", inline=True)
            embed.add_field(name="Status", value=status, inline=True)
            
            # Add transaction ID if available
            if 'transaction_id' in result:
                embed.add_field(name="Transaction ID", value=result['transaction_id'], inline=True)
        else:
            embed.description = f"Execution details: {result}"
            
        embed.set_footer(text=f"Executed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        await channel.send(embed=embed)
    except Exception as e:
        logging.error(f"Error in notify_execution: {str(e)}")

async def notify_risk_alert(alert):
    try:
        channel = discord.utils.get(bot.get_all_channels(), name="risk-alerts")
        if not channel:
            # Fall back to trading-alerts if risk-alerts doesn't exist
            channel = discord.utils.get(bot.get_all_channels(), name="trading-alerts")
            if not channel:
                logging.warning("Risk alerts channel not found")
                return
                
        # Format risk alert as embed
        embed = discord.Embed(
            title="⚠️ Risk Alert",
            color=discord.Color.red()
        )
        
        if isinstance(alert, dict):
            alert_type = alert.get('type', 'general')
            severity = alert.get('severity', 'medium')
            message = alert.get('message', 'No details provided')
            action_required = alert.get('action_required', 'None')
            
            # Set color based on severity
            if severity == "low":
                embed.color = discord.Color.gold()
            elif severity == "medium":
                embed.color = discord.Color.orange()
            elif severity == "high":
                embed.color = discord.Color.red()
                
            embed.add_field(name="Alert Type", value=alert_type, inline=True)
            embed.add_field(name="Severity", value=severity.upper(), inline=True)
            embed.add_field(name="Message", value=message, inline=False)
            embed.add_field(name="Action Required", value=action_required, inline=False)
            
            # Add affected symbols if available
            if 'symbols' in alert:
                embed.add_field(name="Affected Symbols", value=", ".join(alert['symbols']), inline=False)
        else:
            embed.description = f"Alert details: {alert}"
            
        embed.set_footer(text=f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        await channel.send(embed=embed)
    except Exception as e:
        logging.error(f"Error in notify_risk_alert: {str(e)}")

@tasks.loop(hours=24)
async def daily_report():
    try:
        channel = discord.utils.get(bot.get_all_channels(), name="reports")
        if not channel:
            logging.warning("Reports channel not found")
            return
            
        # Get portfolio data
        resp = requests.get(f"{API_BASE_URL}/portfolio")
        if not resp.ok:
            await channel.send("⚠️ Failed to generate daily report: Could not fetch portfolio data")
            return
            
        portfolio = resp.json().get('portfolio', {})
        
        # Create daily report embed
        embed = discord.Embed(
            title="📊 Daily Performance Report",
            description=f"Trading summary for {datetime.datetime.now().strftime('%Y-%m-%d')}",
            color=discord.Color.blue()
        )
        
        # Add portfolio summary
        equity = portfolio.get('equity', 0)
        day_pnl = portfolio.get('day_pnl', 0)
        day_pnl_pct = (day_pnl / (equity - day_pnl)) * 100 if (equity - day_pnl) > 0 else 0
        
        embed.add_field(name="Total Equity", value=f"${equity:,.2f}", inline=True)
        embed.add_field(name="Day P&L", value=f"${day_pnl:,.2f}", inline=True)
        embed.add_field(name="Day P&L %", value=f"{day_pnl_pct:.2f}%", inline=True)
        
        # Add trade summary
        trades = portfolio.get('day_trades', [])
        if trades:
            trades_text = ""
            for trade in trades[:5]:  # Show only the first 5 trades
                action = trade.get('action', 'unknown')
                symbol = trade.get('symbol', 'N/A')
                quantity = trade.get('quantity', 0)
                price = trade.get('price', 0)
                pnl = trade.get('pnl', 0)
                
                trades_text += f"{action.upper()} {quantity} {symbol} @ ${price:.2f} | P&L: ${pnl:.2f}\n"
                
            if trades_text:
                embed.add_field(name="Today's Trades", value=trades_text, inline=False)
                
            if len(trades) > 5:
                embed.add_field(name="Note", value=f"Showing 5 of {len(trades)} trades", inline=False)
        else:
            embed.add_field(name="Today's Trades", value="No trades executed today", inline=False)
        
        embed.set_footer(text=f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        await channel.send(embed=embed)
    except Exception as e:
        logging.error(f"Error in daily_report: {str(e)}")
        try:
            await channel.send(f"⚠️ Error generating daily report: {str(e)}")
        except:
            pass

@tasks.loop(hours=168)  # Weekly (7 days * 24 hours)
async def weekly_report():
    try:
        channel = discord.utils.get(bot.get_all_channels(), name="reports")
        if not channel:
            logging.warning("Reports channel not found")
            return
            
        # Get portfolio data
        resp = requests.get(f"{API_BASE_URL}/portfolio")
        if not resp.ok:
            await channel.send("⚠️ Failed to generate weekly report: Could not fetch portfolio data")
            return
            
        portfolio = resp.json().get('portfolio', {})
        
        # Create weekly report embed
        embed = discord.Embed(
            title="📈 Weekly Performance Report",
            description=f"Trading summary for week ending {datetime.datetime.now().strftime('%Y-%m-%d')}",
            color=discord.Color.dark_blue()
        )
        
        # Add portfolio summary
        equity = portfolio.get('equity', 0)
        week_pnl = portfolio.get('week_pnl', 0)
        week_pnl_pct = (week_pnl / (equity - week_pnl)) * 100 if (equity - week_pnl) > 0 else 0
        
        embed.add_field(name="Total Equity", value=f"${equity:,.2f}", inline=True)
        embed.add_field(name="Week P&L", value=f"${week_pnl:,.2f}", inline=True)
        embed.add_field(name="Week P&L %", value=f"{week_pnl_pct:.2f}%", inline=True)
        
        # Add performance metrics
        metrics = portfolio.get('metrics', {})
        if metrics:
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = metrics.get('max_drawdown', 0) * 100
            win_rate = metrics.get('win_rate', 0) * 100
            
            embed.add_field(name="Sharpe Ratio", value=f"{sharpe:.2f}", inline=True)
            embed.add_field(name="Max Drawdown", value=f"{max_dd:.2f}%", inline=True)
            embed.add_field(name="Win Rate", value=f"{win_rate:.2f}%", inline=True)
        
        # Add top performers
        positions = portfolio.get('positions', {})
        if positions:
            # Sort positions by P&L
            sorted_positions = []
            for symbol, pos in positions.items():
                entry = pos.get('avg_price', 0)
                current = pos.get('current_price', entry)
                qty = pos.get('quantity', 0)
                pnl = (current - entry) * qty
                pnl_pct = ((current / entry) - 1) * 100 if entry > 0 else 0
                sorted_positions.append((symbol, pnl, pnl_pct, qty, current))
                
            sorted_positions.sort(key=lambda x: x[1], reverse=True)
            
            # Top performers
            top_text = ""
            for symbol, pnl, pnl_pct, qty, price in sorted_positions[:3]:
                top_text += f"{symbol}: {qty} shares @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)\n"
                
            if top_text:
                embed.add_field(name="Top Performers", value=top_text, inline=False)
                
            # Bottom performers
            if len(sorted_positions) >= 3:
                bottom_text = ""
                for symbol, pnl, pnl_pct, qty, price in sorted_positions[-3:]:
                    bottom_text += f"{symbol}: {qty} shares @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)\n"
                    
                if bottom_text:
                    embed.add_field(name="Bottom Performers", value=bottom_text, inline=False)
        
        embed.set_footer(text=f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        await channel.send(embed=embed)
    except Exception as e:
        logging.error(f"Error in weekly_report: {str(e)}")
        try:
            await channel.send(f"⚠️ Error generating weekly report: {str(e)}")
        except:
            pass

def run_discord_bot():
    """Run the Discord bot using the token from environment variables"""
    try:
        logging.info("Starting Discord bot...")
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logging.error(f"Failed to start Discord bot: {str(e)}")
        
if __name__ == "__main__":
    run_discord_bot()
