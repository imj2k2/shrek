# Discord bot skeleton for notifications and commands
import discord
from discord.ext import commands, tasks
import os
from config.settings import settings
import requests

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="/", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    daily_report.start()
    weekly_report.start()

@bot.command()
async def positions(ctx):
    resp = requests.get("http://localhost:8000/portfolio")
    if resp.ok:
        positions = resp.json().get('portfolio', {}).get('positions', [])
        await ctx.send(f"Current positions: {positions}")
    else:
        await ctx.send("Could not fetch positions.")

@bot.command()
async def portfolio(ctx):
    resp = requests.get("http://localhost:8000/portfolio")
    if resp.ok:
        portfolio = resp.json().get('portfolio', {})
        await ctx.send(f"Current portfolio: {portfolio}")
    else:
        await ctx.send("Could not fetch portfolio.")

@bot.command()
async def recommendations(ctx):
    await ctx.send("Active trade recommendations: ...")

@bot.command()
async def market_sentiment(ctx):
    await ctx.send("AI-generated market sentiment: ...")

async def notify_signal(signal):
    channel = discord.utils.get(bot.get_all_channels(), name="trading-alerts")
    if channel:
        await channel.send(f"New trading signal: {signal}")

async def notify_execution(result):
    channel = discord.utils.get(bot.get_all_channels(), name="trading-alerts")
    if channel:
        await channel.send(f"Trade executed: {result}")

async def notify_risk_alert(alert):
    channel = discord.utils.get(bot.get_all_channels(), name="risk-alerts")
    if channel:
        await channel.send(f"Risk alert: {alert}")

@tasks.loop(hours=24)
async def daily_report():
    channel = discord.utils.get(bot.get_all_channels(), name="reports")
    if channel:
        await channel.send("Daily P/L summary: ...")

@tasks.loop(hours=168)
async def weekly_report():
    channel = discord.utils.get(bot.get_all_channels(), name="reports")
    if channel:
        await channel.send("Weekly performance report: ...")

def run_discord_bot():
    bot.run(settings.DISCORD_TOKEN)
