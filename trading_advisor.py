#!/usr/bin/env python3
"""
AI Trading Advisor - Daily Trading Recommendations CLI
Main entry point for the AI semiconductor trading advisor system.

Usage:
    python trading_advisor.py                    # Generate daily recommendations
    python trading_advisor.py --status           # Show portfolio status only
    python trading_advisor.py --update-prices    # Update market prices only
    python trading_advisor.py --portfolio FILE   # Use custom portfolio file
    python trading_advisor.py --timeframes 5min,daily  # Multiple timeframes
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.trading.portfolio_tracker import PortfolioTracker
from src.trading.recommendation_engine import TradingRecommendationEngine
from src.analysis.market_analyzer import DailyMarketAnalyzer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True
    )
    
    # File logging
    log_file = Path("logs") / f"trading_advisor_{datetime.now().strftime('%Y%m%d')}.log"
    log_file.parent.mkdir(exist_ok=True)
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        rotation="1 day",
        retention="30 days"
    )


def show_portfolio_status(portfolio_file: str):
    """Display current portfolio status."""
    try:
        logger.info("Displaying portfolio status")
        tracker = PortfolioTracker(portfolio_file)
        
        if not tracker.portfolio_file.exists():
            print(f"Portfolio file not found: {portfolio_file}")
            print("Run the advisor to create an initial portfolio.")
            return
        
        # Update with latest market prices
        print("Fetching latest market prices...")
        analyzer = DailyMarketAnalyzer()
        market_data = analyzer.fetch_market_data(lookback_days=5)
        
        if market_data:
            latest_prices = {}
            for symbol, data in market_data.items():
                if not data.empty:
                    latest_prices[symbol] = data['close'].iloc[-1]
            
            if latest_prices:
                tracker.update_market_prices(latest_prices)
                tracker.save_portfolio()
                print(f"Updated prices for {len(latest_prices)} stocks.")
        
        # Display portfolio status
        tracker.print_portfolio_status()
        
    except Exception as e:
        logger.error(f"Error showing portfolio status: {e}")
        print(f"Error: {e}")


def update_market_prices_only(portfolio_file: str):
    """Update market prices without generating recommendations."""
    try:
        logger.info("Updating market prices")
        
        tracker = PortfolioTracker(portfolio_file)
        analyzer = DailyMarketAnalyzer()
        
        print("Fetching latest market prices...")
        market_data = analyzer.fetch_market_data(lookback_days=5)
        
        if not market_data:
            print("Failed to fetch market data")
            return
        
        # Extract latest prices
        latest_prices = {}
        for symbol, data in market_data.items():
            if not data.empty:
                latest_prices[symbol] = data['close'].iloc[-1]
                print(f"{symbol}: ${data['close'].iloc[-1]:.2f}")
        
        if latest_prices:
            tracker.update_market_prices(latest_prices)
            tracker.save_portfolio()
            print(f"\\nUpdated prices for {len(latest_prices)} stocks and saved portfolio.")
        else:
            print("No price data to update")
        
    except Exception as e:
        logger.error(f"Error updating prices: {e}")
        print(f"Error: {e}")


def generate_recommendations(portfolio_file: str, timeframes: list):
    """Generate and display trading recommendations."""
    try:
        logger.info(f"Generating recommendations for timeframes: {timeframes}")
        
        print("AI SEMICONDUCTOR TRADING ADVISOR")
        print("=" * 40)
        print("Initializing AI models...")
        
        # Initialize recommendation engine
        engine = TradingRecommendationEngine(portfolio_file)
        
        print("Analyzing market conditions...")
        
        # Generate recommendations
        recommendations = engine.generate_recommendations(timeframes)
        
        # Display formatted report
        report = engine.format_recommendations_report(recommendations)
        print(report)
        
        # Save updated portfolio state
        engine.portfolio_tracker.save_portfolio()
        
        logger.info("Recommendations generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        print(f"\\nError generating recommendations: {e}")
        print("Check the log files for detailed error information.")


def interactive_portfolio_update(portfolio_file: str):
    """Interactive mode to update portfolio with executed trades."""
    try:
        logger.info("Starting interactive portfolio update")
        
        tracker = PortfolioTracker(portfolio_file)
        tracker.print_portfolio_status()
        
        print("\\n" + "=" * 50)
        print("PORTFOLIO UPDATE MODE")
        print("=" * 50)
        print("Update your portfolio after executing trades.")
        print("Type 'help' for commands, 'done' to finish.\\n")
        
        while True:
            try:
                command = input("Command: ").strip().lower()
                
                if command == 'done':
                    break
                elif command == 'help':
                    print("Available commands:")
                    print("  buy SYMBOL SHARES PRICE     - Add new position")
                    print("  sell SYMBOL SHARES          - Sell part/all of position")
                    print("  status                      - Show current portfolio")
                    print("  done                        - Save and exit")
                elif command == 'status':
                    tracker.print_portfolio_status()
                elif command.startswith('buy'):
                    parts = command.split()
                    if len(parts) == 4:
                        _, symbol, shares, price = parts
                        shares = float(shares)
                        price = float(price)
                        if tracker.add_position(symbol.upper(), shares, price):
                            print(f"Added {shares} shares of {symbol.upper()} at ${price:.2f}")
                        else:
                            print("Failed to add position")
                    else:
                        print("Usage: buy SYMBOL SHARES PRICE")
                elif command.startswith('sell'):
                    parts = command.split()
                    if len(parts) == 3:
                        _, symbol, shares = parts
                        shares = float(shares)
                        if tracker.remove_position(symbol.upper(), shares):
                            print(f"Sold {shares} shares of {symbol.upper()}")
                        else:
                            print("Failed to sell position")
                    else:
                        print("Usage: sell SYMBOL SHARES")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Save portfolio
        tracker.save_portfolio()
        print("Portfolio saved successfully.")
        
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
        print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Semiconductor Trading Advisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trading_advisor.py                     # Daily recommendations
  python trading_advisor.py --status            # Portfolio status only
  python trading_advisor.py --update-prices     # Update prices only
  python trading_advisor.py --interactive       # Update portfolio manually
  python trading_advisor.py --timeframes daily,weekly  # Multiple timeframes
        """
    )
    
    parser.add_argument(
        "--portfolio", "-p",
        default="portfolio.toml",
        help="Portfolio file path (default: portfolio.toml)"
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show portfolio status only"
    )
    
    parser.add_argument(
        "--update-prices", "-u",
        action="store_true",
        help="Update market prices only"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode to update portfolio"
    )
    
    parser.add_argument(
        "--timeframes", "-t",
        default="daily",
        help="Comma-separated timeframes (daily,weekly,intraday)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Parse timeframes
        timeframes = [tf.strip() for tf in args.timeframes.split(",")]
        
        # Route to appropriate function
        if args.status:
            show_portfolio_status(args.portfolio)
        elif args.update_prices:
            update_market_prices_only(args.portfolio)
        elif args.interactive:
            interactive_portfolio_update(args.portfolio)
        else:
            generate_recommendations(args.portfolio, timeframes)
            
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user.")
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()