import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

TARGET_ASSET = "VFH"  # Vanguard Financials ETF
HEDGE_ASSETS = ["VDC", "VCR", "VIS", "VGT"]  # The Basket
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 126  # 6-Month Rolling Window for Regression

class StatArbBot:
    def __init__(self):
        self.df = None
        self.betas = None
        self.z_scores = None
        self.signals = None
        self.wealth = []

    def fetch_data(self):
        print(f"Downloading data for {TARGET_ASSET} + Basket: {HEDGE_ASSETS}...")
        tickers = [TARGET_ASSET] + HEDGE_ASSETS
        
        data = yf.download(tickers, start=START_DATE, end=END_DATE)['Close']
        
        self.df = data.dropna()
        print(f"Data fetched: {len(self.df)} trading days.")

    def run_rolling_regression(self):
        print("Running Rolling Linear Regression (Dynamic Hedging)...")
        
        rolling_betas = {ticker: [] for ticker in HEDGE_ASSETS}
        rolling_dates = []

        # Iterate through the data with a moving window
        for start in range(len(self.df) - WINDOW_SIZE + 1):
            end = start + WINDOW_SIZE
            
            # 1. Slice the Window
            y_window = self.df[TARGET_ASSET].iloc[start:end]
            x_window = self.df[HEDGE_ASSETS].iloc[start:end]

            # 2. Fit Linear Regression
            model = LinearRegression()
            model.fit(x_window, y_window)
            
            # 3. Store the Betas 
            for i, ticker in enumerate(HEDGE_ASSETS):
                rolling_betas[ticker].append(model.coef_[i])
            
            # Store the date corresponding to the end of this window
            rolling_dates.append(self.df.index[end - 1])

        # Convert dictionary to DataFrame
        self.betas = pd.DataFrame(rolling_betas, index=rolling_dates)
        
        # Align original DF to match the dates where we have betas
        self.df = self.df.loc[self.betas.index]
        print("Betas calculated.")

    def generate_signals(self):
        print("Calculating Z-Scores and Signals...")
        
        # 1. Calculate the 'Predicted' Price of VFH based on the Hedge Assets
        predicted_price = sum(self.betas[asset] * self.df[asset] for asset in HEDGE_ASSETS)
        
        # 2. Calculate Residuals 
        residuals = self.df[TARGET_ASSET] - predicted_price
        
        # 3. Calculate Z-Score 
        self.z_scores = (residuals - residuals.mean()) / residuals.std()

        # 4. Generate Signals logic
        signals_list = []
        prior_z = None
        
        # State machine for signals
        current_signal = None 
        
        for z in self.z_scores:
            if prior_z is None:
                prior_z = z
                signals_list.append("HOLD")
                continue
            
            # --- RISK MANAGEMENT: STOP LOSS ---
            if current_signal == "OPEN SHORT" and z >= 3.0:
                action = "CLOSE SHORT"
                current_signal = None
                
            # If we are Long and Z crashes < -3.0, cut losses.
            elif current_signal == "OPEN LONG" and z <= -3.0:
                action = "CLOSE LONG"
                current_signal = None
            
            # --- STANDARD MEAN REVERSION LOGIC ---
            # ENTRY LOGIC (Short side)
            elif prior_z >= 1.25 and z <= 1.25 and current_signal != "OPEN SHORT":
                action = "OPEN SHORT"
                current_signal = action
            
            # EXIT LOGIC
            elif prior_z >= 0.75 and z <= 0.75 and current_signal == "OPEN SHORT":
                action = "CLOSE SHORT"
                current_signal = None
                
            # ENTRY LOGIC (Long side)
            elif prior_z <= -1.25 and z >= -1.25 and current_signal != "OPEN LONG":
                action = "OPEN LONG"
                current_signal = action
                
            # EXIT LOGIC
            elif prior_z <= -0.75 and z >= -0.75 and current_signal == "OPEN LONG":
                action = "CLOSE LONG"
                current_signal = None
                
            else:
                action = "HOLD"
                
            signals_list.append(action)
            prior_z = z
            
        self.signals = pd.Series(signals_list, index=self.z_scores.index)
        print("Signals generated.")

    def backtest(self):
        print("Running Backtest...")
        wealth = [1.0] # Start with 100% capital
        
        # Current Holdings: {Asset: Quantity}
        positions = {asset: 0 for asset in [TARGET_ASSET] + HEDGE_ASSETS}
        
        # Iterate day by day
        for i in range(1, len(self.df)):
            # signal = self.signals.iloc[i] # Warning: Look-ahead bias if we use today's signal for today's price move
            # Ideally, we trade on the NEXT open. 
            # For simplicity in this demo, we assume we can execute on Close if signal triggers.
            
            current_signal = self.signals.iloc[i]
            
            # 1. Calculate P&L from yesterday's positions
            daily_pnl = 0
            for asset, pos in positions.items():
                price_change = self.df[asset].iloc[i] - self.df[asset].iloc[i-1]
                daily_pnl += pos * price_change
            
            # Update Wealth
            wealth.append(wealth[-1] + daily_pnl)
            
            # 2. Update Positions based on Signal
            if current_signal == "OPEN SHORT":
                positions[TARGET_ASSET] = -1
                for asset in HEDGE_ASSETS:
                    positions[asset] = self.betas[asset].iloc[i]
                    
            elif current_signal == "OPEN LONG":
                positions[TARGET_ASSET] = 1
                for asset in HEDGE_ASSETS:
                    positions[asset] = -self.betas[asset].iloc[i]
                    
            elif current_signal == "CLOSE SHORT" or current_signal == "CLOSE LONG":
                positions = {asset: 0 for asset in [TARGET_ASSET] + HEDGE_ASSETS}

        self.wealth = pd.Series(wealth, index=self.df.index)
        print(f"Backtest Complete. Final Wealth Multiplier: {self.wealth.iloc[-1]:.4f}")

    def plot_results(self):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Z-Score and Thresholds
        axes[0].plot(self.z_scores, label="Z-Score (Spread)", color='blue', alpha=0.7)
        axes[0].axhline(3.0, color='black', linestyle=':', linewidth=2, label="Stop Loss (+3.0)")
        axes[0].axhline(1.25, color='red', linestyle='--', label="Short Threshold (+1.25)")
        axes[0].axhline(0.75, color='orange', linestyle='--', label="Close Short (+0.75)")
        axes[0].axhline(-1.25, color='green', linestyle='--', label="Long Threshold (-1.25)")
        axes[0].axhline(-0.75, color='lime', linestyle='--', label="Close Long (-0.75)")
        axes[0].axhline(-3.0, color='black', linestyle=':', linewidth=2, label="Stop Loss (-3.0)")
        axes[0].set_title(f"Mean Reversion Signal (Target: {TARGET_ASSET})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Strategy Performance
        axes[1].plot(self.wealth, label="Strategy Wealth", color='purple')
        axes[1].set_title("Cumulative P&L (Beta Neutral)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stat_arb_results.png')
        print("Results saved as 'stat_arb_results.png'")

if __name__ == "__main__":
    bot = StatArbBot()
    bot.fetch_data()
    bot.run_rolling_regression()
    bot.generate_signals()
    bot.backtest()
    bot.plot_results()