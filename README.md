# Statistical Arbitrage Engine: Sector ETF Pairs Trading

## Project Overview
This project implements a **Statistical Arbitrage (StatArb)** trading strategy focused on the US Equity ETF market. It utilises a **Mean Reversion** framework to identify and exploit short-term pricing inefficiencies between a target sector (Financials) and a correlated basket of hedge assets.

Using **Rolling Linear Regression**, the engine dynamically calculates hedge ratios (Betas) to construct a "synthetic" fair-value price. When the actual market price deviates significantly from this synthetic price (measured via Z-Scores), the algorithm executes delta-neutral trades to capture the spread.

## Key Features
* **Dynamic Hedging:** Calculates real-time hedge ratios using a 126-day (6-month) rolling window to adapt to changing market correlation regimes.
* **Factor Modeling:** Replicates the returns of the target asset (VFH) using a multi-factor regression model against correlated sectors (VGT, VDC, VIS, VCR).
* **Z-Score Signal Generation:** Normalises price spreads to identify statistically significant entry ($\pm 1.25 \sigma$) and exit ($\pm 0.75 \sigma$) points.
* **Risk Management (Stop-Loss):** Automatically closes positions if the Z-Score breaches extreme levels ($\pm 3.0 \sigma$) to prevent significant capital drawdowns during regime shifts.
* **Vectorized Backtesting:** Simulates historical performance and tracks cumulative P&L based on generated signals.

## The Quantitative Strategy

### 1. The Hypothesis
The "Law of One Price" suggests that assets with similar fundamental drivers should move in unison. By modeling the Financials sector (**VFH**) as a linear combination of Technology, Industrials, and Consumer goods, we can isolate a stationary spread.

### 2. Mathematical Model
We assume the target price $Y_t$ can be modeled as:

$$Y_t = \beta_1 X_{1,t} + \beta_2 X_{2,t} + \dots + \beta_n X_{n,t} + \epsilon_t$$

Where:
* $Y_t$: Price of Target Asset (VFH)
* $X_{i,t}$: Price of Hedge Assets (Basket)
* $\beta_i$: Calculated Hedge Ratios (Sensitivity)
* $\epsilon_t$: The Residual (Spread)

### 3. Execution Logic (Mean Reversion)
The strategy tracks the Z-Score of the residual $\epsilon_t$.

* **Long Entry:** If $Z < -1.25$ (Target is undervalued), we Buy Target and Short Basket.
* **Short Entry:** If $Z > 1.25$ (Target is overvalued), we Short Target and Buy Basket.
* **Take Profit:** Trades are closed when $Z$ reverts to mean levels ($\pm 0.75$).
* **Stop Loss:** Trades are forcibly closed if $Z$ expands beyond $\pm 3.0$.

## Performance Visualization
The system outputs a dual-plot analysis:
1.  **Z-Score Monitor:** Visualizing the spread oscillating between statistical bounds, including the new Stop-Loss thresholds.
2.  **Cumulative Wealth:** Tracking the growth of capital over the backtest period.

## Limitations & Risks
While the strategy demonstrates the potential for high returns during mean-reverting regimes, it carries significant risks and simplifications:

* **Regime Changes (Stationarity Risk):** The model assumes the spread is mean-reverting. However, structural market shifts can cause the spread to drift away from the mean for extended periods.
* **Execution Assumptions:** The backtest assumes trades are executed instantly at the closing price of the signal day. In live trading, slippage and bid-ask spreads would erode margins.
* **Transaction Costs:** The current simulation is "Gross P&L." It does not account for brokerage commissions or the cost of borrowing required to short sell the hedge assets.
* **Look-Ahead Bias:** While the regression uses a rolling window to prevent data leakage, the signal generation uses the current day's close to determine entry, which is theoretically executable only on the next day's open.

## Conclusion
This project successfully demonstrates the mechanics of Statistical Arbitrage and Pairs Trading. It highlights the power of using correlation matrices to construct synthetic assets and the profitability of exploiting short-term market inefficiencies. However, the volatility observed in the equity curve underscores the necessity of robust risk management protocols when trading mean-reversion strategies in trending markets.

## Tech Stack
* **Python:** Core logic and data processing.
* **Pandas/NumPy:** Vectorized calculations and time-series manipulation.
* **Scikit-Learn:** `LinearRegression` for calculating rolling betas.
* **YFinance:** Data ingestion pipeline.
* **Matplotlib:** Visualization of Z-Scores and Equity Curves.

## Installation & Usage

Clone the repository:
```bash
git clone https://github.com/yourusername/sports-arb-engine.git

cd stat-arb-engine
```

Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the simulation:
```bash
python stat_arb_bot.py
```