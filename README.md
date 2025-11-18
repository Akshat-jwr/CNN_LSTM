# CNN-LSTM Trading Strategy for Infosys (INFY)

## Overview

This repository contains two variations of a hybrid **CNN-LSTM deep learning model** for predicting profitable trading opportunities in Infosys (INFY) stock on the National Stock Exchange (NSE) of India. The models combine Convolutional Neural Networks (CNN) for temporal feature extraction with Long Short-Term Memory (LSTM) networks for capturing long-term dependencies in financial time series data.

## Models

### Model 1: Next-Day Direction Prediction (`Infy-CNN_LSTM.ipynb`)

**Target Variable**: Binary classification predicting whether the next day's return will be positive.
- **Label**: `1` if next day's log return > 0, else `0`
- **Prediction Horizon**: 1 day ahead

**Use Case**: Traditional directional trading strategy suitable for daily momentum-based approaches.

### Model 2: Peak Return Prediction (`Infy-CNN_LSTM(1).ipynb`)

**Target Variable**: Binary classification predicting whether a profitable exit point exists within the next 6 days.
- **Label**: `1` if maximum return (close-to-high) in next 6 days exceeds 3%, else `0`
- **Forward Window**: 6 days
- **Profit Threshold**: 3%

**Use Case**: Swing trading strategy focused on identifying entry points with high probability of reaching profit targets.

## Model Architecture

Both models use an identical **CNN-LSTM hybrid architecture**:

```
Input Layer (28 timesteps × 27 features)
    ↓
Conv1D Layer (64 filters, kernel=3, causal padding, ReLU)
    ↓
Batch Normalization
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (30%)
    ↓
LSTM Layer (64 units, dropout=30%, recurrent_dropout=10%)
    ↓
Batch Normalization
    ↓
Dropout (30%)
    ↓
Dense Output (1 unit, sigmoid activation)
```

### Architecture Components

1. **Convolutional Layer (CNN)**:
   - Extracts local temporal patterns from price sequences
   - Uses causal padding to prevent look-ahead bias
   - 64 filters with kernel size 3 capture short-term patterns

2. **LSTM Layer**:
   - Models long-term dependencies and market regime changes
   - 64 hidden units maintain sequence memory
   - Dropout layers prevent overfitting

3. **Regularization**:
   - Batch normalization after CNN and LSTM layers
   - Multiple dropout layers (30% standard, 10% recurrent)
   - Early stopping and learning rate reduction callbacks

## Features (27 Total)

### Price-Based Features
- **log_returns**: Daily logarithmic returns
- **ret_3d, ret_5d, ret_10d**: Multi-horizon returns (3, 5, 10 days)
- **lag_1 to lag_7**: Lagged returns for momentum patterns

### Technical Indicators
- **RSI_14**: Relative Strength Index (14-day)
- **MACD**: MACD histogram (12,26,9 configuration)
- **MACD_line, MACD_signal**: MACD components
- **OBV**: On-Balance Volume

### Volatility Metrics
- **volatility_10**: 10-day rolling standard deviation
- **vol_10d, vol_21d, vol_63d**: Annualized volatility (10, 21, 63 days)
- **vol_ratio**: Short-term to long-term volatility ratio
- **ATR_14**: Average True Range (14-day)

### Higher-Order Statistics
- **ret_skew_21**: 21-day return skewness
- **ret_kurt_21**: 21-day return kurtosis

### Volume Indicators
- **vol_zscore_20**: 20-day volume z-score

## Data Processing

### Data Source
- **Stock**: Infosys (INFY) from NSE
- **Period**: January 2005 - December 2024 (20 years)
- **Frequency**: Daily OHLCV data
- **Source**: NSEPython library

### Adjustments
Historical prices are adjusted for **corporate actions** (1:1 bonus issues):
- 2018-09-04
- 2015-06-15
- 2014-12-02
- 2006-07-13

### Sequence Construction
- **Sequence Length**: 28 days (approximately one trading month)
- **Rolling Windows**: Each prediction uses the previous 28 days of normalized features
- **Scaling**: StandardScaler fitted on training data, applied to validation/test sets

## Training Methodology

### Cross-Validation
- **Strategy**: TimeSeriesSplit with 6 folds
- **Test Size**: 252 days (1 trading year) per fold
- **Purpose**: Prevents data leakage and simulates walk-forward testing

### Model Training
- **Optimizer**: Adam (initial learning rate: 0.001)
- **Loss Function**: Binary crossentropy
- **Batch Size**: 64
- **Max Epochs**: 20 (CV), 50 (final model)
- **Callbacks**:
  - EarlyStopping (patience=5, monitor=val_loss)
  - ReduceLROnPlateau (factor=0.5, patience=3)

### Out-of-Sample (OOS) Testing
- **Train Period**: All data before 2022-01-01
- **Test Period**: 2022-01-01 onwards
- **Purpose**: Final model validation on completely unseen data

## Backtesting Framework

### Signal Generation
Predictions (probabilities) are converted to trading signals using dynamic thresholds:
- **Long Signal**: Prediction ≥ threshold + 0.02
- **Short Signal**: Prediction ≤ threshold - 0.02
- **Neutral**: Otherwise (if neutral band enabled)

### Threshold Optimization
A grid search over threshold values (mean ± 0.10) optimizes for maximum Sharpe ratio on cross-validation data.

### Risk Management

1. **Position Holding**:
   - Minimum hold period: 0-1 days (model dependent)
   
2. **Stop-Loss Mechanisms**:
   - **Daily Stop**: Exit if single-day loss exceeds -2%
   - **Rolling Stop**: Exit if 10-day cumulative loss exceeds -5%

3. **Volatility Targeting**:
   - Target annual volatility: 12%
   - Maximum leverage: 2.0x
   - Position sizing scales inversely with realized volatility

4. **Transaction Costs**:
   - **Cost**: 5 basis points (0.05%) per turnover
   - Applied on position changes to simulate realistic execution

### Performance Metrics
- **Total Return & CAGR**: Annualized compounded returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Hit Rate**: Percentage of profitable trades
- **Turnover**: Number of position changes
- **Comparison**: All metrics computed for both strategy and buy-and-hold benchmark

## Key Differences Between Models

| Aspect | Model 1 (Next-Day) | Model 2 (Peak Return) |
|--------|-------------------|----------------------|
| **Target** | Next-day return direction | 3% profit within 6 days |
| **Horizon** | 1 day | 6 days |
| **Trading Style** | Day/swing trading | Swing/position trading |
| **Signal Frequency** | Higher turnover | Lower turnover |
| **Min Hold** | 0 days | 1 day |
| **Label Balance** | ~50/50 (market dependent) | Skewed (fewer profit opportunities) |

## Results & Outputs

Both notebooks generate:
1. **Cross-validation performance** by fold (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
2. **Threshold sweep analysis** with Sharpe ratio visualization
3. **Out-of-sample backtest summary** comparing strategy vs. buy-and-hold
4. **Equity curve plots** showing cumulative performance
5. **Saved artifacts**:
   - `cv_predictions.npy`: Cross-validation predictions
   - `cv_labels.npy`: True labels
   - `threshold_sweep_cv.csv`: Threshold optimization results
   - `backtest_oos_summary.csv`: Final OOS performance
   - Equity curve plots (PNG)

## Requirements

### Python Libraries
```
numpy
pandas
matplotlib
scikit-learn
tensorflow / keras
nsepython
pandas_ta
quantstats
tabulate
```

### Environment Setup
- **Random Seed**: 42 (for reproducibility)
- **TensorFlow Determinism**: Enabled via environment variables

## Usage

1. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow nsepython pandas-ta quantstats tabulate
   ```

2. **Run notebooks sequentially**:
   - Model 1: `Infy-CNN_LSTM.ipynb`
   - Model 2: `Infy-CNN_LSTM(1).ipynb`

3. **Outputs saved to**: `./INFY_CNN_LSTM_Results/`

## Future Improvements

- **Multi-asset extension**: Apply to portfolio of stocks
- **Ensemble methods**: Combine predictions from both models
- **Feature engineering**: Add market regime indicators, sentiment data
- **Hyperparameter tuning**: Grid search over CNN/LSTM architectures
- **Alternative targets**: Continuous value prediction (expected return)
- **Real-time deployment**: Integrate with live data feeds

## Disclaimer

These models are for **educational and research purposes only**. Past performance does not guarantee future results. Always conduct thorough due diligence and consider transaction costs, slippage, and market impact before deploying any trading strategy in live markets.

## License

MIT License - Feel free to use and modify for your research and trading experiments.

---

**Author**: Akshat Jiwrajka  
**Last Updated**: November 2025
# CNN_LSTM
