Quantitative Trading Algorithm
Overview
Advanced quantitative trading strategy using genetic algorithms and unsupervised learning to detect weak signals in financial markets.
Workflow
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │────│ Feature Engineer │────│ Genetic Algo    │
│   (SPY ETF)     │    │ (72+ indicators) │    │ (ML Pipeline)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│ Trading Signals │◄───│ Performance Test │◄────────────┘
│ (Buy/Sell)      │    │ (Real SPY Data)  │
└─────────────────┘    └──────────────────┘
Features

Hourly SPY data analysis
Multi-horizon predictions (6h, 12h, 24h)
Genetic algorithm optimization
Automated performance testing

Installation
bashgit clone https://github.com/[username]/quant-trading-algo
cd quant-trading-algo
pip install -r requirements.txt
Usage
bash# Train genetic algorithm & test performance
python src/models/genetic_algo.py

# Test saved model separately  
python src/models/trading_simulation.py

# Debug data leakage
python src/models/debug_leakage.py
Project Structure
quant-trading-algo/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variables template
├── config.yaml                 # Configuration
├── data/                       # Data storage (ignored by git)
├── src/                        # Source code
│   ├── data/
│   │   ├── collectors.py       # SPY data collection
│   │   └── processors.py       # Feature engineering
│   ├── models/
│   │   ├── genetic_algo.py     # Main algorithm
│   │   ├── trading_simulation.py # Performance testing
│   │   └── debug_leakage.py    # Data validation
│   ├── utils/                  # Helper functions
│   └── visualization/          # Plots and charts
├── notebooks/                  # Jupyter analysis notebooks
├── tests/                      # Unit tests
├── scripts/                    # Execution scripts
├── results/                    # Output and reports
└── docs/                       # Documentation