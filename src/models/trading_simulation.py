"""
Simulation trading avec données réelles du dernier algorithme génétique
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../data')
from processors import FeatureProcessor
from collectors import SPYCollector

def get_latest_model_predictions():
    """Récupère prédictions du dernier modèle entraîné"""
    
    # Récupération données identiques à genetic_algo.py
    collector = SPYCollector()
    raw_data = collector.get_hourly_data(period="3mo")
    raw_data = collector.get_technical_indicators(raw_data)
    
    # Prix SPY réels
    spy_prices = raw_data['Close'].copy()
    
    processor = FeatureProcessor()
    processed_data = processor.process_features(raw_data)
    
    # Configuration meilleur modèle (à mettre à jour selon dernier run)
    best_config = {
        'unsup_model': 'gaussian_mixture',
        'unsup_params': {'n_components': 5},
        'sup_model': 'xgboost_reg', 
        'sup_params': {'n_estimators': 100, 'max_depth': 6},
        'features_unsup_indices': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39],  # Exemples
        'features_sup_indices': [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38],
        'target': 'direction_12h'
    }
    
    # Recreation du modèle
    feature_cols = [col for col in processed_data.columns 
                   if not any(col.startswith(exc) for exc in ['future_', 'direction_', 'significant_']) 
                   and processed_data[col].dtype in ['float64', 'int64']]
    
    unsup_features = [feature_cols[i] for i in best_config['features_unsup_indices'] if i < len(feature_cols)]
    sup_features = [feature_cols[i] for i in best_config['features_sup_indices'] if i < len(feature_cols)]
    
    # Modèles
    from sklearn.mixture import GaussianMixture
    from xgboost import XGBRegressor
    
    # Split temporel
    target_col = best_config['target']
    if target_col not in processed_data.columns:
        print(f"❌ Target {target_col} non trouvé")
        return None
        
    valid_data = processed_data.dropna()
    split_point = int(len(valid_data) * 0.7)
    
    train_data = valid_data[:split_point]
    test_data = valid_data[split_point:]
    
    # Unsupervised
    X_unsup_train = train_data[unsup_features].fillna(0)
    X_unsup_test = test_data[unsup_features].fillna(0)
    
    unsup_model = GaussianMixture(**best_config['unsup_params'])
    cluster_labels_test = unsup_model.fit(X_unsup_train).predict(X_unsup_test)
    
    # Supervised  
    X_sup_train = train_data[sup_features].fillna(0)
    X_sup_test = test_data[sup_features].fillna(0)
    
    # Ajout features clustering
    X_sup_train_enhanced = X_sup_train.copy()
    X_sup_train_enhanced['cluster'] = unsup_model.predict(train_data[unsup_features].fillna(0))
    
    X_sup_test_enhanced = X_sup_test.copy() 
    X_sup_test_enhanced['cluster'] = cluster_labels_test
    
    y_train = train_data[target_col]
    y_test = test_data[target_col]
    
    sup_model = XGBRegressor(**best_config['sup_params'], verbosity=0)
    sup_model.fit(X_sup_train_enhanced, y_train)
    y_pred = sup_model.predict(X_sup_test_enhanced)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Prix SPY correspondants
    spy_test = spy_prices.reindex(test_data.index).ffill()
    
    return {
        'predictions': y_pred_binary,
        'reality': y_test.values,
        'spy_prices': spy_test,
        'dates': test_data.index,
        'horizon': int(target_col.split('_')[1].replace('h', ''))
    }

def simulate_real_strategy(real_data, initial_capital=10000):
    """Simulation stratégie avec données réelles"""
    
    predictions = real_data['predictions']
    spy_prices = real_data['spy_prices']
    dates = real_data['dates']
    horizon = real_data['horizon']
    
    print(f"📊 SIMULATION STRATÉGIE RÉELLE")
    print(f"Période: {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Horizon: {horizon}h")
    
    # Performance SPY
    spy_return = (spy_prices.iloc[-1] - spy_prices.iloc[0]) / spy_prices.iloc[0]
    buy_hold = initial_capital * (1 + spy_return)
    
    # Stratégie signaux
    signals_buy = (predictions == 1).sum()
    signals_sell = (predictions == 0).sum()
    buy_ratio = signals_buy / len(predictions)
    
    # Allocation dynamique basée sur signaux
    avg_allocation = 0.5 + (buy_ratio - 0.5) * 0.6  # 20% à 80%
    strategy_return = avg_allocation * spy_return + (1 - avg_allocation) * 0.02  # 2% cash
    strategy_value = initial_capital * (1 + strategy_return)
    
    print(f"SPY: ${spy_prices.iloc[0]:.2f} → ${spy_prices.iloc[-1]:.2f} ({spy_return:+.2%})")
    print(f"Signaux: {signals_buy} buy ({buy_ratio:.1%}), {signals_sell} sell")
    print(f"Buy & Hold: ${buy_hold:,.0f} ({spy_return:+.2%})")
    print(f"Stratégie: ${strategy_value:,.0f} ({strategy_return:+.2%})")
    
    # Performance signaux
    if signals_buy > 0:
        buy_mask = predictions == 1
        future_prices = spy_prices.shift(-horizon).ffill()
        buy_performance = (future_prices[buy_mask] / spy_prices[buy_mask] - 1).mean()
        print(f"Avg return {horizon}h après BUY: {buy_performance:+.2%}")
    
    # Graphique
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2,1,1)
    plt.plot(dates, spy_prices, 'k-', linewidth=1.5, label='SPY')
    
    buy_dates = dates[predictions == 1]
    sell_dates = dates[predictions == 0]
    
    if len(buy_dates) > 0:
        plt.scatter(buy_dates, spy_prices.reindex(buy_dates), 
                   color='green', marker='^', s=30, alpha=0.7, label=f'Buy ({len(buy_dates)})')
    if len(sell_dates) > 0:
        plt.scatter(sell_dates, spy_prices.reindex(sell_dates),
                   color='red', marker='v', s=30, alpha=0.7, label=f'Sell ({len(sell_dates)})')
    
    plt.title('SPY avec Signaux Trading')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2,1,2)
    cumulative_strategy = []
    cumulative_spy = []
    
    for i in range(len(dates)):
        spy_perf = (spy_prices.iloc[i] / spy_prices.iloc[0] - 1)
        strategy_perf = avg_allocation * spy_perf
        
        cumulative_spy.append(initial_capital * (1 + spy_perf))
        cumulative_strategy.append(initial_capital * (1 + strategy_perf))
    
    plt.plot(dates, cumulative_spy, 'b-', label='Buy & Hold', linewidth=1.5)
    plt.plot(dates, cumulative_strategy, 'r--', label='Stratégie Signaux', linewidth=1.5)
    plt.title('Performance Cumulée')
    plt.ylabel('Valeur Portfolio ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'spy_return': spy_return,
        'strategy_return': strategy_return,
        'buy_ratio': buy_ratio
    }

if __name__ == "__main__":
    print("🤖 Simulation avec dernier modèle génétique...")
    
    real_data = get_latest_model_predictions()
    if real_data:
        results = simulate_real_strategy(real_data)
        
        if results['strategy_return'] > results['spy_return']:
            print("✅ Stratégie surperforme Buy & Hold")
        else:
            print("❌ Stratégie sous-performe Buy & Hold")
    else:
        print("❌ Erreur récupération données")