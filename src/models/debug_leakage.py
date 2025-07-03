"""
Debug data leakage dans l'algorithme génétique
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('../data')
from processors import FeatureProcessor
from collectors import SPYCollector

def debug_leakage():
    """Debug complet pour identifier le data leakage"""
    
    # Récupération des données
    collector = SPYCollector()
    raw_data = collector.get_hourly_data(period="3mo")
    raw_data = collector.get_technical_indicators(raw_data)
    
    processor = FeatureProcessor()
    processed_data = processor.process_features(raw_data)
    
    print(f"Données: {len(processed_data)} échantillons")
    print(f"Colonnes: {len(processed_data.columns)}")
    
    # 1. Vérifier les colonnes utilisées
    excluded_cols = ['future_', 'direction_', 'significant_', 'vol_regime_detailed', 'trend_regime', 'momentum_regime', 'session_phase']
    feature_cols = [col for col in processed_data.columns if not any(col.startswith(exc) for exc in excluded_cols) and processed_data[col].dtype in ['float64', 'int64']]
    
    print(f"\n=== FEATURES UTILISÉES ({len(feature_cols)}) ===")
    for col in feature_cols[:20]:  # Premiers 20
        print(f"  {col}")
    if len(feature_cols) > 20:
        print(f"  ... et {len(feature_cols)-20} autres")
    
    # 2. Vérifier les targets
    target_cols = [col for col in processed_data.columns if col.startswith('future_')]
    print(f"\n=== TARGETS ({len(target_cols)}) ===")
    for col in target_cols:
        print(f"  {col}")
    
    # 3. Corrélations suspectes
    target = 'future_return_24h'
    if target in processed_data.columns:
        correlations = processed_data[feature_cols].corrwith(processed_data[target]).abs().sort_values(ascending=False)
        
        print(f"\n=== TOP 10 CORRÉLATIONS avec {target} ===")
        for feature, corr in correlations.head(10).items():
            print(f"  {feature}: {corr:.3f}")
        
        # Features avec corrélation > 0.9 = suspect
        suspicious = correlations[correlations > 0.9]
        if len(suspicious) > 0:
            print(f"\n⚠️  FEATURES SUSPECTES (corr > 0.9):")
            for feature, corr in suspicious.items():
                print(f"  {feature}: {corr:.3f}")
    
    # 4. Test split temporel
    print(f"\n=== TEST SPLIT TEMPOREL ===")
    split_point = int(len(processed_data) * 0.7)
    train_data = processed_data[:split_point]
    test_data = processed_data[split_point:]
    
    print(f"Train: {train_data.index[0]} → {train_data.index[-1]} ({len(train_data)} points)")
    print(f"Test:  {test_data.index[0]} → {test_data.index[-1]} ({len(test_data)} points)")
    
    # 5. Test simple : prédiction naïve
    if target in processed_data.columns:
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data[target].fillna(0)
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data[target].fillna(0)
        
        # Prédiction par moyenne (baseline)
        y_pred_mean = np.full(len(y_test), y_train.mean())
        mse_baseline = mean_squared_error(y_test, y_pred_mean)
        
        # Prédiction par dernière valeur (random walk)
        y_pred_last = np.full(len(y_test), y_train.iloc[-1])
        mse_last = mean_squared_error(y_test, y_pred_last)
        
        print(f"\n=== BASELINES ===")
        print(f"MSE moyenne: {mse_baseline:.6f}")
        print(f"MSE dernière valeur: {mse_last:.6f}")
        print(f"Fitness baseline (1/(1+MSE)): {1/(1+mse_baseline):.6f}")
        
        # Si notre algo fait mieux que 0.99, c'est suspect
        if 1/(1+mse_baseline) > 0.99:
            print("⚠️  Baseline déjà très élevée - probable data leakage")
    
    # 6. Vérifier les shifts temporels
    print(f"\n=== VÉRIFICATION SHIFTS ===")
    for col in feature_cols[:5]:
        if 'future' not in col:
            # Corrélation avec versions décalées
            for lag in [1, 6, 12, 24]:
                if len(processed_data) > lag:
                    shifted_corr = processed_data[col].corr(processed_data[col].shift(-lag))
                    print(f"  {col} vs {col}_lag{lag}: {shifted_corr:.3f}")
    
    return processed_data, feature_cols

if __name__ == "__main__":
    debug_leakage()