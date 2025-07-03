"""
Algorithme Génétique pour signaux de trading
Pipeline: Unsupervised (clustering/anomalies) → Supervised (prédiction)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import random
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class Individual:
    """Individu dans l'algorithme génétique"""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.fitness = 0.0
        self.genes = self._random_genes()
    
    def _random_genes(self) -> Dict[str, Any]:
        """Génère des gènes aléatoires"""
        
        # Modèles non-supervisés
        unsup_models = ['kmeans', 'dbscan', 'isolation_forest', 'gaussian_mixture']
        unsup_model = random.choice(unsup_models)
        
        unsup_params = {}
        if unsup_model == 'kmeans':
            unsup_params = {'n_clusters': random.randint(3, 15)}
        elif unsup_model == 'dbscan':
            unsup_params = {'eps': random.uniform(0.1, 1.0), 'min_samples': random.randint(3, 10)}
        elif unsup_model == 'isolation_forest':
            unsup_params = {'contamination': random.uniform(0.05, 0.3)}
        elif unsup_model == 'gaussian_mixture':
            unsup_params = {'n_components': random.randint(2, 10)}
        
        # Modèles supervisés
        sup_models = ['xgboost_reg', 'xgboost_clf', 'rf_reg', 'rf_clf', 'ridge', 'logistic']
        sup_model = random.choice(sup_models)
        
        sup_params = {}
        if 'xgboost' in sup_model:
            sup_params = {
                'n_estimators': random.choice([50, 100, 200]),
                'max_depth': random.randint(3, 8),
                'learning_rate': random.uniform(0.01, 0.3),
                'subsample': random.uniform(0.7, 1.0)
            }
        elif 'rf' in sup_model:
            sup_params = {
                'n_estimators': random.choice([50, 100, 200]),
                'max_depth': random.randint(5, 15),
                'min_samples_split': random.randint(2, 10)
            }
        elif sup_model == 'ridge':
            sup_params = {'alpha': random.uniform(0.1, 10.0)}
        elif sup_model == 'logistic':
            sup_params = {'C': random.uniform(0.1, 10.0)}
        
        return {
            'unsup_model': unsup_model,
            'unsup_params': unsup_params,
            'sup_model': sup_model,
            'sup_params': sup_params,
            'features_unsup': np.random.choice([0, 1], size=self.n_features, p=[0.6, 0.4]),
            'features_sup': np.random.choice([0, 1], size=self.n_features, p=[0.5, 0.5]),
            'target': random.choice(['direction_6h', 'direction_12h', 'direction_24h'])
        }
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutation de l'individu"""
        if random.random() < mutation_rate:
            # Mutation des hyperparamètres
            if random.random() < 0.3:
                self.genes = self._random_genes()
            else:
                # Mutation des features
                if random.random() < 0.5:
                    idx = random.randint(0, self.n_features - 1)
                    self.genes['features_unsup'][idx] = 1 - self.genes['features_unsup'][idx]
                else:
                    idx = random.randint(0, self.n_features - 1)
                    self.genes['features_sup'][idx] = 1 - self.genes['features_sup'][idx]
    
    def crossover(self, other: 'Individual') -> 'Individual':
        """Croisement avec un autre individu"""
        child = Individual(self.n_features)
        
        # Croisement des modèles
        child.genes['unsup_model'] = random.choice([self.genes['unsup_model'], other.genes['unsup_model']])
        child.genes['sup_model'] = random.choice([self.genes['sup_model'], other.genes['sup_model']])
        
        # Croisement des paramètres
        if child.genes['unsup_model'] == self.genes['unsup_model']:
            child.genes['unsup_params'] = self.genes['unsup_params'].copy()
        else:
            child.genes['unsup_params'] = other.genes['unsup_params'].copy()
            
        if child.genes['sup_model'] == self.genes['sup_model']:
            child.genes['sup_params'] = self.genes['sup_params'].copy()
        else:
            child.genes['sup_params'] = other.genes['sup_params'].copy()
        
        # Croisement des features (point de coupure aléatoire)
        crossover_point = random.randint(1, self.n_features - 1)
        child.genes['features_unsup'] = np.concatenate([
            self.genes['features_unsup'][:crossover_point],
            other.genes['features_unsup'][crossover_point:]
        ])
        child.genes['features_sup'] = np.concatenate([
            self.genes['features_sup'][:crossover_point],
            other.genes['features_sup'][crossover_point:]
        ])
        
        child.genes['target'] = random.choice([self.genes['target'], other.genes['target']])
        
        return child

class GeneticAlgorithm:
    """Algorithme génétique principal"""
    
    def __init__(self, data: pd.DataFrame, population_size: int = 100, generations: int = 20):
        self.data = data
        self.population_size = population_size
        self.generations = generations
        
        # Préparation des données - exclure variables catégorielles
        excluded_cols = ['future_', 'direction_', 'significant_', 'vol_regime_detailed', 'trend_regime', 'momentum_regime', 'session_phase']
        self.feature_cols = [col for col in data.columns if not any(col.startswith(exc) for exc in excluded_cols) and data[col].dtype in ['float64', 'int64']]
        self.n_features = len(self.feature_cols)
        
        self.population = []
        self.best_individuals = []
        self.fitness_history = []
    
    def _get_unsupervised_model(self, model_type: str, params: Dict):
        """Crée le modèle non-supervisé"""
        if model_type == 'kmeans':
            return KMeans(**params, random_state=42)
        elif model_type == 'dbscan':
            return DBSCAN(**params)
        elif model_type == 'isolation_forest':
            return IsolationForest(**params, random_state=42)
        elif model_type == 'gaussian_mixture':
            return GaussianMixture(**params, random_state=42)
    
    def _get_supervised_model(self, model_type: str, params: Dict):
        """Crée le modèle supervisé"""
        if model_type == 'xgboost_reg':
            return XGBRegressor(**params, random_state=42, verbosity=0)
        elif model_type == 'xgboost_clf':
            return XGBClassifier(**params, random_state=42, verbosity=0)
        elif model_type == 'rf_reg':
            return RandomForestRegressor(**params, random_state=42)
        elif model_type == 'rf_clf':
            return RandomForestClassifier(**params, random_state=42)
        elif model_type == 'ridge':
            return Ridge(**params, random_state=42)
        elif model_type == 'logistic':
            return LogisticRegression(**params, random_state=42, max_iter=2000)
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """Évalue un individu"""
        try:
            genes = individual.genes
            
            # Sélection des features pour unsupervised
            unsup_features = [col for i, col in enumerate(self.feature_cols) if genes['features_unsup'][i]]
            if len(unsup_features) < 3:
                return 0.0
            
            X_unsup = self.data[unsup_features].fillna(0)
            
            # Modèle non-supervisé
            unsup_model = self._get_unsupervised_model(genes['unsup_model'], genes['unsup_params'])
            
            if genes['unsup_model'] == 'isolation_forest':
                anomaly_scores = unsup_model.fit_predict(X_unsup)
                cluster_labels = anomaly_scores
                distances = unsup_model.decision_function(X_unsup)
            elif genes['unsup_model'] == 'dbscan':
                cluster_labels = unsup_model.fit_predict(X_unsup)
                distances = np.zeros(len(cluster_labels))  # DBSCAN n'a pas de distance
            else:
                cluster_labels = unsup_model.fit_predict(X_unsup)
                if hasattr(unsup_model, 'transform'):
                    distances = np.min(unsup_model.transform(X_unsup), axis=1)
                else:
                    distances = np.zeros(len(cluster_labels))
            
            # Features pour le modèle supervisé
            sup_features = [col for i, col in enumerate(self.feature_cols) if genes['features_sup'][i]]
            if len(sup_features) < 3:
                return 0.0
            
            X_sup = self.data[sup_features].fillna(0)
            
            # Ajout des features du clustering
            X_enhanced = X_sup.copy()
            X_enhanced['cluster_label'] = cluster_labels
            X_enhanced['anomaly_distance'] = distances
            X_enhanced['cluster_size'] = X_enhanced['cluster_label'].map(
                X_enhanced['cluster_label'].value_counts()
            )
            
            # Target
            target_col = genes['target']
            if target_col not in self.data.columns:
                return 0.0
            
            y = self.data[target_col].fillna(0)
            
            # Suppression des lignes avec NaN
            valid_idx = ~(X_enhanced.isnull().any(axis=1) | y.isnull())
            X_final = X_enhanced[valid_idx]
            y_final = y[valid_idx]
            
            if len(X_final) < 50:
                return 0.0
            
            # Split train/test temporel (pas aléatoire)
            split_point = int(len(X_final) * 0.7)
            X_train, X_test = X_final[:split_point], X_final[split_point:]
            y_train, y_test = y_final[:split_point], y_final[split_point:]
            
            # Modèle supervisé avec GPU
            sup_model = self._get_supervised_model(genes['sup_model'], genes['sup_params'])
            
            # Activation GPU pour XGBoost
            if 'xgboost' in genes['sup_model']:
                sup_model.set_params(tree_method='gpu_hist', gpu_id=0)
            
            # Classification vs Regression avec pénalité déséquilibre
            if 'clf' in genes['sup_model']:
                # Conversion en classification binaire
                threshold = y_train.std() * 0.5
                y_train_clf = (np.abs(y_train) > threshold).astype(int)
                y_test_clf = (np.abs(y_test) > threshold).astype(int)
                
                sup_model.fit(X_train, y_train_clf)
                y_pred = sup_model.predict(X_test)
                base_fitness = accuracy_score(y_test_clf, y_pred)
            else:
                sup_model.fit(X_train, y_train)
                y_pred = sup_model.predict(X_test)
                
                # Pour direction: pénaliser déséquilibre buy/sell
                if 'direction' in genes['target']:
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    buy_ratio = (y_pred_binary == 1).mean()
                    
                    # Pénalité si trop déséquilibré (optimal: 40-60%)
                    balance_penalty = 1.0
                    if buy_ratio < 0.3 or buy_ratio > 0.7:
                        balance_penalty = 0.8
                    if buy_ratio < 0.2 or buy_ratio > 0.8:
                        balance_penalty = 0.6
                    
                    base_fitness = accuracy_score(y_test.values, y_pred_binary) * balance_penalty
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    base_fitness = 1.0 / (1.0 + mse)
            
            return base_fitness
            
        except Exception as e:
            logger.warning(f"Erreur évaluation: {e}")
            return 0.0
    
    def _initialize_population(self):
        """Initialise la population"""
        self.population = [Individual(self.n_features) for _ in range(self.population_size)]
    
    def _evaluate_population(self):
        """Évalue toute la population"""
        for individual in self.population:
            individual.fitness = self._evaluate_individual(individual)
    
    def _select_best(self, n: int) -> List[Individual]:
        """Sélectionne les n meilleurs individus"""
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n]
    
    def _create_offspring(self, parents: List[Individual], n_offspring: int) -> List[Individual]:
        """Crée des descendants par croisement"""
        offspring = []
        for _ in range(n_offspring):
            parent1, parent2 = random.sample(parents, 2)
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate=0.1)
            offspring.append(child)
        return offspring
    
    def _create_random_individuals(self, n: int) -> List[Individual]:
        """Crée des individus aléatoires"""
        return [Individual(self.n_features) for _ in range(n)]
    
    def evolve(self):
        """Lance l'évolution génétique"""
        logger.info(f"Début évolution: {self.generations} générations, {self.population_size} individus")
        
        # Initialisation
        self._initialize_population()
        
        for generation in range(self.generations):
            # Évaluation
            self._evaluate_population()
            
            # Statistiques
            fitnesses = [ind.fitness for ind in self.population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            logger.info(f"Génération {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Sélection et reproduction
            if generation < self.generations - 1:
                # 20 meilleurs
                best_20 = self._select_best(20)
                
                # 60 descendants
                offspring_60 = self._create_offspring(best_20, 60)
                
                # 20 aléatoires
                random_20 = self._create_random_individuals(20)
                
                # Nouvelle population
                self.population = best_20 + offspring_60 + random_20
        
        # Sauvegarde des meilleurs
        self.best_individuals = self._select_best(10)
        
        logger.info(f"Évolution terminée. Meilleur fitness: {self.best_individuals[0].fitness:.4f}")
        
        return self.best_individuals
    
    def get_best_model(self) -> Tuple[Dict, float]:
        """Retourne le meilleur modèle avec analyse complète"""
        if not self.best_individuals:
            return None, 0.0
        
        best = self.best_individuals[0]
        
        # Feature importance du meilleur modèle
        try:
            genes = best.genes
            unsup_features = [col for i, col in enumerate(self.feature_cols) if genes['features_unsup'][i]]
            sup_features = [col for i, col in enumerate(self.feature_cols) if genes['features_sup'][i]]
            
            print(f"\n🔍 ANALYSE DU MEILLEUR MODÈLE:")
            print(f"Features unsupervised ({len(unsup_features)}): {unsup_features[:5]}...")
            print(f"Features supervised ({len(sup_features)}): {sup_features[:5]}...")
            
        except Exception as e:
            logger.warning(f"Erreur analyse features: {e}")
        
        return best.genes, best.fitness
    
    def save_best_model_config(self):
        """Sauvegarde la configuration du meilleur modèle"""
        if not self.best_individuals:
            return False
            
        best = self.best_individuals[0]
        genes = best.genes
        
        # Conversion des indices numpy en listes Python
        unsup_features_indices = [i for i, selected in enumerate(genes['features_unsup']) if selected]
        sup_features_indices = [i for i, selected in enumerate(genes['features_sup']) if selected]
        
        config = {
            'fitness': best.fitness,
            'unsup_model': genes['unsup_model'],
            'unsup_params': genes['unsup_params'],
            'sup_model': genes['sup_model'],
            'sup_params': genes['sup_params'],
            'features_unsup_indices': unsup_features_indices,
            'features_sup_indices': sup_features_indices,
            'target': genes['target']
        }
        
        import json
        with open('best_model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration sauvegardée: fitness={best.fitness:.4f}")
        return True
    
    def plot_evolution(self):
        """Affiche l'évolution de la fitness"""
        if not self.fitness_history:
            return
            
        generations = [h['generation'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        avg_fitness = [h['avg_fitness'] for h in self.fitness_history]
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution de la Fitness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\n📈 ÉVOLUTION:")
        print(f"Génération 0:  Best={best_fitness[0]:.4f}, Avg={avg_fitness[0]:.4f}")
        print(f"Génération {len(generations)-1}: Best={best_fitness[-1]:.4f}, Avg={avg_fitness[-1]:.4f}")
    def plot_predictions(self):
        """Compare prédictions vs réalité SPY"""
        if not self.best_individuals:
            return
            
        best = self.best_individuals[0]
        genes = best.genes
        
        try:
            # Recréer le modèle avec les meilleures features
            unsup_features = [col for i, col in enumerate(self.feature_cols) if genes['features_unsup'][i]]
            sup_features = [col for i, col in enumerate(self.feature_cols) if genes['features_sup'][i]]
            
            if len(unsup_features) < 3 or len(sup_features) < 3:
                print("⚠️ Pas assez de features pour la visualisation")
                return
                
            X_unsup = self.data[unsup_features].fillna(0)
            
            # Clustering
            unsup_model = self._get_unsupervised_model(genes['unsup_model'], genes['unsup_params'])
            
            if genes['unsup_model'] == 'isolation_forest':
                cluster_labels = unsup_model.fit_predict(X_unsup)
                distances = unsup_model.decision_function(X_unsup)
            else:
                cluster_labels = unsup_model.fit_predict(X_unsup)
                distances = np.zeros(len(cluster_labels))
            
            # Features enhanced
            X_sup = self.data[sup_features].fillna(0)
            X_enhanced = X_sup.copy()
            X_enhanced['cluster_label'] = cluster_labels
            X_enhanced['anomaly_distance'] = distances
            X_enhanced['cluster_size'] = X_enhanced['cluster_label'].map(X_enhanced['cluster_label'].value_counts())
            
            # Target
            target_col = genes['target']
            y = self.data[target_col].fillna(0)
            
            # Nettoyage
            valid_idx = ~(X_enhanced.isnull().any(axis=1) | y.isnull())
            X_final = X_enhanced[valid_idx]
            y_final = y[valid_idx]
            
            # Split temporel
            split_point = int(len(X_final) * 0.7)
            X_train, X_test = X_final[:split_point], X_final[split_point:]
            y_train, y_test = y_final[:split_point], y_final[split_point:]
            
            # Entraînement
            sup_model = self._get_supervised_model(genes['sup_model'], genes['sup_params'])
            sup_model.fit(X_train, y_train)
            y_pred = sup_model.predict(X_test)
            
            # Extraction horizon
            if 'direction' in target_col:
                horizon_str = target_col.split('_')[1]
            elif 'significant' in target_col:
                horizon_str = target_col.split('_')[2]
            else:
                horizon_str = "12h"
            
            horizon = int(horizon_str.replace('h', ''))
            
            # Conversion en binaire si nécessaire
            if hasattr(sup_model, 'predict_proba'):
                y_pred_proba = sup_model.predict_proba(X_test)[:, 1]
                y_pred_binary = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Visualisation
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10))
            
            # Subplot 1: Prédictions vs Réalité
            plt.subplot(2, 1, 1)
            test_times = y_test.index
            plt.plot(test_times, y_test.values, 'b-', label='Réalité', linewidth=2)
            plt.plot(test_times, y_pred_binary, 'r--', label='Prédictions', linewidth=2, alpha=0.8)
            plt.title(f'Prédictions vs Réalité - Direction {horizon}h')
            plt.xlabel('Temps')
            plt.ylabel('Direction (0=Down, 1=Up)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: SPY avec signaux
            plt.subplot(2, 1, 2)
            test_times = y_test.index
            
            # Prix SPY
            if 'Close' in self.data.columns:
                spy_prices = self.data['Close'].reindex(test_times).fillna(method='ffill')
            else:
                spy_prices = pd.Series(index=test_times, data=np.cumsum(np.random.randn(len(test_times)) * 0.01) + 100)
            
            plt.plot(test_times, spy_prices, 'k-', label='SPY', linewidth=1.5)
            
            # Signaux avec décalage
            shifted_times = test_times - pd.Timedelta(hours=horizon)
            
            buy_mask = y_pred_binary == 1
            sell_mask = y_pred_binary == 0
            
            if buy_mask.sum() > 0:
                buy_times = shifted_times[buy_mask]
                buy_prices = spy_prices.iloc[buy_mask]
                plt.scatter(buy_times, buy_prices, color='green', marker='^', s=60, 
                           label=f'Signal Buy ({buy_mask.sum()})', alpha=0.8)
            
            if sell_mask.sum() > 0:
                sell_times = shifted_times[sell_mask]
                sell_prices = spy_prices.iloc[sell_mask]
                plt.scatter(sell_times, sell_prices, color='red', marker='v', s=60, 
                           label=f'Signal Sell ({sell_mask.sum()})', alpha=0.8)
            
            plt.title(f'SPY avec Signaux de Trading (Horizon: {horizon}h)')
            plt.xlabel('Temps')
            plt.ylabel('Prix SPY ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Stats corrigées
            accuracy = (y_pred_binary == y_test.values).mean()
            print(f"\n📊 PERFORMANCE:")
            print(f"Horizon: {horizon}h")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Signaux Buy: {buy_mask.sum()}")
            print(f"Signaux Sell: {sell_mask.sum()}")
            
        except Exception as e:
            logger.warning(f"Erreur visualisation: {e}")
            print("⚠️ Impossible de créer la visualisation")

# Test function
def test_genetic_algorithm():
    """Test de l'algorithme génétique"""
    import sys
    sys.path.append('../data')
    from processors import FeatureProcessor
    from collectors import SPYCollector
    
    # Données
    collector = SPYCollector()
    raw_data = collector.get_hourly_data(period="6mo")
    raw_data = collector.get_technical_indicators(raw_data)
    
    processor = FeatureProcessor()
    processed_data = processor.process_features(raw_data)
    
    if len(processed_data) < 100:
        print("❌ Pas assez de données")
        return
    
    # Algorithme génétique
    ga = GeneticAlgorithm(processed_data, population_size=100, generations=30) 
    best_individuals = ga.evolve()
    
    best_genes, best_fitness = ga.get_best_model()
    ga.save_best_model_config()
    
    print(f"✅ Évolution terminée")
    print(f"🏆 Meilleur fitness: {best_fitness:.4f}")
    print(f"🧬 Meilleur modèle:")
    print(f"   Unsupervised: {best_genes['unsup_model']}")
    print(f"   Supervised: {best_genes['sup_model']}")
    print(f"   Target: {best_genes['target']}")
    
    # Appel externe à trading_simulation
    print("\n🚀 Lancement simulation trading...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "trading_simulation.py"])
    
    return ga

if __name__ == "__main__":
    test_genetic_algorithm()