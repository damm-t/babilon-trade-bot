"""
Weight Tuning Module for Phase 5
Implements grid search and optimization for hybrid signal weights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from itertools import product
from dataclasses import dataclass
import json
from datetime import datetime

from .hybrid_signal import HybridSignalEngine, HybridSignal
from .backtest import BacktestEngine

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of weight optimization"""
    best_weights: Dict[str, float]
    best_score: float
    best_sharpe: float
    best_returns: float
    best_drawdown: float
    optimization_history: List[Dict]
    total_combinations: int
    execution_time: float


class WeightTuner:
    """Weight tuning optimizer for hybrid signals"""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
        self.optimization_history = []
        
    def grid_search_weights(self, 
                          feature_data: pd.DataFrame,
                          price_data: pd.DataFrame,
                          weight_ranges: Optional[Dict[str, List[float]]] = None,
                          optimization_metric: str = 'sharpe_ratio') -> OptimizationResult:
        """Perform grid search optimization for weights"""
        
        if weight_ranges is None:
            weight_ranges = self._get_default_weight_ranges()
        
        logger.info(f"Starting grid search optimization with {optimization_metric} metric")
        logger.info(f"Weight ranges: {weight_ranges}")
        
        start_time = datetime.now()
        best_score = -np.inf
        best_weights = None
        best_results = None
        total_combinations = 0
        
        # Generate all weight combinations
        weight_names = list(weight_ranges.keys())
        weight_values = list(weight_ranges.values())
        
        for weight_combination in product(*weight_values):
            total_combinations += 1
            if total_combinations % 100 == 0:
                logger.info(f"Processed {total_combinations} combinations...")
            
            # Create weight dictionary
            weights = dict(zip(weight_names, weight_combination))
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Test this weight combination
            try:
                results = self._test_weight_combination(weights, feature_data, price_data)
                
                # Store optimization history
                optimization_record = {
                    'weights': weights.copy(),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'total_return': results.get('total_return', 0),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'win_rate': results.get('win_rate', 0),
                    'profit_factor': results.get('profit_factor', 0)
                }
                self.optimization_history.append(optimization_record)
                
                # Check if this is the best combination
                current_score = results.get(optimization_metric, 0)
                if current_score > best_score:
                    best_score = current_score
                    best_weights = weights.copy()
                    best_results = results
                    logger.info(f"New best score: {best_score:.4f} with weights: {weights}")
                
            except Exception as e:
                logger.warning(f"Error testing weights {weights}: {e}")
                continue
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Grid search completed: {total_combinations} combinations in {execution_time:.2f}s")
        logger.info(f"Best {optimization_metric}: {best_score:.4f}")
        logger.info(f"Best weights: {best_weights}")
        
        return OptimizationResult(
            best_weights=best_weights or {},
            best_score=best_score,
            best_sharpe=best_results.get('sharpe_ratio', 0) if best_results else 0,
            best_returns=best_results.get('total_return', 0) if best_results else 0,
            best_drawdown=best_results.get('max_drawdown', 0) if best_results else 0,
            optimization_history=self.optimization_history.copy(),
            total_combinations=total_combinations,
            execution_time=execution_time
        )
    
    def _get_default_weight_ranges(self) -> Dict[str, List[float]]:
        """Get default weight ranges for optimization"""
        return {
            'ml_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
            'rule_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
            'sentiment_weight': [0.2, 0.3, 0.4, 0.5],
            'momentum_weight': [0.1, 0.2, 0.3, 0.4],
            'volatility_weight': [0.05, 0.1, 0.15, 0.2],
            'volume_weight': [0.05, 0.1, 0.15, 0.2],
            'trend_weight': [0.2, 0.3, 0.4, 0.5]
        }
    
    def _test_weight_combination(self, weights: Dict[str, float], 
                               feature_data: pd.DataFrame, 
                               price_data: pd.DataFrame) -> Dict[str, float]:
        """Test a specific weight combination"""
        
        # Create hybrid signal engine with these weights
        config = {
            'ml_weight': weights.get('ml_weight', 0.6),
            'rule_weight': weights.get('rule_weight', 0.4),
            'sentiment_weight': weights.get('sentiment_weight', 0.3),
            'momentum_weight': weights.get('momentum_weight', 0.2),
            'volatility_weight': weights.get('volatility_weight', 0.1),
            'volume_weight': weights.get('volume_weight', 0.1),
            'trend_weight': weights.get('trend_weight', 0.3)
        }
        
        engine = HybridSignalEngine(config)
        
        # Run backtest with this configuration
        results = self.backtest_engine.run_backtest(
            feature_data=feature_data,
            price_data=price_data,
            signal_engine=engine,
            initial_capital=10000,
            commission=0.001
        )
        
        return results
    
    def optimize_with_genetic_algorithm(self, 
                                       feature_data: pd.DataFrame,
                                       price_data: pd.DataFrame,
                                       population_size: int = 50,
                                       generations: int = 20,
                                       mutation_rate: float = 0.1) -> OptimizationResult:
        """Optimize weights using genetic algorithm"""
        
        logger.info(f"Starting genetic algorithm optimization: {population_size} individuals, {generations} generations")
        
        start_time = datetime.now()
        
        # Initialize population
        population = self._initialize_population(population_size)
        best_individual = None
        best_score = -np.inf
        
        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")
            
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                try:
                    results = self._test_weight_combination(individual, feature_data, price_data)
                    fitness = results.get('sharpe_ratio', 0)
                    fitness_scores.append(fitness)
                    
                    if fitness > best_score:
                        best_score = fitness
                        best_individual = individual.copy()
                        logger.info(f"New best fitness: {fitness:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Error evaluating individual: {e}")
                    fitness_scores.append(-np.inf)
            
            # Store generation results
            generation_record = {
                'generation': generation + 1,
                'best_fitness': max(fitness_scores) if fitness_scores else 0,
                'avg_fitness': np.mean(fitness_scores) if fitness_scores else 0,
                'best_individual': best_individual
            }
            self.optimization_history.append(generation_record)
            
            # Create next generation
            if generation < generations - 1:  # Don't evolve the last generation
                population = self._evolve_population(population, fitness_scores, mutation_rate)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Get final results for best individual
        final_results = self._test_weight_combination(best_individual, feature_data, price_data)
        
        logger.info(f"Genetic algorithm completed in {execution_time:.2f}s")
        logger.info(f"Best fitness: {best_score:.4f}")
        logger.info(f"Best individual: {best_individual}")
        
        return OptimizationResult(
            best_weights=best_individual or {},
            best_score=best_score,
            best_sharpe=final_results.get('sharpe_ratio', 0),
            best_returns=final_results.get('total_return', 0),
            best_drawdown=final_results.get('max_drawdown', 0),
            optimization_history=self.optimization_history.copy(),
            total_combinations=population_size * generations,
            execution_time=execution_time
        )
    
    def _initialize_population(self, population_size: int) -> List[Dict[str, float]]:
        """Initialize random population of weight combinations"""
        population = []
        
        for _ in range(population_size):
            # Generate random weights
            weights = {
                'ml_weight': np.random.uniform(0.2, 0.8),
                'rule_weight': np.random.uniform(0.2, 0.8),
                'sentiment_weight': np.random.uniform(0.1, 0.5),
                'momentum_weight': np.random.uniform(0.05, 0.4),
                'volatility_weight': np.random.uniform(0.02, 0.2),
                'volume_weight': np.random.uniform(0.02, 0.2),
                'trend_weight': np.random.uniform(0.1, 0.5)
            }
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
            
            population.append(weights)
        
        return population
    
    def _evolve_population(self, population: List[Dict[str, float]], 
                         fitness_scores: List[float], 
                         mutation_rate: float) -> List[Dict[str, float]]:
        """Evolve population using selection, crossover, and mutation"""
        
        # Convert fitness scores to probabilities
        fitness_array = np.array(fitness_scores)
        fitness_array = np.maximum(fitness_array, 0)  # Ensure non-negative
        probabilities = fitness_array / np.sum(fitness_array) if np.sum(fitness_array) > 0 else np.ones_like(fitness_array) / len(fitness_array)
        
        new_population = []
        
        # Keep best individual (elitism)
        best_idx = np.argmax(fitness_scores)
        new_population.append(population[best_idx].copy())
        
        # Generate rest of population through crossover and mutation
        for _ in range(len(population) - 1):
            # Select parents
            parent1 = self._select_parent(population, probabilities)
            parent2 = self._select_parent(population, probabilities)
            
            # Create offspring through crossover
            offspring = self._crossover(parent1, parent2)
            
            # Apply mutation
            offspring = self._mutate(offspring, mutation_rate)
            
            new_population.append(offspring)
        
        return new_population
    
    def _select_parent(self, population: List[Dict[str, float]], 
                      probabilities: np.ndarray) -> Dict[str, float]:
        """Select parent using roulette wheel selection"""
        idx = np.random.choice(len(population), p=probabilities)
        return population[idx].copy()
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Perform crossover between two parents"""
        offspring = {}
        
        for key in parent1.keys():
            # Uniform crossover
            if np.random.random() < 0.5:
                offspring[key] = parent1[key]
            else:
                offspring[key] = parent2[key]
        
        # Normalize weights
        total_weight = sum(offspring.values())
        if total_weight > 0:
            offspring = {k: v / total_weight for k, v in offspring.items()}
        
        return offspring
    
    def _mutate(self, individual: Dict[str, float], mutation_rate: float) -> Dict[str, float]:
        """Apply mutation to individual"""
        mutated = individual.copy()
        
        for key in mutated.keys():
            if np.random.random() < mutation_rate:
                # Add random noise
                noise = np.random.normal(0, 0.1)
                mutated[key] = max(0.01, min(0.99, mutated[key] + noise))
        
        # Normalize weights
        total_weight = sum(mutated.values())
        if total_weight > 0:
            mutated = {k: v / total_weight for k, v in mutated.items()}
        
        return mutated
    
    def save_optimization_results(self, results: OptimizationResult, filename: str):
        """Save optimization results to file"""
        data = {
            'best_weights': results.best_weights,
            'best_score': results.best_score,
            'best_sharpe': results.best_sharpe,
            'best_returns': results.best_returns,
            'best_drawdown': results.best_drawdown,
            'total_combinations': results.total_combinations,
            'execution_time': results.execution_time,
            'optimization_history': results.optimization_history
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filename}")
    
    def load_optimization_results(self, filename: str) -> OptimizationResult:
        """Load optimization results from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return OptimizationResult(
            best_weights=data['best_weights'],
            best_score=data['best_score'],
            best_sharpe=data['best_sharpe'],
            best_returns=data['best_returns'],
            best_drawdown=data['best_drawdown'],
            optimization_history=data['optimization_history'],
            total_combinations=data['total_combinations'],
            execution_time=data['execution_time']
        )


def create_weight_tuner(backtest_engine: BacktestEngine) -> WeightTuner:
    """Create a weight tuner with backtest engine"""
    return WeightTuner(backtest_engine)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be used with actual backtest engine and data
    print("Weight tuner module ready for optimization")
