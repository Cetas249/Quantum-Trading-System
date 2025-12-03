"""
quantum/optimization_algorithms.py
Quantum optimization implementations for Python 3.14
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from dataclasses import dataclass
from typing import Any, Callable, Optional
from collections.abc import Sequence
from functools import lru_cache

from qiskit import QuantumCircuit
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

@dataclass
class QuantumResult:
    """Result from quantum computation"""
    optimal_weights: np.ndarray
    optimization_result: dict[str, Any]
    quantum_params: np.ndarray
    iterations: int
    convergence_achieved: bool

class QuantumOptimizer:
    """Python 3.14 compatible quantum optimization engine"""
    
    def __init__(self, num_qubits: int = 6) -> None:
        self.num_qubits: int = num_qubits
        self.optimization_history: list[dict[str, float]] = []
        self._qml_available: bool = qml is not None
        self.device: Optional[Any] = qml.device('default.qubit', wires=num_qubits) if self._qml_available else None
    
    def create_portfolio_optimization_circuit(
        self, 
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Callable:
        """Create quantum circuit for portfolio optimization (QAOA-inspired)"""
        
        if self._qml_available:
            @qml.qnode(self.device)
            def portfolio_circuit(params: np.ndarray) -> list[float]:
                # Initialize superposition
                for i in range(self.num_qubits):
                    qml.Hadamard(wires=i)
                
                # Problem-specific gates
                for layer in range(len(params) // 2):
                    for i in range(self.num_qubits):
                        qml.RZ(params[2 * layer] * expected_returns[i], wires=i)
                    
                    for i in range(self.num_qubits):
                        qml.RX(params[2 * layer + 1], wires=i)
                    
                    # Entanglement
                    for i in range(self.num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        else:
            def portfolio_circuit(params: np.ndarray) -> list[float]:
                """Deterministic fallback when PennyLane isn't available."""
                seed = int(abs(float(np.sum(params))) * 1e6) % (2**32 - 1 or 1)
                rng = np.random.default_rng(seed)
                return rng.uniform(-1.0, 1.0, size=self.num_qubits).tolist()
        
        return portfolio_circuit
    
    def quantum_portfolio_optimization(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_tolerance: float = 0.5
    ) -> QuantumResult:
        """Optimize portfolio using quantum annealing approach"""
        
        num_layers: int = 3
        params: np.ndarray = np.random.uniform(0, 2*np.pi, 2*num_layers)
        
        circuit: Callable = self.create_portfolio_optimization_circuit(
            params, expected_returns, cov_matrix
        )
        
        def cost_function(params: np.ndarray) -> float:
            expectations: list[float] = circuit(params)
            weights: np.ndarray = np.abs(expectations) / (np.sum(np.abs(expectations)) + 1e-8)
            
            portfolio_return: float = float(np.dot(weights, expected_returns))
            portfolio_risk: float = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
            
            return -(portfolio_return - risk_tolerance * portfolio_risk)
        
        from scipy.optimize import minimize
        result = minimize(cost_function, params, method='COBYLA')
        
        optimal_expectations: list[float] = circuit(result.x)
        optimal_weights: np.ndarray = np.abs(optimal_expectations) / (np.sum(np.abs(optimal_expectations)) + 1e-8)
        
        return QuantumResult(
            optimal_weights=optimal_weights,
            optimization_result=result.__dict__,
            quantum_params=result.x,
            iterations=result.nit,
            convergence_achieved=result.success
        )
    
    @lru_cache(maxsize=128)
    def variational_quantum_algorithm(
        self,
        data_shape: tuple[int, ...],
        num_features: int
    ) -> Callable:
        """VQA for market prediction with caching"""
        
        if self._qml_available:
            @qml.qnode(self.device)
            def vqa_circuit(params: np.ndarray, x: np.ndarray) -> float:
                # Data encoding
                for i, feature in enumerate(x[:self.num_qubits]):
                    qml.RY(float(feature), wires=i)
                
                # Variational layers
                num_var_layers: int = len(params) // (2 * self.num_qubits)
                for layer in range(num_var_layers):
                    for i in range(self.num_qubits):
                        qml.RY(params[layer * 2 * self.num_qubits + i], wires=i)
                        qml.RZ(params[layer * 2 * self.num_qubits + self.num_qubits + i], wires=i)
                    
                    for i in range(self.num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                
                return qml.expval(qml.PauliZ(0))
        else:
            def vqa_circuit(params: np.ndarray, x: np.ndarray) -> float:
                """Lightweight fallback returning a bounded deterministic score."""
                if params.size == 0 or x.size == 0:
                    return 0.0
                size = min(params.size, x.size)
                dot_product = float(np.dot(params[:size], x[:size]))
                return float(np.tanh(dot_product / (size or 1)))
        
        return vqa_circuit

class QuantumInspiredOptimization:
    """Classical algorithms inspired by quantum principles (Python 3.14 style)"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1) -> None:
        self.population_size: int = population_size
        self.mutation_rate: float = mutation_rate
        self.fitness_history: list[float] = []
    
    def quantum_inspired_genetic_algorithm(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Sequence[tuple[float, float]],
        generations: int = 100
    ) -> dict[str, Any]:
        """Quantum-inspired genetic algorithm"""
        
        population: list[np.ndarray] = []
        for _ in range(self.population_size):
            individual: list[float] = [
                np.random.uniform(low, high) for low, high in bounds
            ]
            population.append(np.array(individual))
        
        best_fitness: float = float('inf')
        best_individual: Optional[np.ndarray] = None
        
        for generation in range(generations):
            fitness_scores: list[float] = [
                objective_function(ind) for ind in population
            ]
            
            min_fitness_idx: int = int(np.argmin(fitness_scores))
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_individual = population[min_fitness_idx].copy()
            
            self.fitness_history.append(best_fitness)
            
            new_population: list[np.ndarray] = []
            
            for _ in range(self.population_size):
                parent1: np.ndarray = self._quantum_selection(population, fitness_scores)
                parent2: np.ndarray = self._quantum_selection(population, fitness_scores)
                
                child: np.ndarray = self._quantum_crossover(parent1, parent2)
                child = self._quantum_mutation(child, bounds)
                
                new_population.append(child)
            
            population = new_population
        
        return {
            'best_solution': best_individual,
            'best_fitness': best_fitness,
            'fitness_history': self.fitness_history
        }
    
    def _quantum_selection(
        self,
        population: list[np.ndarray],
        fitness_scores: list[float]
    ) -> np.ndarray:
        """Quantum superposition-inspired selection"""
        fitness_array: np.ndarray = np.array(fitness_scores)
        inv_fitness: np.ndarray = 1 / (fitness_array + 1e-8)
        probabilities: np.ndarray = inv_fitness / np.sum(inv_fitness)
        
        selected_idx: int = int(np.random.choice(len(population), p=probabilities))
        return population[selected_idx]
    
    def _quantum_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Quantum entanglement-inspired crossover"""
        child: list[float] = []
        for i in range(len(parent1)):
            alpha: float = float(np.random.random())
            gene: float = float(alpha * parent1[i] + (1 - alpha) * parent2[i])
            child.append(gene)
        return np.array(child)
    
    def _quantum_mutation(
        self,
        individual: np.ndarray,
        bounds: Sequence[tuple[float, float]]
    ) -> np.ndarray:
        """Quantum tunneling-inspired mutation"""
        mutated: np.ndarray = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                low, high = bounds[i]
                mutated[i] = float(np.random.uniform(low, high))
        return mutated

class QiskitPortfolioOptimizer:
    """Portfolio optimization using Qiskit QAOA"""
    
    def __init__(self):
        self.sampler = Sampler()
        self.optimizer = COBYLA()
    
    def optimize_portfolio(self, expected_returns, covariance, risk_factor=0.5):
        """QAOA-based portfolio optimization"""
        # Define quadratic program
        qp = QuadraticProgram()
        num_assets = len(expected_returns)
        
        for i in range(num_assets):
            qp.binary_var(f'x{i}')
        
        # Objective: maximize return - risk
        qp.maximize(
            linear=expected_returns.tolist(),
            quadratic=-risk_factor * covariance.tolist()
        )
        
        # Solve with QAOA
        qaoa = QAOA(sampler=self.sampler, optimizer=self.optimizer, reps=2)
        optimizer = MinimumEigenOptimizer(qaoa)
        
        result = optimizer.solve(qp)
        return result