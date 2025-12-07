"""
Hidden Markov Model Implementation for ECG Scanpath Pattern Recognition
This module implements the core HMM algorithms:
- Forward Algorithm: Computes likelihood P(O|λ)
- Viterbi Algorithm: Finds most likely hidden state sequence
- Supervised Training: Learns model parameters from labeled data
"""

import numpy as np
import json


# =============================================================================
# CONSTANTS: ECG Interpretation Domain
# =============================================================================

# Hidden states: Cognitive diagnostic phases during ECG interpretation
DIAGNOSTIC_PHASES = [
    'Rhythm-Check',           # Check heart rhythm using Lead II
    'Axis-Determination',     # Determine heart axis using I and aVF
    'P-wave-Analysis',        # Analyze P-wave morphology
    'PR-interval-Assessment', # Measure PR interval
    'QRS-Analysis',           # Analyze QRS complex in precordial leads
    'ST-segment-Evaluation',  # Check for ST elevation/depression
    'T-wave-Examination',     # Examine T-wave morphology
    'QT-interval-Measurement',# Measure QT interval
    'Lead-by-Lead-Review'     # Final systematic review
]

# Observable states: The 12 ECG leads that can be fixated
ECG_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
             'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# =============================================================================
# HIDDEN MARKOV MODEL CLASS
# =============================================================================

class HiddenMarkovModel:
    """
    Hidden Markov Model for ECG scanpath analysis.
    
    An HMM is defined by λ = (A, B, π) where:
    - A: Transition probability matrix (N × N)
    - B: Emission probability matrix (N × M)  
    - π: Initial state distribution (N × 1)
    
    Where N = number of hidden states, M = number of observations.
    """
    
    def __init__(self, states=None, observations=None):
        """
        Initialize HMM with states and observations.
        
        Args:
            states: List of hidden state names (default: DIAGNOSTIC_PHASES)
            observations: List of observation names (default: ECG_LEADS)
        """
        # Set default states and observations for ECG domain
        self.states = states if states else DIAGNOSTIC_PHASES
        self.observations = observations if observations else ECG_LEADS
        
        # Number of hidden states and observations
        self.N = len(self.states)  # 9 diagnostic phases
        self.M = len(self.observations)  # 12 ECG leads
        
        # Create mapping dictionaries for fast lookup
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.obs_to_idx = {o: i for i, o in enumerate(self.observations)}
        self.idx_to_obs = {i: o for i, o in enumerate(self.observations)}
        
        # Initialize model parameters (will be learned during training)
        self.A = None   # Transition matrix
        self.B = None   # Emission matrix
        self.pi = None  # Initial distribution
    
    
    # =========================================================================
    # FORWARD ALGORITHM - Evaluation Problem
    # =========================================================================
    
    def forward(self, observations):
        """
        Forward Algorithm: Compute log-likelihood of observation sequence.
        
        Given an observation sequence O = (o_1, o_2, ..., o_T), computes
        P(O|λ) - the probability that this HMM generated the sequence.
        
        Algorithm (using log-space for numerical stability):
            1. Initialization: α_1(i) = π_i × B_i(o_1)
            2. Induction: α_t(j) = [Σ_i α_{t-1}(i) × A_{i,j}] × B_j(o_t)
            3. Termination: P(O|λ) = Σ_i α_T(i)
        
        Complexity: O(N² × T) time, O(N × T) space
        
        Args:
            observations: List of observation symbols (e.g., ['II', 'I', 'aVF'])
        
        Returns:
            log_likelihood: Log probability of the observation sequence
        """
        # Convert observations to indices
        T = len(observations)
        obs_indices = [self.obs_to_idx[o] for o in observations]
        
        # Small constant to prevent log(0)
        eps = 1e-300
        
        # Convert to log-space for numerical stability
        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)
        
        # Initialize log-alpha matrix
        log_alpha = np.zeros((T, self.N))
        
        # Step 1: Initialization
        # α_1(i) = π_i × B_i(o_1)
        log_alpha[0] = log_pi + log_B[:, obs_indices[0]]
        
        # Step 2: Induction (forward pass)
        # α_t(j) = [Σ_i α_{t-1}(i) × A_{i,j}] × B_j(o_t)
        for t in range(1, T):
            for j in range(self.N):
                # Log-sum-exp trick for numerical stability
                log_alpha[t, j] = self._logsumexp(
                    log_alpha[t-1] + log_A[:, j]
                ) + log_B[j, obs_indices[t]]
        
        # Step 3: Termination
        # P(O|λ) = Σ_i α_T(i)
        log_likelihood = self._logsumexp(log_alpha[T-1])
        
        return log_likelihood
    
    
    def _logsumexp(self, x):
        """
        Compute log(sum(exp(x))) in a numerically stable way.
        
        This prevents overflow/underflow when working with log probabilities.
        """
        max_x = np.max(x)
        return max_x + np.log(np.sum(np.exp(x - max_x)))
    
    
    # =========================================================================
    # VITERBI ALGORITHM - Decoding Problem
    # =========================================================================
    
    def viterbi(self, observations):
        """
        Viterbi Algorithm: Find most likely hidden state sequence.
        
        Given observations O, finds the state sequence Q* that maximizes
        P(Q|O,λ) - the most probable path through the hidden states.
        
        Algorithm:
            1. Initialization: δ_1(i) = π_i × B_i(o_1), ψ_1(i) = 0
            2. Recursion: δ_t(j) = max_i[δ_{t-1}(i) × A_{i,j}] × B_j(o_t)
                         ψ_t(j) = argmax_i[δ_{t-1}(i) × A_{i,j}]
            3. Termination: q*_T = argmax_i[δ_T(i)]
            4. Backtracking: q*_t = ψ_{t+1}(q*_{t+1})
        
        Complexity: O(N² × T) time, O(N × T) space
        
        Args:
            observations: List of observation symbols
        
        Returns:
            best_path: List of most likely hidden states
            log_prob: Log probability of the best path
        """
        # Convert observations to indices
        T = len(observations)
        obs_indices = [self.obs_to_idx[o] for o in observations]
        
        # Small constant to prevent log(0)
        eps = 1e-300
        
        # Convert to log-space
        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)
        
        # Initialize delta (max probability) and psi (backpointer) matrices
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        
        # Step 1: Initialization
        # δ_1(i) = π_i × B_i(o_1)
        delta[0] = log_pi + log_B[:, obs_indices[0]]
        psi[0] = 0  # No predecessor for first state
        
        # Step 2: Recursion (forward pass with max instead of sum)
        for t in range(1, T):
            for j in range(self.N):
                # Find the best previous state
                candidates = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[j, obs_indices[t]]
        
        # Step 3: Termination
        # Find best final state
        best_last_state = np.argmax(delta[T-1])
        log_prob = delta[T-1, best_last_state]
        
        # Step 4: Backtracking
        # Recover the best path by following backpointers
        best_path_indices = [0] * T
        best_path_indices[T-1] = best_last_state
        
        for t in range(T-2, -1, -1):
            best_path_indices[t] = psi[t+1, best_path_indices[t+1]]
        
        # Convert indices back to state names
        best_path = [self.idx_to_state[i] for i in best_path_indices]
        
        return best_path, log_prob
    
    
    # =========================================================================
    # SUPERVISED TRAINING - Learning with Labeled Data
    # =========================================================================
    
    def train_supervised(self, training_data, smoothing=0.01):
        """
        Train HMM parameters using Maximum Likelihood Estimation.
        Given labeled training data (observations + hidden states), estimates:
        - A[i,j]: P(state_j | state_i) - transition probabilities
        - B[j,k]: P(obs_k | state_j) - emission probabilities  
        - π[i]: P(state_i at t=1) - initial state probabilities
        
        Uses Laplace smoothing to handle unseen transitions/emissions.
        
        Args:
            training_data: List of (observations, states) tuples
                observations: List of ECG leads (e.g., ['II', 'I', 'aVF'])
                states: List of diagnostic phases (same length)
            smoothing: Laplace smoothing parameter (default 0.01)
        """
        # Initialize count matrices
        transition_counts = np.zeros((self.N, self.N)) + smoothing
        emission_counts = np.zeros((self.N, self.M)) + smoothing
        initial_counts = np.zeros(self.N) + smoothing
        
        # Count occurrences from training data
        for observations, states in training_data:
            # Count initial state
            initial_state_idx = self.state_to_idx[states[0]]
            initial_counts[initial_state_idx] += 1
            
            # Count transitions and emissions
            for t in range(len(states)):
                state_idx = self.state_to_idx[states[t]]
                obs_idx = self.obs_to_idx[observations[t]]
                
                # Count emission
                emission_counts[state_idx, obs_idx] += 1
                
                # Count transition (if not last state)
                if t < len(states) - 1:
                    next_state_idx = self.state_to_idx[states[t + 1]]
                    transition_counts[state_idx, next_state_idx] += 1
        
        # Normalize to get probabilities
        # Each row must sum to 1
        self.A = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        self.B = emission_counts / emission_counts.sum(axis=1, keepdims=True)
        self.pi = initial_counts / initial_counts.sum()
    
    
    # =========================================================================
    # SAVE / LOAD MODEL
    # =========================================================================
    
    def save(self, filepath):
        """Save trained model to JSON file."""
        model_data = {
            'states': self.states,
            'observations': self.observations,
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'pi': self.pi.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load(self, filepath):
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.states = model_data['states']
        self.observations = model_data['observations']
        self.N = len(self.states)
        self.M = len(self.observations)
        
        # Rebuild mappings
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.obs_to_idx = {o: i for i, o in enumerate(self.observations)}
        self.idx_to_obs = {i: o for i, o in enumerate(self.observations)}
        
        # Load parameters
        self.A = np.array(model_data['A'])
        self.B = np.array(model_data['B'])
        self.pi = np.array(model_data['pi'])


# =============================================================================
# SCANPATH CLASSIFIER
# =============================================================================

class ScanpathClassifier:
    """
    Binary classifier for ECG scanpaths: Expert vs Novice.
    Uses two HMMs:
    - Expert HMM: Trained on expert scanpaths (systematic patterns)
    - Novice HMM: Trained on novice scanpaths (erratic patterns)
    
    Classification: Compare likelihoods P(O|λ_expert) vs P(O|λ_novice)
    """
    
    def __init__(self):
        """Initialize with two untrained HMMs."""
        self.expert_hmm = HiddenMarkovModel()
        self.novice_hmm = HiddenMarkovModel()
    
    def train(self, expert_data, novice_data):
        """
        Train both HMMs on their respective data.
        
        Args:
            expert_data: List of (observations, states) for experts
            novice_data: List of (observations, states) for novices
        """
        self.expert_hmm.train_supervised(expert_data)
        self.novice_hmm.train_supervised(novice_data)
    
    def classify(self, observations):
        """
        Classify a scanpath as EXPERT or NOVICE.
        
        Uses Forward algorithm to compute likelihoods, then compares.
        
        Args:
            observations: List of ECG leads (e.g., ['II', 'I', 'aVF', 'V1'])
        
        Returns:
            prediction: 'EXPERT' or 'NOVICE'
            expert_likelihood: Log P(O|λ_expert)
            novice_likelihood: Log P(O|λ_novice)
        """
        # Compute likelihood under each model using Forward algorithm
        expert_ll = self.expert_hmm.forward(observations)
        novice_ll = self.novice_hmm.forward(observations)
        
        # Classify based on higher likelihood
        if expert_ll > novice_ll:
            prediction = 'EXPERT'
        else:
            prediction = 'NOVICE'
        
        return prediction, expert_ll, novice_ll