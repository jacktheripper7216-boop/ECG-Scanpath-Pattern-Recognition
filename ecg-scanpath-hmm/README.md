# ECG Scanpath Pattern Recognition Using Hidden Markov Models

A computational framework for analyzing medical expertise through eye-tracking scanpath patterns during ECG interpretation.

## Authors

- Mohamed Amine El Bacha (mohamedamine.elbacha@um6p.ma)
- Youssef Kaya (youssef.kaya@um6p.ma)
- Riad Benbrahim (riad.benbrahim@um6p.ma)
- 
**Course:** Computational Theory - Fall 2025  
**Institution:** Mohammed VI Polytechnic University (UM6P)

---

## Project Overview

This project applies Hidden Markov Models (HMMs) to analyze and classify eye-tracking scanpaths recorded during 12-lead ECG interpretation. The goal is to distinguish between expert and novice reading patterns based on their visual scanning behaviour.

### Research Question

To what extent can expert visual interpretation strategies of 12-lead ECGs be modeled as structured probabilistic sequences using a Hidden Markov Model? How accurately can this model distinguish expert scanpaths from novice ones?

### Key Idea

When medical professionals read an ECG, they go through cognitive diagnostic phases (hidden states) that cause them to look at specific ECG leads (observable states). Experts follow systematic patterns based on clinical guidelines, while novices exhibit more random and disorganized scanning behaviour.

---

## Model Description

### Hidden Markov Model Components

The HMM is defined as a 5-tuple: `λ = (S, O, A, B, π)`

**Hidden States (S)** - 9 cognitive diagnostic phases:
1. Rhythm-Check
2. Axis-Determination
3. P-wave-Analysis
4. PR-interval-Assessment
5. QRS-Analysis
6. ST-segment-Evaluation
7. T-wave-Examination
8. QT-interval-Measurement
9. Lead-by-Lead-Review

**Observable States (O)** - 12 ECG leads:
- Limb leads: I, II, III
- Augmented leads: aVR, aVL, aVF
- Precordial leads: V1, V2, V3, V4, V5, V6

**Parameters:**
- `A` (9x9): Transition probability matrix between hidden states
- `B` (9x12): Emission probability matrix from hidden states to observations
- `π` (9x1): Initial state distribution

### Classification Approach

Two separate HMMs are trained:
- Expert HMM: trained on expert scanpath data
- Novice HMM: trained on novice scanpath data

Classification is performed using likelihood comparison:
- Compute log P(O | expert_model) and log P(O | novice_model)
- Assign the scanpath to the class with higher likelihood

---

## Installation

### Requirements

- Python 3.8 or higher
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ecg-scanpath-hmm.git
cd ecg-scanpath-hmm

# Install dependencies
pip install numpy
```

---

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

This will:
2. Train expert and novice HMM models on generated data 
3. Evaluate classification performance
4. Display results and analysis

### Command Line Options

```bash
python main.py --n-train 150      # Number of training samples per class
python main.py --n-test 50        # Number of test samples per class
python main.py --seed 42          # Random seed for reproducibility
python main.py --output results/  # Output directory
python main.py --save-models      # Save trained models to disk
```

### Using Individual Modules

```python
# Train classifier
from hmm import ScanpathClassifier

classifier = ScanpathClassifier()
classifier.train(expert_data, novice_data)

# Classify new scanpath
prediction, expert_ll, novice_ll = classifier.classify(observation_sequence)

# Decode hidden states
hidden_states = classifier.decode_cognitive_states(observation_sequence)
```

---

## Algorithms Implemented

### Forward Algorithm
Computes the probability of an observation sequence given the model.
- Time complexity: O(N^2 * T)
- Space complexity: O(N * T)
- Used for classification (likelihood computation)

### Viterbi Algorithm
Finds the most likely sequence of hidden states.
- Time complexity: O(N^2 * T)
- Space complexity: O(N * T)
- Used for decoding cognitive phases from scanpaths

### Baum-Welch Algorithm
Trains HMM parameters from unlabeled observation sequences.
- Iterative Expectation-Maximization approach
- Used when hidden state labels are not available

### Maximum Likelihood Estimation
Trains HMM parameters from labeled data.
- Direct counting with Laplace smoothing
- Used when both observations and hidden states are known

---

## Dataset

### Synthetic Data Generation

The dataset is synthetically generated based on:
- AHA/ACCF guidelines for systematic ECG interpretation
- Clinical workflow patterns observed in expert cardiologists
- Research on expert-novice differences in medical image interpretation

### Expert Patterns
- Systematic progression through diagnostic phases
- Focused fixations on diagnostically relevant leads
- Smooth transitions following clinical workflow

### Novice Patterns
- Erratic, non-systematic viewing
- Random transitions between phases
- Incomplete coverage of ECG regions

### Data Format (CSV)

| Column | Description |
|--------|-------------|
| participant_id | Unique identifier (E001 for expert, N001 for novice) |
| expertise_level | "expert" or "novice" |
| trial_id | Trial identifier |
| fixation_id | Sequential fixation number |
| ecg_lead | Observed ECG lead (I, II, III, aVR, aVL, aVF, V1-V6) |
| diagnostic_phase | Hidden cognitive state |
| duration_ms | Fixation duration in milliseconds |

---

## Limitations

The high classification accuracy should be interpreted with caution. This performance is largely due to the synthetic nature of our dataset, where expert and novice patterns were designed with clear distinctions. Real clinical data would likely show more overlap between classes. Validation with real eye-tracking data is needed to confirm generalizability.

---


## References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.

2. Surawicz, B., et al. (2009). AHA/ACCF/HRS recommendations for the standardization and interpretation of the electrocardiogram. Circulation, 119(10), e235-e240.

3. Holmqvist, K., et al. (2011). Eye Tracking: A Comprehensive Guide to Methods and Measures. Oxford University Press.

---

## License

This project is developed for educational purposes as part of the Computational Theory course at UM6P.
