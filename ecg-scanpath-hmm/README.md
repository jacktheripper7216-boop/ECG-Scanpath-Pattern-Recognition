# ECG Scanpath Pattern Recognition Using Hidden Markov Models

## Project Overview

This project applies Hidden Markov Models (HMMs) to analyze eye-tracking scanpaths during 12-lead ECG interpretation. The model distinguishes between expert and novice clinicians based on their visual scanning patterns.

## Authors

- Riad Benbrahim (riad.benbrahim@um6p.ma)
- Mohamed Amine El Bacha (mohamedamine.elbacha@um6p.ma)
- Youssef Kaya (youssef.kaya@um6p.ma)

Course: Computational Theory  
University: UM6P

## Files

| File | Description |
|------|-------------|
| `main.py` | Main program with menu interface |
| `hmm.py` | Core HMM algorithms (Forward, Viterbi, Training) |
| `evaluation.py` | Accuracy and confusion matrix computation |
| `scanpath_dataset.csv` | Dataset with expert and novice scanpaths |

## Usage

Run the program:
```bash
python main.py
```

Menu options:
1. **Run full evaluation pipeline** - Train, test, and show accuracy
2. **Enter a scanpath to classify** - Interactive mode
3. **Exit**

## Algorithms

### Forward Algorithm
Computes P(O|λ) - the probability that an HMM generated an observation sequence.
Used for classification by comparing likelihoods from expert and novice models.

### Viterbi Algorithm  
Finds the most likely hidden state sequence Q* = argmax P(Q|O,λ).
Decodes the cognitive diagnostic phases from observed ECG lead fixations.

### Supervised Training
Learns model parameters (A, B, π) from labeled training data using Maximum Likelihood Estimation.

## Results

- Accuracy: 97.5%
- Confusion Matrix:

```
                 Predicted
              EXPERT    NOVICE
Actual EXPERT    20         0
       NOVICE     1        19
```

## Model Structure

- **Hidden States (N=9):** Diagnostic phases (Rhythm-Check, Axis-Determination, P-wave-Analysis, etc.)
- **Observations (M=12):** ECG leads (I, II, III, aVR, aVL, aVF, V1-V6)
- **Parameters:** 198 total (81 transitions + 108 emissions + 9 initial)
