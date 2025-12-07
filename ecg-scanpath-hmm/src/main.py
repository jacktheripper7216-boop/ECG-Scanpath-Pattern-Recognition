"""
ECG Scanpath Pattern Recognition Using Hidden Markov Models
Main program with two options:
1. Run full evaluation pipeline (train, test, show accuracy)
2. Enter a scanpath to classify (interactive mode)
"""
import csv
import numpy as np
from pathlib import Path
from hmm import HiddenMarkovModel, ScanpathClassifier
from evaluation import compute_accuracy, compute_confusion_matrix, print_confusion_matrix
# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(filepath):
    """
    Load scanpath data from CSV file.
    CSV format: participant_id, expertise_level, trial_id, fixation_id, ecg_lead, diagnostic_phase, duration_ms
    Returns:
        expert_data: List of (observations, states) tuples for experts
        novice_data: List of (observations, states) tuples for novices
    """
    # look for dataset in the known path
    fp = Path(filepath)
    if not fp.exists():
        # locate dataset
        script_dir = Path(__file__).parent
        candidate = script_dir.parent / 'data' / 'scanpath_dataset.csv'
        if candidate.exists():
            fp = candidate
        else:
            candidate2 = script_dir.parent.parent / 'data' / 'scanpath_dataset.csv'
            if candidate2.exists():
                fp = candidate2
            else:
                raise FileNotFoundError(f"Could not find dataset at '{filepath}' or expected locations: {candidate} or {candidate2}")

    # Dictionary to collect data by participant
    samples = {}
    with fp.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['participant_id']
            # Initialize new participant
            if pid not in samples:
                samples[pid] = {
                    'expertise': row['expertise_level'],
                    'observations': [],
                    'states': []
                }
            # Add fixation data
            samples[pid]['observations'].append(row['ecg_lead'])
            samples[pid]['states'].append(row['diagnostic_phase'])
    
    # Separate into expert and novice lists
    expert_data = []
    novice_data = []
    
    for pid, data in samples.items():
        pair = (data['observations'], data['states'])
        if data['expertise'] == 'expert':
            expert_data.append(pair)
        else:
            novice_data.append(pair)
    
    return expert_data, novice_data


# =============================================================================
# MENU DISPLAY
# =============================================================================

def display_menu():
    """Display main menu and get user choice."""
    print("\n" + "=" * 60)
    print("ECG SCANPATH PATTERN RECOGNITION")
    print("Using Hidden Markov Models")
    print("=" * 60)
    print("""
Options:
    1. Run full evaluation pipeline
    2. Enter a scanpath to classify
    3. Exit
""")
    return input("Enter choice (1/2/3): ").strip()


# =============================================================================
# OPTION 1: FULL EVALUATION PIPELINE
# =============================================================================

def run_evaluation():
    """ steps:
    Run the full evaluation pipeline:
    1. Load data from CSV
    2. Split into train/test sets
    3. Train expert and novice HMMs
    4. Test on held-out data
    5. Report accuracy and confusion matrix
    it returns:
        classifier: Trained ScanpathClassifier for use in interactive mode
    """
    print("\n" + "-" * 60)
    print("RUNNING EVALUATION PIPELINE")
    print("-" * 60)
    # Step 1: Load data
    expert_data, novice_data = load_dataset('scanpath_dataset.csv')
    print(f"\nDataset: {len(expert_data)} expert, {len(novice_data)} novice sequences")
    
    # Step 2: Split into train (80%) and test (20%)
    n_expert_train = int(len(expert_data) * 0.8)
    n_novice_train = int(len(novice_data) * 0.8)
    expert_train = expert_data[:n_expert_train]
    expert_test = expert_data[n_expert_train:]
    novice_train = novice_data[:n_novice_train]
    novice_test = novice_data[n_novice_train:]
    
    print(f"Training set: {len(expert_train)} expert, {len(novice_train)} novice")
    print(f"Test set: {len(expert_test)} expert, {len(novice_test)} novice")
    
    # Step 3: Train classifier
    classifier = ScanpathClassifier()
    classifier.train(expert_train, novice_train)
    
    # Step 4: Classify test samples
    y_true = []
    y_pred = []
    
    # Test expert samples
    for obs, states in expert_test:
        pred, _, _ = classifier.classify(obs)
        y_true.append('EXPERT')
        y_pred.append(pred)
    
    # Test novice samples
    for obs, states in novice_test:
        pred, _, _ = classifier.classify(obs)
        y_true.append('NOVICE')
        y_pred.append(pred)
    
    # Step 5: Compute and display results
    accuracy = compute_accuracy(y_true, y_pred)
    matrix, counts = compute_confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.1%} ({int(accuracy * len(y_true))}/{len(y_true)} correct)")
    
    print_confusion_matrix(matrix, counts)
    
    return classifier


# =============================================================================
# OPTION 2: INTERACTIVE CLASSIFICATION
# =============================================================================

def classify_user_scanpath(classifier):
    """
    the user input a scanpath and get classification + Viterbi decoding.
    Uses:
    - Forward algorithm: Computes likelihood for classification
    - Viterbi algorithm: Decodes hidden state sequence
    """
    print("\n" + "-" * 60)
    print("SCANPATH CLASSIFICATION")
    print("-" * 60)
    print("""
Valid ECG leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
Enter scanpath as comma-separated leads.
Example: II, I, aVF, V1, V2, V3, V4, V5
""")
    
    # Get user input
    user_input = input("Enter scanpath: ").strip()
    
    # Parse input (handle comma or space separation)
    if ',' in user_input:
        scanpath = [s.strip() for s in user_input.split(',')]
    else:
        scanpath = user_input.split()
    
    # Validate ECG leads
    valid_leads = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
    invalid = [lead for lead in scanpath if lead not in valid_leads]
    
    if invalid:
        print(f"\nError: Invalid leads: {invalid}")
        print(f"Valid leads are: {sorted(valid_leads)}")
        return
    
    if len(scanpath) < 2:
        print("\nError: Scanpath must have at least 2 fixations.")
        return
    
    print(f"\nInput: {scanpath}")
    
    # =========================================================================
    # FORWARD ALGORITHM - Classification
    # =========================================================================
    print("\n" + "=" * 60)
    print("FORWARD ALGORITHM - Classification")
    print("=" * 60)
    
    # Get classification using Forward algorithm
    prediction, expert_ll, novice_ll = classifier.classify(scanpath)
    
    print(f"""
Log-Likelihoods:
    P(O | Expert Model) = {expert_ll:.4f}
    P(O | Novice Model) = {novice_ll:.4f}

Decision: {expert_ll:.4f} {'>' if expert_ll > novice_ll else '<'} {novice_ll:.4f}
""")
    
    # Display verdict
    print("    ╔════════════════════════════════════╗")
    print(f"    ║  VERDICT:  {prediction:^22}  ║")
    print("    ╚════════════════════════════════════╝")
    
    # =========================================================================
    # VITERBI ALGORITHM - State Sequence Decoding
    # =========================================================================
    print("\n" + "=" * 60)
    print("VITERBI ALGORITHM - Hidden State Sequence")
    print("=" * 60)
    
    # Use the winning model for decoding
    if prediction == 'EXPERT':
        decoded_states, log_prob = classifier.expert_hmm.viterbi(scanpath)
    else:
        decoded_states, log_prob = classifier.novice_hmm.viterbi(scanpath)
    
    # Display decoded sequence
    print(f"\n{'Step':<6}{'Observation':<12}{'Hidden State':<28}")
    print("-" * 46)
    for i, (obs, state) in enumerate(zip(scanpath, decoded_states), 1):
        print(f"{i:<6}{obs:<12}{state:<28}")
    
    # Show state progression summary
    unique_states = []
    for s in decoded_states:
        if not unique_states or unique_states[-1] != s:
            unique_states.append(s)
    
    print(f"\nState progression:")
    print(f"  {' -> '.join(unique_states)}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function with menu loop."""
    
    classifier = None  # this will be trained when needed
    
    while True:
        choice = display_menu()
        if choice == '1':
            # Run full evaluation pipeline
            classifier = run_evaluation()
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            # Interactive scanpath classification
            if classifier is None:
                # Train classifier first
                print("\nTraining classifier...")
                expert_data, novice_data = load_dataset('scanpath_dataset.csv')
                classifier = ScanpathClassifier()
                classifier.train(expert_data, novice_data)
                print("Training complete.")
            
            classify_user_scanpath(classifier)
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            print("\nGoodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()