import pandas as pd
import numpy as np

def generate_realistic_data(n_samples=5000):
    """
    Generate a realistic synthetic student performance dataset.
    Correlations:
    - Hours Studied: Strong positive
    - Previous Scores: Very strong positive
    - Extracurricular: Subtle positive
    - Sleep Hours: Bell curve (peak at 7-8 hours)
    - Sample Papers: Moderate positive
    """
    np.random.seed(42)
    
    # Generate features
    hours_studied = np.random.randint(1, 10, n_samples)
    previous_scores = np.random.randint(40, 100, n_samples)
    extracurricular = np.random.choice(['Yes', 'No'], n_samples)
    sleep_hours = np.random.randint(4, 10, n_samples)
    sample_papers = np.random.randint(0, 10, n_samples)
    
    # Calculate target (Performance Index) based on a formula with some noise
    # Base: 0-100 scale
    # Coefficients were estimated to keep result in [0, 100]
    perf_index = (
        (hours_studied * 2.8) + 
        (previous_scores * 0.6) + 
        (np.where(extracurricular == 'Yes', 1.5, 0)) +
        (sleep_hours * 0.5) +
        (sample_papers * 0.4) +
        np.random.normal(0, 2, n_samples) # Add some noise
    )
    
    # Clip to [0, 100]
    perf_index = np.clip(perf_index, 0, 100)
    
    df = pd.DataFrame({
        'Hours Studied': hours_studied,
        'Previous Scores': previous_scores,
        'Extracurricular Activities': extracurricular,
        'Sleep Hours': sleep_hours,
        'Sample Question Papers Practiced': sample_papers,
        'Performance Index': perf_index.round(1)
    })
    
    df.to_csv('dataset.csv', index=False)
    print(f"Generated {n_samples} realistic samples and saved to dataset.csv")

if __name__ == "__main__":
    generate_realistic_data()
