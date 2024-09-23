
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to check for bias in the dataset
def check_data_bias(data, protected_columns):
    """
    Check for data bias in protected attributes like gender or race.
    :param data: DataFrame containing the dataset.
    :param protected_columns: List of columns that are protected attributes.
    :return: Dictionary with bias analysis for each protected column.
    """
    bias_report = {}
    
    for column in protected_columns:
        # Count unique values in the protected attribute
        value_counts = data[column].value_counts(normalize=True) * 100
        
        # Check for imbalances (if any category exceeds 60% of the total)
        max_percentage = value_counts.max()
        bias_report[column] = {
            "distribution": value_counts.to_dict(),
            "max_percentage": max_percentage,
            "is_biased": max_percentage > 60
        }
        
    return bias_report

# Function to calculate basic model interpretability metrics (e.g., feature importance)
def explain_model_importance(model, feature_names):
    """
    Calculate feature importance for interpretability assessment.
    :param model: Trained model with a feature importance attribute (e.g., tree-based models).
    :param feature_names: List of feature names corresponding to the input data.
    :return: DataFrame with feature importance values.
    """
    # Assuming the model has a feature_importances_ attribute (like in tree-based models)
    importance = model.feature_importances_
    
    # Create a DataFrame for better readability
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
    
    return importance_df

# Function to generate a compliance report
def generate_compliance_report(data, protected_columns, model=None, feature_names=None):
    """
    Generate a compliance report that checks for bias and model interpretability.
    :param data: DataFrame containing dataset.
    :param protected_columns: List of protected attribute columns to check for bias.
    :param model: Trained model to evaluate interpretability (optional).
    :param feature_names: List of feature names corresponding to the input data (optional).
    :return: Dictionary with compliance report.
    """
    report = {}
    
    # Check for bias in data
    report['bias_check'] = check_data_bias(data, protected_columns)
    
    # Check for model interpretability if model and feature names are provided
    if model and feature_names:
        report['interpretability'] = explain_model_importance(model, feature_names).to_dict(orient='records')
    
    return report

# Example usage with simulated data
data = pd.DataFrame({
    'gender': np.random.choice(['male', 'female'], size=100),
    'race': np.random.choice(['group_1', 'group_2', 'group_3'], size=100),
    'age': np.random.randint(18, 70, size=100),
    'income': np.random.randint(20000, 100000, size=100)
})

# Simulated feature importance for a model (optional)
feature_importances = np.random.rand(4)
model = type('Model', (object,), {'feature_importances_': feature_importances})()

# Generate compliance report
protected_columns = ['gender', 'race']
compliance_report = generate_compliance_report(data, protected_columns, model, ['gender', 'race', 'age', 'income'])
