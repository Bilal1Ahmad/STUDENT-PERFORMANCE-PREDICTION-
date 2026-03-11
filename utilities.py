"""
========================================================================
UTILITY SCRIPT - MODEL TESTING AND PREDICTION
========================================================================
This script provides utilities for testing the trained models outside
of the Streamlit app. Useful for debugging and batch predictions.
========================================================================
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ========================================================================
# LOAD MODELS AND PREPROCESSING OBJECTS
# ========================================================================

def load_models():
    """Load all trained models and preprocessing objects."""
    model_dir = Path('./model')
    
    # Load models
    with open(model_dir / 'linear_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    
    with open(model_dir / 'random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Load preprocessing objects
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(model_dir / 'label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open(model_dir / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return lr_model, rf_model, scaler, label_encoders, feature_names


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def create_prediction_input(hours_studied, attendance, previous_score, sleep_hours,
                           internet_access='Yes', motivation_level='Medium',
                           extracurricular='No', tutoring_sessions=2):
    """
    Create an input dictionary for predictions.
    
    Parameters:
    -----------
    hours_studied : float
        Hours spent studying per week (0-50)
    attendance : float
        Attendance percentage (0-100)
    previous_score : float
        Previous scores average (0-100)
    sleep_hours : float
        Sleep hours per night (0-12)
    internet_access : str
        'Yes' or 'No'
    motivation_level : str
        'Low', 'Medium', or 'High'
    extracurricular : str
        'Yes' or 'No'
    tutoring_sessions : int
        Number of tutoring sessions per month (0-20)
    
    Returns:
    --------
    dict : Dictionary with all input features ready for prediction
    """
    
    input_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Previous_Scores': previous_score,
        'Sleep_Hours': sleep_hours,
        'Internet_Access': 1 if internet_access == 'Yes' else 0,
        'Motivation_Level': 0 if motivation_level == 'Low' else (
            1 if motivation_level == 'Medium' else 2
        ),
        'Extracurricular_Activities': 1 if extracurricular == 'Yes' else 0,
        'Tutoring_Sessions': tutoring_sessions,
    }
    
    return input_data


def predict_score(lr_model, rf_model, scaler, feature_names, input_dict):
    """
    Make predictions using both models.
    
    Parameters:
    -----------
    lr_model : sklearn model
        Trained Linear Regression model
    rf_model : sklearn model
        Trained Random Forest model
    scaler : StandardScaler
        Fitted scaler for feature normalization
    feature_names : list
        List of feature names in correct order
    input_dict : dict
        Input data dictionary
    
    Returns:
    --------
    tuple : (lr_prediction, rf_prediction, average_prediction)
    """
    
    # Create full input with default values for missing features
    full_input = {}
    for feature in feature_names:
        if feature in input_dict:
            full_input[feature] = input_dict[feature]
        else:
            full_input[feature] = 0  # Default value for missing features
    
    # Create DataFrame
    input_df = pd.DataFrame([full_input])
    input_df = input_df[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make predictions
    lr_pred = lr_model.predict(input_scaled)[0]
    rf_pred = rf_model.predict(input_scaled)[0]
    avg_pred = (lr_pred + rf_pred) / 2
    
    return lr_pred, rf_pred, avg_pred


def assess_prediction(avg_score):
    """
    Provide an assessment of the predicted score.
    
    Parameters:
    -----------
    avg_score : float
        The average predicted score (0-100)
    
    Returns:
    --------
    str : Assessment message
    """
    
    if avg_score >= 90:
        return "🌟 Excellent! Outstanding performance expected!"
    elif avg_score >= 80:
        return "✅ Very Good! Great performance expected!"
    elif avg_score >= 70:
        return "👍 Good! Satisfactory performance expected!"
    elif avg_score >= 60:
        return "📌 Fair! Consider improving study habits!"
    else:
        return "⚠️ Low! Significant improvement needed!"


# ========================================================================
# TESTING & DEMONSTRATION
# ========================================================================

def test_models():
    """Test models with sample predictions."""
    
    print("="*70)
    print("STUDENT PERFORMANCE PREDICTION MODEL - TESTING")
    print("="*70)
    
    # Load models
    print("\n📥 Loading models...")
    lr_model, rf_model, scaler, label_encoders, feature_names = load_models()
    print("✓ Models loaded successfully!")
    
    # Test Case 1: Good Student
    print("\n" + "-"*70)
    print("TEST CASE 1: Good Student")
    print("-"*70)
    
    student1 = create_prediction_input(
        hours_studied=20,
        attendance=95,
        previous_score=85,
        sleep_hours=8,
        internet_access='Yes',
        motivation_level='High',
        extracurricular='Yes',
        tutoring_sessions=3
    )
    
    lr1, rf1, avg1 = predict_score(lr_model, rf_model, scaler, feature_names, student1)
    
    print(f"Study Hours: 20/week")
    print(f"Attendance: 95%")
    print(f"Previous Score: 85")
    print(f"Sleep Hours: 8")
    print(f"Internet Access: Yes")
    print(f"Motivation: High")
    print(f"Extracurricular: Yes")
    print(f"Tutoring: 3 sessions/month")
    print(f"\nPredictions:")
    print(f"  Linear Regression: {lr1:.1f}")
    print(f"  Random Forest: {rf1:.1f}")
    print(f"  Average: {avg1:.1f}")
    print(f"\nAssessment: {assess_prediction(avg1)}")
    
    # Test Case 2: Average Student
    print("\n" + "-"*70)
    print("TEST CASE 2: Average Student")
    print("-"*70)
    
    student2 = create_prediction_input(
        hours_studied=12,
        attendance=75,
        previous_score=70,
        sleep_hours=6.5,
        internet_access='Yes',
        motivation_level='Medium',
        extracurricular='No',
        tutoring_sessions=1
    )
    
    lr2, rf2, avg2 = predict_score(lr_model, rf_model, scaler, feature_names, student2)
    
    print(f"Study Hours: 12/week")
    print(f"Attendance: 75%")
    print(f"Previous Score: 70")
    print(f"Sleep Hours: 6.5")
    print(f"Internet Access: Yes")
    print(f"Motivation: Medium")
    print(f"Extracurricular: No")
    print(f"Tutoring: 1 session/month")
    print(f"\nPredictions:")
    print(f"  Linear Regression: {lr2:.1f}")
    print(f"  Random Forest: {rf2:.1f}")
    print(f"  Average: {avg2:.1f}")
    print(f"\nAssessment: {assess_prediction(avg2)}")
    
    # Test Case 3: Struggling Student
    print("\n" + "-"*70)
    print("TEST CASE 3: Struggling Student")
    print("-"*70)
    
    student3 = create_prediction_input(
        hours_studied=5,
        attendance=60,
        previous_score=55,
        sleep_hours=5,
        internet_access='No',
        motivation_level='Low',
        extracurricular='No',
        tutoring_sessions=0
    )
    
    lr3, rf3, avg3 = predict_score(lr_model, rf_model, scaler, feature_names, student3)
    
    print(f"Study Hours: 5/week")
    print(f"Attendance: 60%")
    print(f"Previous Score: 55")
    print(f"Sleep Hours: 5")
    print(f"Internet Access: No")
    print(f"Motivation: Low")
    print(f"Extracurricular: No")
    print(f"Tutoring: 0 sessions/month")
    print(f"\nPredictions:")
    print(f"  Linear Regression: {lr3:.1f}")
    print(f"  Random Forest: {rf3:.1f}")
    print(f"  Average: {avg3:.1f}")
    print(f"\nAssessment: {assess_prediction(avg3)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Good Student:       {avg1:.1f} (Difference: {abs(lr1-rf1):.1f})")
    print(f"Average Student:    {avg2:.1f} (Difference: {abs(lr2-rf2):.1f})")
    print(f"Struggling Student: {avg3:.1f} (Difference: {abs(lr3-rf3):.1f})")
    print("="*70)
    print("\n✓ Testing complete!")


if __name__ == "__main__":
    # Run tests
    test_models()
    
    print("\n💡 For custom predictions, use the Streamlit app:")
    print("   streamlit run app.py")
