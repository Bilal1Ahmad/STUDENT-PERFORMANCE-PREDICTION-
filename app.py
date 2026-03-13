

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# LOAD MODELS AND PREPROCESSING OBJECTS
@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects from pickle files."""
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


# Load models at startup
try:
    lr_model, rf_model, scaler, label_encoders, feature_names = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# STREAMLIT APP LAYOUT
# Header
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
        📚 Student Performance Prediction System
    </h1>
    <p style='text-align: center; color: #666;'>
        Predict student exam scores using machine learning
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

if models_loaded:
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Student Information")
        st.info("Enter student details to get a prediction")
        
        # Create input fields with default values
        hours_studied = st.slider(
            "📖 Hours Studied (per week)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.5,
            help="Average hours spent studying each week"
        )
        
        attendance = st.slider(
            "📊 Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
            help="Percentage of classes attended"
        )
        
        previous_score = st.slider(
            "📈 Previous Scores (average)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
            help="Average score from previous assessments"
        )
        
        sleep_hours = st.slider(
            "😴 Sleep Hours (per night)",
            min_value=0.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
            help="Average hours of sleep per night"
        )
        
        # Additional information
        st.subheader("📚 Additional Factors")
        
        internet_access = st.selectbox(
            "🌐 Internet Access",
            ["Yes", "No"],
            help="Do you have access to internet for studies?"
        )
        
        motivation_level = st.selectbox(
            "💪 Motivation Level",
            ["Low", "Medium", "High"],
            help="Assess your current motivation level"
        )
        
        extracurricular = st.selectbox(
            "🎭 Extracurricular Activities",
            ["Yes", "No"],
            help="Do you participate in extracurricular activities?"
        )
        
        tutoring_sessions = st.slider(
            "👨‍🏫 Tutoring Sessions (per month)",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
            help="Number of tutoring sessions attended per month"
        )
    
    # Right column for predictions and visualizations
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Prepare data for prediction
        try:
            # Create a dictionary with all input values
            input_data = {
                'Hours_Studied': hours_studied,
                'Attendance': attendance,
                'Previous_Scores': previous_score,
                'Sleep_Hours': sleep_hours,
                'Internet_Access': 1 if internet_access == 'Yes' else 0,
                'Motivation_Level': 0 if motivation_level == 'Low' else (1 if motivation_level == 'Medium' else 2),
                'Extracurricular_Activities': 1 if extracurricular == 'Yes' else 0,
                'Tutoring_Sessions': tutoring_sessions,
            }
            
            # We need to create a full DataFrame with all features
            # For simplicity, we'll create default values for other features
            full_input = {}
            for feature in feature_names:
                if feature in input_data:
                    full_input[feature] = input_data[feature]
                elif feature in label_encoders:
                    # Use mode value for categorical features not specified
                    full_input[feature] = 0
                else:
                    # Use mean value for numerical features not specified
                    full_input[feature] = 0
            
            # Create DataFrame
            input_df = pd.DataFrame([full_input])
            
            # Ensure correct order of features
            input_df = input_df[feature_names]
            
            # Scale the features
            input_scaled = scaler.transform(input_df)
            
            # Make predictions
            lr_prediction = lr_model.predict(input_scaled)[0]
            rf_prediction = rf_model.predict(input_scaled)[0]
            
            # Display predictions in nice boxes
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.metric(
                    label="Linear Regression Prediction",
                    value=f"{lr_prediction:.1f}",
                    delta=None,
                    help="Predicted score using Linear Regression model"
                )
            
            with col2b:
                st.metric(
                    label="Random Forest Prediction",
                    value=f"{rf_prediction:.1f}",
                    delta=None,
                    help="Predicted score using Random Forest model"
                )
            
            # Calculate average prediction
            avg_prediction = (lr_prediction + rf_prediction) / 2
            
            st.success(f"### Average Prediction: {avg_prediction:.1f}/100")
            
            # Assess performance
            if avg_prediction >= 90:
                st.balloons()
                assessment = "🌟 Excellent! Outstanding performance expected!"
            elif avg_prediction >= 80:
                assessment = "✅ Very Good! Great performance expected!"
            elif avg_prediction >= 70:
                assessment = "👍 Good! Satisfactory performance expected!"
            elif avg_prediction >= 60:
                assessment = "📌 Fair! Consider improving study habits!"
            else:
                assessment = "⚠️ Low! Significant improvement needed!"
            
            st.info(assessment)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            summary_data = {
                'Study Hours': hours_studied,
                'Attendance (%)': attendance,
                'Previous Score': previous_score,
                'Sleep Hours': sleep_hours,
                'Internet Access': internet_access,
                'Motivation Level': motivation_level,
                'Extracurricular': extracurricular,
                'Tutoring Sessions': tutoring_sessions
            }
            
            summary_df = pd.DataFrame(list(summary_data.items()), 
                                     columns=['Factor', 'Value'])
            st.table(summary_df)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please check your input values and try again.")
    
    # Full-width section for insights
    st.markdown("---")
    st.subheader("💡 Key Insights & Recommendations")
    
    col_insights1, col_insights2, col_insights3 = st.columns(3)
    
    with col_insights1:
        st.markdown("""
        #### Study Hours Impact
        - **Most Important Factor** ✓
        - Strong positive correlation (0.45)
        - Each additional hour helps!
        """)
    
    with col_insights2:
        st.markdown("""
        #### Attendance Matters
        - **Very Important** ✓
        - Strongest correlation (0.58)
        - Don't miss classes!
        """)
    
    with col_insights3:
        st.markdown("""
        #### Previous Performance
        - **Moderate Factor** ✓
        - Correlation (0.18)
        - Consistent improvement possible
        """)
    
    # Additional Statistics Section
    st.markdown("---")
    st.subheader("📊 Model Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **Linear Regression Model**
        - R² Score: 0.6888
        - Mean Absolute Error: 1.02
        - Simple and interpretable
        """)
    
    with col_info2:
        st.markdown("""
        **Random Forest Model**
        - R² Score: 0.6707
        - Mean Absolute Error: 1.10
        - Captures non-linear patterns
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>📚 Student Performance Prediction System v1.0</p>
        <p>Built with Streamlit, Scikit-learn, and Python</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Could not load models. Please ensure all model files are in the './model' directory.")
    st.info("Run the Jupyter notebook first to train and save the models.")
