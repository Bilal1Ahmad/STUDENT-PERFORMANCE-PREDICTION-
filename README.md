# Student Performance Prediction System

A complete Data Science project for predicting student exam scores using machine learning models. The system includes data preprocessing, exploratory data analysis, model training, and a web application for making predictions.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Performance](#models--performance)
- [Requirements](#requirements)

## 📚 Project Overview

This project demonstrates a complete end-to-end machine learning workflow:

1. **Data Preprocessing**: Handle missing values, remove duplicates, encode categorical variables
2. **Exploratory Data Analysis (EDA)**: Visualize relationships and correlations
3. **Model Training**: Train Linear Regression and Random Forest models
4. **Model Evaluation**: Compare performance metrics (MSE, RMSE, MAE, R²)
5. **Web Application**: Interactive Streamlit app for predictions

## ✨ Features

### Data Processing
- ✅ Missing value imputation (median for numerical, mode for categorical)
- ✅ Duplicate removal
- ✅ Label encoding for categorical variables
- ✅ Feature scaling using StandardScaler

### Exploratory Data Analysis
- 📊 Study Hours vs Final Score scatter plot
- 📊 Attendance vs Final Score scatter plot
- 📊 Previous Score vs Final Score scatter plot
- 📊 Sleep Hours vs Final Score scatter plot
- 🔥 Comprehensive correlation heatmap

### Machine Learning Models
- 🤖 **Linear Regression**: Simple, interpretable model
- 🌲 **Random Forest**: Ensemble method capturing non-linear patterns

### Performance Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### Web Application
- 🎨 Beautiful, user-friendly interface
- 📝 Input fields for student information:
  - Study hours per week
  - Attendance percentage
  - Previous scores
  - Sleep hours per night
  - Internet access
  - Motivation level
  - Extracurricular activities
  - Tutoring sessions
- 🎯 Real-time predictions from both models
- 📊 Performance assessment and recommendations
- 💡 Key insights about important factors

## 📁 Project Structure

```
Student-Performance-Prediction-System/
│
├── Dataset/
│   └── StudentPerformanceFactors.csv    # Raw dataset with 6607 records
│
├── model/
│   ├── linear_regression_model.pkl      # Trained Linear Regression model
│   ├── random_forest_model.pkl          # Trained Random Forest model
│   ├── scaler.pkl                       # Feature scaler
│   ├── label_encoders.pkl               # Categorical encoders
│   ├── feature_names.pkl                # Feature names list
│   └── training_summary.txt             # Training summary
│
├── notebook/
│   └── Analysis.ipynb                   # Jupyter notebook with:
│                                        # - Data loading & preprocessing
│                                        # - EDA with visualizations
│                                        # - Model training & evaluation
│                                        # - Model persistence
│
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 📊 Dataset

**File**: `StudentPerformanceFactors.csv`

**Size**: 6,607 records × 20 features

**Target Variable**: `Exam_Score` (Student's final exam score)

**Key Features**:
- `Hours_Studied`: Hours spent studying per week
- `Attendance`: Percentage of classes attended
- `Sleep_Hours`: Average hours of sleep per night
- `Previous_Scores`: Average score from previous assessments
- `Internet_Access`: Whether student has internet access
- `Tutoring_Sessions`: Number of tutoring sessions per month
- And 14 other educational and personal factors

**Data Quality**:
- Missing values: Handled with median/mode imputation
- Duplicates: None found
- Outliers: None removed (kept for realistic predictions)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Step 1: Clone or Download the Project
```bash
cd Student-Performance-Prediction-System
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit jupyter
```

## 💻 Usage

### Option 1: Run Analysis Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open and Run** `notebook/Analysis.ipynb`
   - Executes full pipeline
   - Generates visualizations
   - Trains and saves models
   - Creates model files in `model/` directory

### Option 2: Launch Streamlit Web App

**Prerequisite**: Run the notebook first to generate model files

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Using the Web App**:
1. Enter student information using the sliders and dropdowns
2. View predictions from both Linear Regression and Random Forest models
3. See the average prediction with assessment
4. Review key insights about important factors

## 📊 Models & Performance

### Linear Regression
```
Mean Squared Error (MSE):    4.3993
Root Mean Squared Error (RMSE): 2.0974
Mean Absolute Error (MAE):   1.0155
R² Score:                    0.6888
```
- **Strengths**: Simple, interpretable, fast
- **Use Case**: When linear relationships are important

### Random Forest
```
Mean Squared Error (MSE):    4.6553
Root Mean Squared Error (RMSE): 2.1576
Mean Absolute Error (MAE):   1.0992
R² Score:                    0.6707
```
- **Strengths**: Captures non-linear patterns, feature interactions
- **Use Case**: When model complexity is acceptable

### Best Model: Linear Regression
- Lowest MSE and highest R² score
- Average prediction error: ~2.1 points

## 🔍 Key Insights

### Feature Importance (Correlations with Exam Score)
1. **Attendance** (0.581) - Most important factor
2. **Hours Studied** (0.445) - Strong positive impact
3. **Previous Scores** (0.175) - Moderate indicator
4. **Tutoring Sessions** (0.157) - Helpful support
5. **Other Factors**: Sleep hours, motivation, internet access, etc.

### Recommendations
- 📚 **Increase study hours**: Each hour contributes ~1.7 points
- 📊 **Improve attendance**: Attendance is the strongest predictor
- 😴 **Maintain healthy sleep**: 7-8 hours recommended
- 👥 **Consider tutoring**: Extra sessions help consolidate knowledge

## 📋 Requirements

```
pandas==2.3.3
numpy==2.4.3
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.8.0
streamlit==1.55.0
jupyter==1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

## 📈 Workflow Summary

```
Raw Data
    ↓
Missing Value Handling
    ↓
Duplicate Removal
    ↓
Categorical Encoding
    ↓
Feature Scaling
    ↓
Train-Test Split
    ↓
Model Training
    ├── Linear Regression
    └── Random Forest
    ↓
Model Evaluation
    ↓
Model Persistence (Pickle)
    ↓
Streamlit Web App
    ↓
User Predictions
```

## 🔧 Technical Details

### Data Pipeline
- **Test Split**: 80-20 (5,285 training, 1,322 testing samples)
- **Scaler**: StandardScaler for feature normalization
- **Cross-validation**: 20% held-out test set

### Model Configuration
- **Linear Regression**: Scikit-learn default parameters
- **Random Forest**: 
  - n_estimators: 100 trees
  - max_depth: 20
  - min_samples_split: 5
  - min_samples_leaf: 2

### Feature Engineering
- Label encoding for categorical variables
- No feature selection (all 19 features used)
- No polynomial features
- Standardized scaling for ML models

## 📝 Code Organization

### Notebook Sections
1. **Imports & Configuration**: Libraries and visualization settings
2. **Data Loading**: CSV reading and exploration
3. **Data Preprocessing**: Cleaning and encoding
4. **Exploratory Data Analysis**: Visualizations and correlations
5. **Model Training**: Linear Regression & Random Forest
6. **Model Evaluation**: Metrics and comparisons
7. **Model Persistence**: Saving trained models

### App Components
- **Model Loading**: @st.cache_resource for efficiency
- **User Inputs**: Streamlit widgets (sliders, selectboxes)
- **Predictions**: Real-time calculations
- **Visualizations**: Data summary tables
- **Insights**: Key findings and recommendations

## 🎯 Future Enhancements

- [ ] Gradient Boosting models (XGBoost, LightGBM)
- [ ] Feature importance visualization
- [ ] Prediction confidence intervals
- [ ] Student cohort comparison
- [ ] Historical performance tracking
- [ ] Hyperparameter tuning UI
- [ ] Model comparison dashboard
- [ ] Data upload functionality

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Created as a comprehensive Data Science project demonstrating:
- Data preprocessing and cleaning
- Exploratory data analysis
- Machine learning model training
- Model evaluation and comparison
- Deployment with Streamlit

## 📞 Support

For issues or questions:
1. Check the notebook for detailed implementation
2. Review model configuration in the app.py
3. Ensure all dependencies are installed correctly
4. Verify model files exist in the `model/` directory

---

**Happy Predicting! 🎓📈**
# STUDENT-PERFORMANCE-PREDICTION-
