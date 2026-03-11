# Configuration & Implementation Details

## Project Configuration

### Environment Setup

**Python Version**: 3.8+
**Virtual Environment**: `.venv`
**Package Manager**: pip

```bash
# Create environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

---

## Dataset Configuration

### Input Data Source
- **File**: `Dataset/StudentPerformanceFactors.csv`
- **Records**: 6,607 students
- **Features**: 20 (19 features + 1 target)
- **Format**: CSV

### Data Preprocessing Settings

| Step | Configuration |
|------|---------------|
| Missing Values | Median fill (numerical), Mode fill (categorical) |
| Duplicates | Removed completely |
| Categorical Encoding | LabelEncoder for all object columns |
| Feature Scaling | StandardScaler (fit on training data) |
| Train-Test Split | 80-20 split (random_state=42) |

### Features Used (19 total)

**Numerical Features** (11):
- Hours_Studied
- Attendance
- Sleep_Hours
- Previous_Scores
- Tutoring_Sessions
- Parental_Involvement
- Access_to_Resources
- Motivation_Level
- Internet_Access
- Physical_Activity
- Distance_from_Home

**Categorical Features** (8):
- Extracurricular_Activities
- School_Type
- Peer_Influence
- Learning_Disabilities
- Teacher_Quality
- Family_Income
- Parental_Education_Level
- Gender

**Target Variable**: Exam_Score (numerical, 0-100)

---

## Model Configuration

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# Uses default scikit-learn parameters:
# - fit_intercept: True
# - normalize: False
# - copy_X: True
# - n_jobs: None
```

**Training**: Simple, closed-form solution
**Complexity**: O(n × p²) where n=samples, p=features
**Interpretability**: High (coefficients show feature impact)

### Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,        # 100 decision trees
    max_depth=20,            # Max tree depth
    min_samples_split=5,     # Min samples to split node
    min_samples_leaf=2,      # Min samples in leaf
    random_state=42,         # Reproducibility
    n_jobs=-1               # Use all processors
)
```

**Training**: Ensemble of 100 decision trees
**Complexity**: O(log(n) × p × d) per prediction
**Interpretability**: Medium (feature importance available)

---

## Feature Scaling

### StandardScaler Settings
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(
    copy=True,              # Copy input data
    with_mean=True,         # Center features
    with_std=True           # Scale to unit variance
)

# Formula: (X - mean) / std
```

### Scaling Per Feature (Training Set)
- Calculated on **training data only**
- Applied to both training and testing data
- **Same scaler used in production** (saved in `scaler.pkl`)

---

## Label Encoding Configuration

### Categorical Features Encoding
```python
from sklearn.preprocessing import LabelEncoder

# Applied to these columns:
categorical_cols = [
    'Parental_Involvement',      # Low, Medium, High → 0, 1, 2
    'Access_to_Resources',        # Low, Medium, High → 0, 1, 2
    'Extracurricular_Activities', # No, Yes → 0, 1
    'Motivation_Level',           # Low, Medium, High → 0, 1, 2
    'Internet_Access',            # No, Yes → 0, 1
    'Teacher_Quality',            # Low, Medium, High → 0, 1, 2
    'School_Type',                # Private, Public → 0, 1
    'Peer_Influence',             # Negative, Neutral, Positive → 0, 1, 2
    'Learning_Disabilities',      # No, Yes → 0, 1
    'Family_Income',              # Low, Medium, High → 0, 1, 2
    'Parental_Education_Level',   # High School, College, Postgraduate → 0, 1, 2
    'Distance_from_Home',         # Near, Moderate, Far → 0, 1, 2
    'Gender'                       # Female, Male → 0, 1
]
```

---

## Data Split Configuration

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% for testing
    random_state=42          # Reproducible split
)
```

### Split Results
- **Total Samples**: 6,607
- **Training Set**: 5,285 samples (80%)
- **Testing Set**: 1,322 samples (20%)

---

## Model Performance Metrics

### Linear Regression Performance
```
Mean Squared Error (MSE):      4.3993
Root Mean Squared Error (RMSE): 2.0974
Mean Absolute Error (MAE):     1.0155
R² Score:                      0.6888

Interpretation:
- MSE: Average squared error of 4.4 points
- RMSE: Typical error of ~2.1 points
- MAE: Average absolute error of 1.0 point
- R²: Model explains 69% of variance
```

### Random Forest Performance
```
Mean Squared Error (MSE):      4.6553
Root Mean Squared Error (RMSE): 2.1576
Mean Absolute Error (MAE):     1.0992
R² Score:                      0.6707

Interpretation:
- MSE: Average squared error of 4.7 points
- RMSE: Typical error of ~2.2 points
- MAE: Average absolute error of 1.1 points
- R²: Model explains 67% of variance
```

### Best Model
**Linear Regression** (selected based on R² score)
- Better generalization
- Lower MSE
- More interpretable

---

## Feature Importance Analysis

### Correlation with Target (Exam_Score)

Top 10 Correlated Features:
```
1. Attendance                 0.5811 ★★★★★ (Very Strong)
2. Hours_Studied              0.4455 ★★★★☆ (Strong)
3. Previous_Scores            0.1751 ★☆☆☆☆ (Weak)
4. Tutoring_Sessions          0.1565 ★☆☆☆☆ (Weak)
5. Peer_Influence             0.1002 ☆☆☆☆☆ (Very Weak)
6. Distance_from_Home         0.0889 ☆☆☆☆☆ (Very Weak)
7. Extracurricular_Activities 0.0644 ☆☆☆☆☆ (Very Weak)
8. Internet_Access            0.0515 ☆☆☆☆☆ (Very Weak)
9. Parental_Education_Level   0.0446 ☆☆☆☆☆ (Very Weak)
10. Physical_Activity          0.0278 ☆☆☆☆☆ (Very Weak)
```

---

## Model Persistence Configuration

### Files Saved
```
model/
├── linear_regression_model.pkl    # Main model 1
├── random_forest_model.pkl        # Main model 2
├── scaler.pkl                     # Feature normalization
├── label_encoders.pkl             # Categorical encoding
├── feature_names.pkl              # Feature ordering
└── training_summary.txt           # Summary statistics
```

### Serialization Method
- **Library**: Python's built-in `pickle`
- **Binary Format**: .pkl files
- **Size**: ~100KB total for all models

### Loading Models
```python
import pickle

with open('model/linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## Streamlit Configuration

### App Settings
```python
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Input Ranges (Streamlit Sliders)
| Input | Min | Max | Step | Default |
|-------|-----|-----|------|---------|
| Study Hours | 0 | 50 | 0.5 | 15 |
| Attendance | 0 | 100 | 1 | 85 |
| Previous Score | 0 | 100 | 1 | 75 |
| Sleep Hours | 0 | 12 | 0.5 | 7 |
| Tutoring Sessions | 0 | 20 | 1 | 2 |

### Dropdown Options
| Input | Options |
|-------|---------|
| Internet Access | Yes, No |
| Motivation Level | Low, Medium, High |
| Extracurricular | Yes, No |

---

## Visualization Configuration

### Matplotlib Settings
```python
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline
```

### Plot Configurations

**Scatter Plots**
- Figure size: (14, 10)
- Alpha: 0.6 (transparency)
- Grid: True with alpha 0.3

**Heatmap**
- Figure size: (16, 12)
- Colormap: 'coolwarm'
- Annotations: 2 decimal places
- Center: 0

**Prediction Plots**
- Figure size: (14, 5)
- Reference line: Red dashed (perfect prediction)
- Alpha: 0.6

---

## Jupyter Notebook Configuration

### Cell Execution Order
1. Imports and configuration
2. Data loading
3. Missing value handling
4. Duplicate removal
5. Feature encoding
6. EDA visualizations (scatter plots)
7. Correlation heatmap
8. Data preparation for modeling
9. Linear Regression training
10. Random Forest training
11. Model evaluation
12. Prediction visualization
13. Model persistence

### Notebook Settings
```python
%matplotlib inline
warnings.filterwarnings('ignore')
```

---

## Deployment Configuration

### Streamlit Cloud (Recommended)
```bash
# Requirements:
# - GitHub repository with code
# - requirements.txt with dependencies
# - .streamlit/config.toml (optional)

# Deploy:
# 1. Push to GitHub
# 2. Connect to Streamlit Cloud
# 3. Deploy automatically
```

### Local Deployment
```bash
# Run on local machine
streamlit run app.py

# Access at: http://localhost:8501
```

### Requirements File Contents
```
pandas==2.3.3
numpy==2.4.3
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.8.0
streamlit==1.55.0
jupyter==1.0.0
ipykernel==7.2.0
```

---

## Performance Optimization

### Data Processing
- ✓ Vectorized operations (NumPy/Pandas)
- ✓ No loops for preprocessing
- ✓ Efficient memory usage

### Model Training
- ✓ Random Forest uses all processors (n_jobs=-1)
- ✓ Pre-fitted scaler (no re-computation)
- ✓ Cached model loading in Streamlit (@st.cache_resource)

### Prediction Speed
- Linear Regression: <1ms per prediction
- Random Forest: <5ms per prediction
- Feature scaling: <1ms per prediction

---

## Monitoring & Maintenance

### Model Monitoring
- Track prediction accuracy over time
- Monitor input feature distributions
- Alert if performance degrades

### Maintenance Tasks
- Retrain quarterly with new data
- Update test set performance metrics
- Version control model files
- Document any configuration changes

---

## Error Handling

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Models not found | Missing files | Run notebook first |
| Scaling error | Wrong feature order | Check feature_names.pkl |
| Prediction error | Invalid inputs | Validate input ranges |
| Encoding error | Missing categories | Use label encoders |

---

## References

### Libraries Used
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **matplotlib/seaborn**: Visualization
- **streamlit**: Web interface
- **pickle**: Model serialization

### Model References
- Linear Regression: Simple least-squares approach
- Random Forest: Ensemble of decision trees with bootstrap aggregation

---

**Configuration Last Updated**: March 11, 2026
