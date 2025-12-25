# Stress Level Prediction & Mental Health Advisor

A comprehensive web-based application that predicts stress levels using machine learning and provides personalized mental health advice. This system analyzes 20 different stress-related factors to assess an individual's stress level and offers tailored recommendations.

## 🎯 Project Overview

This application combines machine learning with mental health guidance to:
- **Assess Stress Levels**: Predict whether a user has low, moderate, or high stress
- **Provide Personalized Advice**: Generate actionable recommendations based on stress level
- **Identify Key Factors**: Use explainable AI to highlight the most significant stress contributors
- **User-Friendly Interface**: Interactive web interface with intuitive input forms

## 🏗️ Project Structure

```
├── app.py                          # Main Flask application
├── stress_prediction.py             # Stress prediction testing module
├── advice_engine.py                 # Advice generation logic
├── model_training.py                # ML model training script
├── check_setup.py                   # Environment validation
├── test_dataset.py                  # Dataset verification
├── dataset/
│   └── StressLevelDataset.csv       # Training dataset
├── models/
│   ├── stress_model.pkl             # Trained RandomForest model
│   └── scaler.pkl                   # StandardScaler for feature normalization
├── templates/
│   ├── welcome.html                 # User welcome & name entry
│   ├── guide.html                   # Information guide
│   ├── measure.html                 # Stress measurement form
│   ├── result.html                  # Prediction results & advice
│   └── index.html                   # Alternative UI template
├── static/
│   ├── style.css                    # Custom styling
│   └── images/                      # UI images
└── README.md                        # This file
```

## 📊 Features

### 20 Stress-Related Factors
The application evaluates stress across multiple dimensions:

**Psychological Factors**
- Anxiety Level
- Self-Esteem
- Mental Health History
- Depression

**Physical Health Indicators**
- Headache
- Blood Pressure
- Sleep Quality
- Breathing Problems

**Environmental Factors**
- Noise Level
- Living Conditions
- Safety
- Basic Needs

**Academic/Professional Factors**
- Academic Performance
- Study Load
- Teacher-Student Relationship
- Future Career Concerns

**Social Factors**
- Social Support
- Peer Pressure
- Extracurricular Activities
- Bullying

### Stress Levels
The system classifies stress into three categories:
- **Low Stress** (Level 1): You are doing well
- **Moderate Stress** (Level 2): Maintain work-life balance
- **High Stress** (Level 3): Professional help recommended

### Personalized Recommendations
Each stress level includes tailored advice:
- **Low Stress**: Maintain healthy habits
- **Moderate Stress**: Engagement in physical activities, regular breaks
- **High Stress**: Sleep improvement, meditation, workload reduction, professional consultation

### Explainable AI
The results page displays the **top 3 contributing factors** to the stress prediction, helping users understand which aspects most influenced their score.

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn
  - Model: Random Forest Classifier (300 estimators)
  - Preprocessing: StandardScaler
- **Frontend**: Bootstrap 5, HTML5, CSS3
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: joblib
- **Data Format**: CSV

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Required Libraries
```
flask
numpy
pandas
scikit-learn
joblib
matplotlib
seaborn
```

### Installation

1. **Clone or download the project**
   ```bash
   cd "My project"
   ```

2. **Verify setup** (check all dependencies are installed)
   ```bash
   python check_setup.py
   ```

3. **Verify dataset**
   ```bash
   python test_dataset.py
   ```

4. **Train the model** (if models/ folder doesn't exist or needs retraining)
   ```bash
   python model_training.py
   ```

## 📈 Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to: `http://localhost:5000`

3. **Use the Application**
   - Enter your name on the welcome page
   - Read the stress information guide
   - Rate your stress levels (1-10) for each of the 20 factors
   - Receive personalized prediction and advice

## 📝 How It Works

### 1. Data Collection
Users rate themselves on 20 stress-related factors using a scale of 1-10.

### 2. Feature Scaling
Input values are normalized using StandardScaler to match the training data distribution.

### 3. Prediction
The trained Random Forest model classifies the stress level:
- Class 0 → Low Stress
- Class 1 → Moderate Stress
- Class 2 → High Stress

### 4. Advice Generation
Based on the predicted stress level, the system generates relevant recommendations.

### 5. Explainability
Feature importance scores identify the top 3 factors contributing to the prediction.

## 🔧 Key Files Description

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application with route handlers |
| `model_training.py` | Trains Random Forest model on dataset |
| `stress_prediction.py` | Testing module for stress predictions |
| `advice_engine.py` | Generates personalized advice based on stress level |
| `check_setup.py` | Validates all required libraries are installed |
| `test_dataset.py` | Verifies dataset integrity and structure |

## 📊 Model Details

**Algorithm**: Random Forest Classifier
- **Estimators**: 300 trees
- **Random State**: 42 (for reproducibility)
- **Class Weights**: Balanced (handles class imbalance)
- **Train-Test Split**: 80-20 ratio
- **Scaling**: StandardScaler

**Performance Metrics**: Includes precision, recall, and F1-score (displayed after training)

## 🌐 Web Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET, POST | Welcome page and username entry |
| `/guide` | GET | Information guide about stress factors |
| `/measure` | GET | Stress measurement form (20 questions) |
| `/result` | POST | Displays predictions, advice, and top factors |

## 🛡️ Security Features

- Session-based username storage
- Secret key configuration for Flask session management
- Input validation and type conversion

## 💡 Usage Tips

1. **Honest Assessment**: Rate factors based on your actual experience
2. **Context**: Consider the past week or month when rating
3. **Professional Help**: For high stress, consider consulting a mental health professional
4. **Regular Monitoring**: Use periodically to track stress changes

## 📌 Notes

- The model requires a properly formatted CSV dataset with the 20 features and a `stress_level` column
- Feature order in the application must match the dataset order
- All features are scaled before prediction
- Session data is temporary and stored server-side

## 🔮 Future Enhancements

- Database integration for user history tracking
- Multiple language support
- Enhanced visualizations and charts
- Integration with mental health resources
- Mobile application version
- Push notifications for stress alerts
- User progress analytics

## 📄 License

This project is created for educational and mental health assessment purposes.

## ✍️ Author

Developed as a comprehensive stress prediction and mental health advisory system.

---

**Last Updated**: December 23, 2025
