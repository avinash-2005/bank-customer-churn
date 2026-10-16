# Customer Churn Prediction System

**Behavior + Emotion Based Customer Churn Prediction System for Digital Banking**

A comprehensive machine learning project that combines behavioral analytics and sentiment analysis to predict customer churn in digital banking platforms.

---

## 🎯 Project Overview

This project demonstrates a production-ready customer churn prediction system that analyzes both customer behavior patterns and emotional sentiment from feedback to provide accurate churn risk assessments. The system features a professional Streamlit dashboard with banking-grade UI/UX design.

### Key Features

- **🔍 Behavioral Analytics**: Analyzes login frequency, session duration, inactivity periods, and transaction patterns
- **💭 Sentiment Analysis**: Processes customer feedback using NLP (VADER + TextBlob) to extract emotional insights
- **🤖 Machine Learning**: Random Forest classifier with combined behavioral and sentiment features
- **📊 Professional Dashboard**: Clean, modern FinTech interface with real-time analytics
- **🎨 Banking-Grade UI**: Professional design with consistent color scheme and typography

---

## 🏗️ Project Structure

```
project_root/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # Dataset files
│   ├── customer_behavior.csv  # Behavioral analytics data
│   └── customer_feedback.csv  # Customer feedback text data
│
├── models/                    # Trained ML models (generated after training)
│   ├── churn_model.pkl        # Random Forest model
│   └── scaler.pkl             # Feature scaler
│
├── src/                       # Source code modules
│   ├── preprocessing.py       # Data cleaning & preprocessing
│   ├── sentiment.py           # NLP sentiment analysis
│   ├── model.py               # ML model training & prediction
│   └── utils.py               # Helper functions & utilities
│
└── assets/                    # Static assets
    └── styles.css             # Custom CSS styling
```

---

## 🛠️ Technology Stack

### Backend & ML
- **Python 3.10+**: Core programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **joblib**: Model serialization

### NLP & Sentiment Analysis
- **VADER**: Valence Aware Dictionary and sEntiment Reasoner
- **TextBlob**: Simplified text processing
- **NLTK**: Natural Language Toolkit

### Frontend & Visualization
- **Streamlit**: Web application framework
- **Plotly**: Interactive charts and visualizations
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd customer-churn-prediction
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

---

## 📊 Data Description

### Customer Behavioral Data (`customer_behavior.csv`)

| Column | Description | Type |
|--------|-------------|------|
| customer_id | Unique customer identifier | Integer |
| login_frequency | Number of logins per week | Float |
| session_duration | Average session duration in minutes | Float |
| inactivity_days | Days since last activity | Integer |
| transaction_count | Number of transactions | Integer |
| churn | Churn status (0=No, 1=Yes) | Integer |

### Customer Feedback Data (`customer_feedback.csv`)

| Column | Description | Type |
|--------|-------------|------|
| customer_id | Unique customer identifier | Integer |
| feedback_text | Customer feedback message | String |

---

## 🧠 Model Architecture

### Feature Engineering

1. **Behavioral Features**
   - Login frequency (normalized)
   - Session duration (normalized)
   - Inactivity days (normalized)
   - Transaction count (normalized)

2. **Sentiment Features**
   - Sentiment score (-1 to 1 scale)
   - Sentiment label (Positive/Neutral/Negative)

### Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Features**: 5 total (4 behavioral + 1 sentiment)
- **Target**: Binary churn prediction (0/1)
- **Evaluation**: Accuracy, Precision, Recall, F1-Score

### Risk Classification

- **Low Risk**: Churn probability < 30%
- **Medium Risk**: Churn probability 30-70%
- **High Risk**: Churn probability > 70%

---

## 🎨 UI/UX Design

### Color Palette

- **Primary**: Deep Blue (#0B1C2D)
- **Secondary**: Slate Gray (#5A6C7D)
- **Accent**: Teal (#1FB6A6)
- **Background**: Light Gray (#F5F7FA)
- **Cards**: White (#FFFFFF)

### Design Principles

- **Minimal & Clean**: Banking-grade aesthetics
- **Professional**: Trustworthy and reliable appearance
- **Accessible**: Clear typography and readable metrics
- **Responsive**: Works on desktop and tablet devices

---

## 📱 Application Features

### Dashboard
- **KPI Metrics**: Total customers, average churn probability, high-risk count
- **Risk Distribution**: Visual breakdown of customer risk levels
- **Sentiment Analysis**: Customer sentiment distribution
- **Feature Importance**: ML model feature relevance

### Customer Analysis
- **Individual Customer View**: Detailed customer metrics
- **Risk Assessment**: Real-time churn probability with gauge chart
- **Sentiment Analysis**: Customer feedback sentiment analysis
- **Behavioral Metrics**: Login patterns and transaction history

### Model Metrics
- **Model Information**: Algorithm type and training details
- **Feature Importance**: Detailed feature relevance analysis
- **Performance Metrics**: Model evaluation statistics

---

## 🔧 Development & Testing

### Running Individual Modules

```bash
# Test preprocessing
python src/preprocessing.py

# Test sentiment analysis
python src/sentiment.py

# Test model training
python src/model.py

# Test utilities
python src/utils.py
```

### Model Training Pipeline

The system automatically handles:
1. Data loading and validation
2. Missing value imputation
3. Text preprocessing and sentiment analysis
4. Feature normalization and scaling
5. Model training with cross-validation
6. Model persistence (pickle files)

---

## 📈 Model Performance

### Evaluation Metrics

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive prediction accuracy
- **Recall**: Ability to identify all churn cases
- **F1-Score**: Harmonic mean of precision and recall

### Feature Importance

The model typically finds these features most important:
1. Inactivity days
2. Login frequency
3. Sentiment score
4. Transaction count
5. Session duration

---

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Production Deployment

For production deployment, consider:

1. **Streamlit Cloud**: Deploy directly to Streamlit's cloud platform
2. **Docker**: Containerize the application
3. **Cloud Services**: AWS, Google Cloud, or Azure deployment
4. **Heroku**: Platform-as-a-Service deployment

### Environment Variables

Create a `.env` file for configuration:

```env
# Model paths
MODEL_PATH=models/churn_model.pkl
SCALER_PATH=models/scaler.pkl

# Data paths
BEHAVIOR_DATA=data/customer_behavior.csv
FEEDBACK_DATA=data/customer_feedback.csv
```

---

## 🤝 Contributing

### Development Guidelines

1. **Code Style**: Follow PEP 8 Python style guidelines
2. **Documentation**: Add docstrings to all functions and classes
3. **Testing**: Write unit tests for new features
4. **Commits**: Use clear, descriptive commit messages

### Project Structure for Contributors

- `src/`: Core application logic
- `data/`: Dataset files (sample data included)
- `models/`: Trained model files (generated)
- `assets/`: Static assets (CSS, images)
- `tests/`: Unit tests (to be added)

---

## 📚 Academic Context

### Project Information

- **Type**: College Minor Project
- **Domain**: FinTech / Digital Banking
- **Focus**: Machine Learning + NLP Application
- **Innovation**: Combined behavioral and sentiment analysis

### Learning Objectives

- Practical ML pipeline development
- NLP sentiment analysis implementation
- Professional UI/UX design
- Real-world business problem solving

---

## 🔒 Security & Privacy

### Data Considerations

- **Sample Data**: Project uses anonymized sample data
- **Privacy**: No real customer information is stored
- **Security**: Follow banking industry security practices

### Best Practices

- Input validation and sanitization
- Secure model serialization
- Proper error handling
- Logging and monitoring

---

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   # Ensure you're in the project root
   cd customer-churn-prediction
   pip install -r requirements.txt
   ```

2. **Model Loading Issues**
   ```bash
   # Train the model first by clicking "Load & Train Model"
   # Or run training script manually
   python src/model.py
   ```

3. **Port Already in Use**
   ```bash
   # Kill existing Streamlit process
   pkill -f streamlit
   
   # Or use different port
   streamlit run app.py --server.port 8502
   ```

### Performance Optimization

- Use smaller datasets for initial testing
- Clear model cache if needed
- Monitor memory usage with large datasets

---

## 📄 License

This project is developed for academic purposes. Please ensure proper attribution if using components of this code.

---

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the code documentation
3. Create an issue in the project repository
4. Contact the project maintainer

---

## 🎉 Acknowledgments

- **Streamlit Team**: For the excellent web app framework
- **Scikit-learn**: For comprehensive ML algorithms
- **VADER/TextBlob**: For sentiment analysis tools
- **Plotly**: For interactive visualizations

---

**Project Status**: ✅ Complete and Ready for Demonstration

**Last Updated**: February 2026

**Version**: 1.0.0
