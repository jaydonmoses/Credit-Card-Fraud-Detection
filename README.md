# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

This project implements a machine learning pipeline to detect fraudulent credit card transactions using a credit card fraud dataset. The workflow includes data preprocessing, exploratory data analysis, model training, and evaluation.

## Features
- Data cleaning and preprocessing
- Exploratory data analysis with visualizations
- Model training with Random Forest and cross-validation
- Performance evaluation using classification metrics (confusion matrix, classification report, ROC AUC)

## Technologies Used
- **Python 3.8+**
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Development:** Virtual environment, Git version control

## Project Structure
```
Credit-Card-Fraud-Detection/
├── src/
│   └── fraud_detection.py    # Main analysis script with proper documentation
├── tests/
│   └── test_fraud_detection.py    # Unit tests for validation
├── images/                   # Generated visualizations
├── requirements.txt          # Project dependencies
├── .gitignore               # Git ignore file
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-username]/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Download `creditcard_2023.csv` and place it in the project root directory
   - The script uses a sample of 10,000 rows for faster processing

## How to Run
1. **Run the main analysis:**
   ```bash
   cd src
   python fraud_detection.py
   ```

2. **Run tests:**
   ```bash
   cd tests
   python test_fraud_detection.py
   ```

## Methodology

### Data Preprocessing
- **Sampling:** Uses first 10,000 rows for efficient processing during development
- **Feature Scaling:** StandardScaler applied to normalize all features
- **Train-Test Split:** 80-20 split with stratification

### Model Selection
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:** 100 estimators, max depth 10, min samples split 5
- **Validation:** 3-fold cross-validation with F1 scoring

### Evaluation Metrics
- **Classification Report:** Precision, Recall, F1-score for both classes
- **Confusion Matrix:** Visual representation of classification performance
- **ROC-AUC:** Area under the receiver operating characteristic curve
- **Feature Importance:** Ranking of most influential features

## Results
The script outputs comprehensive model performance metrics and several visualizations to help understand the data and the effectiveness of the fraud detection model:

### Feature Importances
![Feature Importances](images/feature_importances.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Feature Correlation Heatmap
![Feature Correlation Heatmap](images/correlation_heatmap.png)

### ROC Curve
![ROC Curve](images/roc_curve.png)

## Key Findings
- **High Performance:** The Random Forest model achieves excellent performance on fraud detection
- **Feature Insights:** V12, V14, and V17 are among the most important features for fraud detection
- **Class Imbalance:** The dataset is highly imbalanced, typical of fraud detection scenarios
- **ROC-AUC:** Strong discriminative ability between fraudulent and legitimate transactions

## Future Improvements
- Implement advanced sampling techniques (SMOTE, ADASYN) for class imbalance
- Experiment with other algorithms (XGBoost, Neural Networks)
- Add real-time prediction capabilities
- Implement ensemble methods combining multiple models
- Add feature engineering and selection techniques

## Testing
The project includes comprehensive unit tests to validate:
- Data loading and preprocessing functions
- Model training and prediction capabilities
- Data validation and error handling

Run tests with: `python tests/test_fraud_detection.py`

## Project Highlights for Resume
- Built an end-to-end machine learning pipeline for fraud detection on a real-world, highly imbalanced dataset.
- Applied ensemble learning (Random Forest) and advanced evaluation metrics (ROC AUC, confusion matrix).
- Demonstrated strong data analysis, feature engineering, and model evaluation skills.
- Visualized data distributions and model results for clear communication of findings.

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- **Author:** [Your Name]
- **Email:** [your.email@example.com]
- **LinkedIn:** [Your LinkedIn Profile]
- **GitHub:** [Your GitHub Profile]

