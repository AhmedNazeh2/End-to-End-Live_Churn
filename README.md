
# Customer Churn Prediction Project

This project implements a machine learning pipeline for predicting customer churn. It uses various features such as demographic information, account data, and past behavior to predict whether a customer will exit (churn) from a service. The notebook includes steps for data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Building](#model-building)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [Output](#output)

## Project Overview
This project aims to predict customer churn using machine learning techniques. It involves loading and exploring the dataset, handling missing data, visualizing key features, and building classification models to predict churn. The models used in this project include Logistic Regression, Random Forest Classifier, and XGBoost.

## Data Preprocessing
- **Loading Data**: The dataset is read from a CSV file.
- **Missing Values**: We identify and handle missing data.
- **Feature Engineering**: Includes encoding categorical variables and scaling numerical features.
- **Outliers**: Outliers in certain features (e.g., age) are handled by removing extreme values.

## Model Building
Several models are built using the preprocessed data:
1. **Logistic Regression** - A baseline model for comparison.
2. **Random Forest Classifier** - A robust model for classification.
3. **XGBoost** - An optimized gradient boosting model.

Each model is evaluated using the F1-score metric to assess its performance on imbalanced data.

## Evaluation Metrics
- **F1-score**: We use the F1-score to evaluate model performance, especially due to class imbalance in the target variable (`Exited`).

## Hyperparameter Tuning
We apply **GridSearchCV** to tune hyperparameters of the Random Forest model for better performance.

## Usage
To run this project:
1. Install required dependencies (see below).
2. Place the `dataset.csv` file in the same directory.
3. Execute the notebook step by step to load, preprocess, and train the models.
4. View the results and visualizations at each stage.

## Dependencies
This project requires the following Python libraries:
- scikit-learn==1.5.1
imbalanced-learn==0.12.3
fastapi==0.112.1
uvicorn==0.30.6
matplotlib==3.9.2
seaborn==0.13.2
python-multipart==0.0.9
xgboost==2.1.1

You can install the dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
```

## Output
After running the notebook, the following outputs will be available:
- **Model Performance**: Evaluation results for each model with metrics such as F1-score.
- **Trained Models**: The best-performing models will be saved using `joblib` for later use. The models are saved in the `artifacts/` folder.
- **Visualizations**: Data visualizations including bar charts, histograms, and count plots to explore key features.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Happy coding!
