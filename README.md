# Heart Disease Detection

This project aims to predict the presence of heart disease in patients using machine learning techniques (Logistic Regression). The dataset used for this project is a collection of medical data related to heart disease.

## Logistic Regression

Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

### Definition

Logistic Regression is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead, or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc.

### Equation

The logistic regression model is represented by the following equation:

![Equation](<https://latex.codecogs.com/png.latex?P(Y=1|X)=\frac{1}{1+e^{-({\beta_0}+{\beta_1}X_1+{\beta_2}X_2+...+{\beta_n}X_n)}}>)

Where:

- `P(Y=1|X)` is the probability of the dependent variable being 1 given the independent variables `X`.
- `` `β_0` `` is the intercept term.
- `` `β_1, β_2, ..., β_n` `` are the coefficients of the independent variables `X_1, X_2, ..., X_n`.

### Use Cases

Logistic Regression is widely used in various fields, including:

1. **Medical Field**: Predicting the presence or absence of a disease (e.g., heart disease, diabetes).
2. **Marketing**: Predicting whether a customer will buy a product or not.
3. **Finance**: Predicting the likelihood of a loan default.
4. **Social Sciences**: Predicting the likelihood of an event occurring (e.g., voting behavior).

### Calculations

#### 1. Odds and Log-Odds

- **Odds**: The odds of an event occurring is the ratio of the probability of the event occurring to the probability of the event not occurring.

  ![Equation](https://latex.codecogs.com/png.latex?\text{Odds}=\frac{P}{1-P})

- **Log-Odds**: The natural logarithm of the odds.

  ![Equation](<https://latex.codecogs.com/png.latex?\text{Log-Odds}=\log(\frac{P}{1-P})>)

#### 2. Maximum Likelihood Estimation (MLE)

The coefficients `` `β_0, β_1, ..., β_n` `` are estimated using Maximum Likelihood Estimation (MLE), which finds the values that maximize the likelihood function.

#### 3. Decision Boundary

The decision boundary is the threshold at which the predicted probability is converted into a class label. Typically, a threshold of 0.5 is used.

#### 4. Model Evaluation

- **Confusion Matrix**: A table used to describe the performance of a classification model.
- **Accuracy**: The ratio of correctly predicted instances to the total instances.

  ![Equation](https://latex.codecogs.com/png.latex?\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN})

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.

  ![Equation](https://latex.codecogs.com/png.latex?\text{Precision}=\frac{TP}{TP+FP})

- **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all observations in the actual class.

  ![Equation](https://latex.codecogs.com/png.latex?\text{Recall}=\frac{TP}{TP+FN})

- **F1 Score**: The weighted average of Precision and Recall.

  ![Equation](https://latex.codecogs.com/png.latex?\text{F1%20Score}=2\times\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}})

## Conclusion

Logistic Regression is a powerful and widely used statistical method for binary classification problems. It is easy to implement and interpret, making it a popular choice for many applications.

## References

- [Logistic Regression - Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Introduction to Logistic Regression](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)

## Project Structure

```
heart-disease-prediction/
|
|-- heart_disease_data.csv
|-- code.ipynb
|-- README.md
```

- **`heart_disease_data.csv`**: The dataset containing medical information of patients.
- **`code.ipynb`**: The Jupyter notebook containing the code for data processing, model training, and evaluation.
- **`README.md`**: This file, providing an overview of the project.

## Dataset

The dataset contains the following columns:

- `age`: Age of the patient
- `sex`: Sex of the patient (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholesterol in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment (0-2)
- `ca`: Number of major vessels (0-3) colored by fluoroscopy
- `thal`: Thalassemia (0-3)
- `target`: Diagnosis of heart disease (1 = presence, 0 = absence)

## Steps

1. **Importing Dependencies**: Import necessary libraries such as `numpy`, `pandas`, `sklearn`.
2. **Data Collection & Processing**: Load the dataset and perform initial data exploration.
3. **Data Splitting**: Split the data into training and testing sets.
4. **Model Training**: Train a Logistic Regression model using the training data.
5. **Model Evaluation**: Evaluate the model's performance using accuracy scores.
6. **Predictive System**: Build a system to predict heart disease based on input features.

## Usage

1. Clone the repository.
2. Open `code.ipynb` in Jupyter Notebook or any compatible environment.
3. Run the cells sequentially to execute the code.

## Results

The Logistic Regression model achieved the following accuracy:

- **Training Data Accuracy**: 85.12%
- **Test Data Accuracy**: 81.97%

## Conclusion

This project demonstrates the use of Logistic Regression for predicting heart disease. The model can be further improved by tuning hyperparameters, using different algorithms, or incorporating more data.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)

Feel free to contribute to this project by opening issues or submitting pull requests.
