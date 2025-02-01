# Heart Disease Detection

This project aims to predict the presence of heart disease in patients using machine learning techniques (Logistic Regression). The dataset used for this project is a collection of medical data related to heart disease.

## Logistic Regression

Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

### Definition

Logistic Regression is used to model the probability of a certain class or event existing, such as pass/fail, win/lose, alive/dead, or healthy/sick. This can be extended to model several classes of events, such as determining whether an image contains a cat, dog, lion, etc.

### Equation

The logistic regression model is represented by the following equation:

![Logistic Regression Equation](<https://latex.codecogs.com/png.latex?P(Y%3D1%7CX)%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-%28%5Cbeta_0%20%2B%20%5Cbeta_1X_1%20%2B%20%5Cbeta_2X_2%20%2B%20...%20%2B%20%5Cbeta_nX_n%29%7D%7D>)

Where:

- `P(Y=1|X)` is the probability of the dependent variable being 1 given the independent variables `X`.
- `` `β_0` `` is the intercept term.
- `` `β_1, β_2, ..., β_n` `` are the coefficients of the independent variables `` `X_1, X_2, ..., X_n` ``.

### Use Cases

Logistic Regression is widely used in various fields, including:

1. **Medical Field**: Predicting the presence or absence of a disease (e.g., heart disease, diabetes).
2. **Marketing**: Predicting whether a customer will buy a product or not.
3. **Finance**: Predicting the likelihood of a loan default.
4. **Social Sciences**: Predicting the likelihood of an event occurring (e.g., voting behavior).

### Calculations

#### 1. Odds and Log-Odds

- **Odds**: The odds of an event occurring is the ratio of the probability of the event occurring to the probability of the event not occurring.

  ![Odds Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BOdds%7D%20%3D%20%5Cfrac%7BP%7D%7B1-P%7D)

- **Log-Odds**: The natural logarithm of the odds.

  ![Log-Odds Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BLog-Odds%7D%20%3D%20%5Clog%28%5Cfrac%7BP%7D%7B1-P%7D%29)

#### 2. Maximum Likelihood Estimation (MLE)

The coefficients `` `β_0, β_1, ..., β_n` `` are estimated using Maximum Likelihood Estimation (MLE), which finds the values that maximize the likelihood function.

#### 3. Decision Boundary

The decision boundary is the threshold at which the predicted probability is converted into a class label. Typically, a threshold of 0.5 is used.

#### 4. Model Evaluation

- **Confusion Matrix**: A table used to describe the performance of a classification model.
- **Accuracy**: The ratio of correctly predicted instances to the total instances.

  ![Accuracy Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BAccuracy%7D%20%3D%20%5Cfrac%7BTP%20%2B%20TN%7D%7BTP%20%2B%20TN%20%2B%20FP%20%2B%20FN%7D)

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.

  ![Precision Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BPrecision%7D%20%3D%20%5Cfrac%7BTP%7D%7BTP%20%2B%20FP%7D)

- **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all observations in the actual class.

  ![Recall Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BRecall%7D%20%3D%20%5Cfrac%7BTP%7D%7BTP%20%2B%20FN%7D)

- **F1 Score**: The weighted average of Precision and Recall.

  ![F1 Score Formula](https://latex.codecogs.com/png.latex?F1%20%3D%202%20%5Ctimes%20%5Cfrac%7B%5Ctext%7BPrecision%7D%20%5Ctimes%20%5Ctext%7BRecall%7D%7D%7B%5Ctext%7BPrecision%7D%20%2B%20%5Ctext%7BRecall%7D%7D)

## Project Structure

```
heart-disease-prediction/
|
|-- heart_disease_data.csv
|-- code.ipynb
|-- README.md
```

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

## Results

The Logistic Regression model achieved the following accuracy:

- Training Data Accuracy: **85.12%**
- Test Data Accuracy: **81.97%**

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
