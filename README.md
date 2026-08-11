# Machine Learning Assignment 2

## a. Problem Statement

The objective of this assignment is to develop and compare multiple machine learning classification models for predicting the grade class of students using the Synthetic Student Performance Dataset.

The models are trained using the available student-related features and evaluated using multiple classification evaluation metrics. The performance of the models is compared to identify the best-performing model for the selected dataset.

---

## b. Dataset Description

The dataset used for this assignment is the Synthetic Student Performance Dataset.

The dataset contains 5,000 student records and 15 columns. The dataset includes demographic, academic, and student-support-related information about students.

The columns in the dataset are:

- StudentID
- Age
- Gender
- Ethnicity
- ParentalEducation
- StudyTimeWeekly
- Absences
- Tutoring
- ParentalSupport
- Extracurricular
- Sports
- Music
- Volunteering
- GPA
- GradeClass

The target variable used for classification is `GradeClass`.

During exploratory data analysis, the dataset structure, missing values, duplicate records, unique values, target class distribution, and feature distributions were examined.

`StudentID` was excluded because it is an identifier and does not provide meaningful predictive information. `GPA` was also excluded from the model features to avoid using a variable that is directly related to student academic performance and could make the prediction of `GradeClass` less meaningful. `GradeClass` was used as the target variable.

The dataset was divided into training and testing sets, and feature scaling was performed as required for the implemented models.

---

## c. GitHub Repository Link

[GitHub Repository Link](https://github.com/sheenamuhammed/ML_Assignment_2)

The repository contains the required project files, including:

- app.py
- requirements.txt
- test_data.csv
- model folder containing the saved trained models and scaler
- README.md

---

## d. Models Used

The following machine learning classification models were implemented:

- Logistic Regression
- Decision Tree
- kNN
- Naive Bayes
- Random Forest (Ensemble)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4720 | 0.6724 | 0.4009 | 0.4720 | 0.3928 | 0.2013 |
| Decision Tree | 0.3430 | 0.5398 | 0.3523 | 0.3430 | 0.3473 | 0.0693 |
| kNN | 0.4220 | 0.6056 | 0.3643 | 0.4220 | 0.3775 | 0.1154 |
| Naive Bayes | 0.4780 | 0.6714 | 0.3949 | 0.4780 | 0.3908 | 0.2111 |
| Random Forest (Ensemble) | 0.4450 | 0.6619 | 0.3836 | 0.4450 | 0.3947 | 0.1578 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved an accuracy of 0.4720 and an AUC of 0.6724. It achieved the highest AUC among the evaluated models and provided competitive overall performance. |
| Decision Tree | Decision Tree achieved an accuracy of 0.3430 and an AUC of 0.5398, which were the lowest among the evaluated models. Its MCC of 0.0693 was also the lowest, indicating the weakest overall performance. |
| kNN | kNN achieved an accuracy of 0.4220 and an AUC of 0.6056. Its performance was better than the Decision Tree but lower than Logistic Regression, Naive Bayes, and Random Forest on most evaluation metrics. |
| Naive Bayes | Naive Bayes achieved the highest accuracy of 0.4780, the highest recall of 0.4780, and the highest MCC of 0.2111. Its AUC of 0.6714 was also very close to that of Logistic Regression. |
| Random Forest (Ensemble) | Random Forest achieved an accuracy of 0.4450 and an AUC of 0.6619. It achieved the highest F1 score of 0.3947 among the evaluated models and provided competitive overall performance. |

### Overall Winner for your dataset?

**Naive Bayes** was selected as the overall winner for the dataset.

It achieved the highest Accuracy (0.4780), highest Recall (0.4780), and highest MCC (0.2111). Although Logistic Regression achieved a slightly higher AUC (0.6724) and Random Forest achieved the highest F1 score (0.3947), Naive Bayes provided the strongest overall performance based on the evaluation results.