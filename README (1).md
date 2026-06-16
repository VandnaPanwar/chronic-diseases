**Overview**

This project builds a machine learning pipeline for early detection and prediction of chronic disease risk using Electronic Health Record (EHR)-style data. The goal is to develop predictive models that flag elevated chronic-disease risk from demographic, lifestyle, and lab-test features, supporting earlier clinical attention.

**Dataset**

The project uses a synthetic chronic diseases dataset (`synthetic_chronic_diseases.csv`) generated to resemble the structure of real-world EHR data. It includes demographics (age, gender), lifestyle factors (smoking status, physical activity level), and lab/clinical measurements (BMI, cholesterol, blood pressure, blood sugar, hypertension status), with a binary target label (`Chronic_Disease`). The data is synthetic and generated from a probabilistic risk model with injected noise, so it approximates plausible relationships but should not be treated as clinically validated.

**Data Preprocessing**

Missing numerical values are imputed with the median; missing categorical values with the mode. Duplicate rows are removed. Outliers in numeric columns are clipped using the IQR rule. Binary categorical variables (gender, smoking, hypertension) are label-encoded, and the three-level `Physical_Activity` variable is one-hot encoded. Numerical features are standardized using Z-score scaling.

**Feature Selection**

Three complementary feature selection approaches are implemented and compared:

- **Chi-Square test** — measures the statistical association between each feature and the target label.
- **Recursive Feature Elimination (RFE)** — a wrapper method that iteratively removes the weakest feature(s) according to a Logistic Regression estimator.
- **Particle Swarm Optimization (PSO)** — a metaheuristic search implemented from scratch, where particles represent candidate binary feature subsets and are guided toward higher 5-fold cross-validated accuracy (with a mild penalty for using more features) via inertia, cognitive, and social velocity terms.

The final feature set used for modeling is the subset selected by PSO, since it directly optimizes downstream classifier performance rather than a proxy statistic.

**Model Development**

Logistic Regression, Decision Tree, Random Forest, and Naive Bayes serve as baselines. A Convolutional Neural Network (CNN) — typically used for image data — is adapted for tabular input by treating each patient's selected features as a 1D sequence and applying `Conv1D` layers to learn local feature interactions before pooling into a dense classification head.

Data is split 70/30 into training and test sets (stratified by class), and all models are additionally evaluated with 5-fold cross-validation on the training set.

**Model Evaluation**

Models are evaluated using accuracy, precision, recall, F1-score, and AUC-ROC.

On this synthetic dataset, baseline models (Logistic Regression, Naive Bayes) and the CNN perform comparably, in the roughly 65–71% test accuracy range, with AUC-ROC between approximately 0.71 and 0.79. The CNN does **not** outperform the simpler baselines here. This is an expected and informative result: the engineered tabular features lack the strong local spatial structure that convolution is designed to exploit, so a model with more parameters doesn't have an inherent advantage on this dataset. The Decision Tree is the weakest model, consistent with its tendency to overfit without ensembling.

**Results and Discussion**

The results show that on this dataset, simpler models are at least as effective as a CNN, and in some cases marginally better. This highlights an important and often-overlooked lesson in applied ML: model complexity should match the structure of the data, and more sophisticated architectures aren't automatically better. The PSO-based feature selection step still adds value by identifying a compact, high-signal feature subset (consistently surfacing hypertension, smoking, cholesterol, and physical activity level as strong predictors, consistent with the Chi-Square rankings).

**Future Work**

Future iterations could incorporate richer, less hand-engineered inputs (e.g. longitudinal/time-series vitals, where a CNN's local-pattern detection would have a more natural advantage), additional risk factors such as family history and genetic markers, and proper handling of class imbalance and calibration for clinical deployment. Any real-world application would also require external validation on real (de-identified) EHR data, fairness audits across demographic subgroups, and clinical oversight before deployment.

**Conclusion**

This project implements a complete pipeline — preprocessing, three feature-selection strategies, classical ML baselines, and a CNN adapted for tabular data — evaluated with a stratified train/test split and 5-fold cross-validation. The honest result, that the CNN does not clearly beat simpler baselines on this dataset, is itself a useful and defensible finding about when deep learning is and isn't the right tool.

**Installation and Requirements**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nbformat
```

**How to Run**

Open `chronic_diseases_extended.ipynb` and run the notebook top to bottom. It contains all code for data preprocessing, feature selection (Chi-Square, RFE, PSO), model training, evaluation, and results visualization.

Make sure `synthetic_chronic_diseases.csv` is in the same directory as the notebook, or update the file path in the data-loading cell.
