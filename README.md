# Data Science Projects

Welcome to my data science projects repository! This repository contains a collection of my data science projects, showcasing my skills and expertise in the field. Each project demonstrates different aspects of data analysis, machine learning, and visualization.

## Projects

1. [Titanic Survival Prediction](/1_titanic)

- **Description** : This project predicts the survival during the titanic disaster based on socio-economic passengers data.
- **Tools used** : data cleaning, feature engineering, one-hot encoding, feature selection and classifier fitting. 
- **Results** : The best classifier is Random Forest, with a train accuracy of 0.98 and an F1-score of 0.98. The kaggle submission scores 0.77 on the test set.




## General Steps for Building a Machine Learning Model


1. Data Collection
- When participating in a Kaggle competition, this step is already completed for you.

2. Understand the Dataset
- What do we need to predict (what is the target variable) ?
- Is it a classification or a regression problem ?
- Explore the data to gain insights into the features (columns) and their meanings.

3. Data Preprocessing / Data Cleaning
- Remove duplicates rows if any.
- Handle missing data (zero, null, blank or NaNs): either impute them or drop rows/columns with excessive missing data. 

4. Exploratory Data Analysis (EDA)
- Visualize the data with plots and graphs
- Univariate analysis: histograms, measures of central tendency (mean, median, mode), and measures of dispersion (range, standard deviation). 
- Bivariate analysis: understand the relationships between features and the target variable using scatterplots, correlation coefficients / matrix.
- Identify outliers: detect and handle outliers that might skew your model's predictions.
- Decide what features needs to be normalized, selected and/or engineering and errors and bias need to be removed.

5. Feature Engineering / Feature Selection
- Create new features: engineer relevant features that might improve predictive performance.
- Remove irrelevant features: eliminate features that do not contribute much to the prediction. Many ML algorithms, such as Random Forest, provide a feature importance score.
- Feature normalization : improves distance-based calculations (for KNN, SVM classifiers) and prevents feature dominance.
- Handle skwed features : many algorithm assume normally distributed data. Fix skewness by applying log transform.
- Remove colinear features : improves the model stability and interpretability, and reduces overfitting.
- Encode categorical features: convert categorical features into numerical form using one-hot encoding or label encoding.

**NOTE :** steps 3, 4, and 5 are usually done simultaneously.
      
6. Choose Evaluation Metrics
- Decide on the evaluation metric you'll use to measure the performance of your model. 
- For regression problems, common metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.
- For classification problems, common metrics include: accuracy, Precision, Recall, F1-score and AUC. The confusion matric and ROC Curve can also bring useful insights. 

7. Select Algorithms
- Start with simple regression algorithms like Linear Regression and gradually explore more complex models like Random Forest, Gradient Boosting, or XGBoost.
- Consider creating ensemble models (eg. simple average, weighted average, or voting ensembles) by combining the predictions of multiple models (e.g., stacking or blending) for potentially improved accuracy.
- The model chosen depends on the data. A more complex model does not always constitute a better model.

8. Model Validation
- Split the data into training and validation sets. A common split is 70-30 or 80-20 for training and validation, respectively. This method is computationally less intensive and often used for initial model exploration or when dealing with very large datasets.
- K-Fold Cross Validation. This method provides a more reliable evaluation, especially with smaller datasets.
- Model validation is important to assess the model's generalization performance (i.e. assess how well the model performs on unseen data). This helps prevent overfitting and gives you a more reliable estimate of your model's performance.

9. Hyperparameter Tuning
- Tune the hyperparameters of your chosen algorithms on the validation dataset using techniques like grid search or random search to find the best combination.
- Optuna is an efficient and effective way to search for optimal hyperparameters.

10. Regularization
- Implement regularization techniques like L1 (Lasso) or L2 (Ridge) regularization to prevent overfitting.
- Many ML algorithms include regularization parameters, including L1 and L2, sometimes called reg_alpha or reg_lambda. Read up on your chosen algorithms regularization parameters and tune them accordingly on your validation set.

11. Train the final model
- Fit the best model using the optimal hyperparameters found on the whole training set (including the validation set).
- Model persistence : save the model weights for future use.

12. Generate predictions on the test set

13. Interpretability
- Try to make your model interpretable by using techniques like SHAP values or partial dependence plots to understand how different features affect predictions.
- The Kaggle course Machine Learning Explainability covers both partial dependence plots and SHAP values in lessons 3 and 4 respectively.

14. Documentation and Reporting
- Keep detailed records of your work, including data preprocessing steps, model selection, hyperparameter tuning, and results. This will help you remember your thought process when returning to your project at a later time.
- Create a report or notebook to present your findings and methodology.



