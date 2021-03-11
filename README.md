# House Rent Predictor

A Machine Learning model to predict the rent price of the house based on the parameters like `area, no of bedrooms,society, location etc.` I have done some feature engineering and transformations to generalise the regression model and make accurate predictions.

Dataset is from Mumbai, extracted from the website 99aces.com and uploaded in the .csv file.

# Aim: To Predict Monthly House Rent. 

## Approch: 
- Import Dataset and have a Quick look at it.
- Analyze and Clean Data. Look into each and every feature and make them clean for use.
- Remove Outliers and exception data by visualizing features.
- Visualize and have a look how features related with monthly rent.
- Prepare hand engineered features and Scale Data for Machine learning model.
- Try Different ML models and calculate their accuracy scores.
- Fine tune the model using GridsearchCV and Select best model.

EDA and Data cleaning takes much of time and the results are worth noted. Among all feature engineeing, the most important of all is target variable transformation.

Monthly House Rent feature is `highly skewed` as shown by distribution and CDF charts.
<img src="https://github.com/G0rav/House_Rent_Predictor/blob/main/src/without%20log%20transformation.png">

I've used `log tranformation` to make it normally distributed.
<img src="https://github.com/G0rav/House_Rent_Predictor/blob/main/src/with%20log%20transformation.png">

# Result
Because of this transformation, there is magnificent reduction root mean score error (rmse):

```
- without log transformation rmse: 12192.737654367309
- with log transformation rmse: 0.19852773815105582
```

Tried almost every Regression models to predict, optimized them using GRidsearchCv and RandomizedsearchCv to find the best Hyper parameter.
Linear Regression and XGBoost Regression comes out to be the best models with rmse of **.19**

Comparison Table:

|Model |LinearRegression	|Ridge	|Lasso	|DecisionTreeRegressorOHE	|RandomForestRegressorOHE	|XGBRegressorOHE	|GradientBoostingRegressorOHE	|DecisionTreeRegressorLE	|RandomForestRegressorLE	|XGBRegressorLE	|GradientBoostingRegressorLE |
|----|----|----|----|----|----|----|----|----|----|----|----|
|With Training Data r2 score	|0.9210	|0.9209	|0.0000	|0.0000	|0.0000	|0.0000	|0.0000	|0.9988	|0.9779	|0.8047	|0.8064|
|With Test Data r2 score	|0.9276|	0.9276	|-0.0001	|-0.0001	|-0.0001	|-0.0001	|-0.0001	|0.7221	|0.8536	|0.7959	|0.7986|
|With Training Data rmse	|0.2071	|0.2071	|0.7365	|0.7365	|0.7365	|0.7365	|0.7365	|0.0251	|0.1095	|0.3255	|0.3241|
|With Test Data rmse	|0.1985	|0.1985	|0.7377	|0.7377	|0.7377	|0.7377	|0.7377	|0.3888	|0.2822	|0.3332	|0.3310|
