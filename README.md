### AUTOMATED MACHINE LEARNING
Python Libraies Used:
Pandas
Numpy
Sklearn
Statsmodel

This project provides a service of Classification,Regression or Forecasting depending on the user's choice. 
When user provides a data set and type of operation(Classification,Regression or Forecasting), the program reads the dataset using Pandas. The User uploads the dataset and metadata through the 'user_input.py' file in the API folder and then starts the training procedure.

The API was created using flask and tested using Postman

To train the machine learning model, we select the hyperparameters using the Grid Search Technique. The best hyperparameter is selected and the model is trained and saved in the 'Models' Folder. Depending on the technique used the models are saved in the respective technique model (Classification,Regression or Forecasting)

To provide the service to the User, we use the 'technique'_services.py file. On providing an input through postman in JSON format, the output is formed from the saved model and outputted in JSON format.

Machine Learning Models Used :
Classification : Support Vector Machine(svm.SVC - support vector classifier), Random Forest Classifier
Regression : Lasso Regression, Ridge Regression
Forecasting : SARIMA - Seasonal ARIMA(For Seasonal and Univariate Data), VAR - Vector Autoregression, VARMAX - Vector Auto Regression Moving Avgrage(For Multivariate Data, can be used for both seasonal and non seasonal data)

DataSet for Classification - Non Time Series Dataset
DataSet for Regression - Time Series and Non Time Series Dataset can be used. (The time Column is removed and 
the the regression algorithm is performed).
Dataset for Forecasting - Time Series data set. 
** For Forecasting , The user must mention the time column

