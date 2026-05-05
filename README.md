# 2-stage Hierarchical Framework for Shipment Delay and Severity Estimation
### Training
*python -m shipment_prediction --train __[xgb | mlp | lstm | all]__ --type __[binary | regression | regression2 | both]__*
#### Note: 
* *__all__* - trains 3 models for the given type
* *__both__* - trains both binary and regression models *(this does not include regression2)*
* *__regression2__* - trains a regression model with the additional *__disruption_occurred__* feature
* *__after training is successfully completed, models are saved within the ./models directory__*
### Testing
#### Binary Models
*python -m shipment_prediction --binary_model_path ./models/__NameOfBinaryModel.joblib__*
#### Regression Models
*python -m shipment_prediction --regression_model_path ./models/__NameOfRegressionModel.joblib__*
#### 2-stage Hierarchical Framework
*python -m shipment_prediction --binary_model_path ./models/__NameOfBinaryModel.joblib__ --regression_model_path ./models/__NameOfRegressionModel.joblib__*
#### Note:
* *__--regression_model_path__* - also works for *regression2* trained models *(e.i., _regression_model_stage2_)*
* *__2-stage Hierarchical Framework__* - only works with one binary model and one regression model *(e.i., regression or regression2)*