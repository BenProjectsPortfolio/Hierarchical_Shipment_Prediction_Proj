import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd

# when training and testing models with the additional feature for regression stage 2,
# remember to uncomment the additional feature in the regression_numerical_features list
# and uncomment the plot title in the plot_feature_importances functions and regression_prediction in train.py


VALID_MODEL_TYPES = ['xgb', 'mlp', 'lstm']
VALID_TASK_TYPES = ['binary', 'regression']
#LSTM_INPUT_SHAPE = (5, 63)  # (timesteps, features) -- Full feature set, 63 features in total.

# Binary dataset setting
BINARY_LSTM_INPUT_SHAPE = (5, 18)  # (timesteps, features)
BINARY_EMBEDDING_VOCAB_SIZES = { # dataset[column].nunique() for categorical columns
    'Origin_Port': 8, 
    #'Destination_Port': 9, 
    #'Transport_Mode': 4, 
    #'Product_Category': 5, 
    'Weather_Condition': 5, 
    #'Route': 64, 
    #'Route_Mode': 256
}
binary_numerical_features = [ # For LSTM Feature Importance plotting - Note: they are in order as numerical features are not labeled once embedded
    #'Origin_Port',         # in categorical
    #'Destination_Port',    # in categorical
    #'Transport_Mode',      # in categorical
    #'Product_Category',    # in categorical
    #'Distance_km', 
    #'Weight_MT', 
    'Fuel_Price_Index',
    'Geopolitical_Risk_Score', 
    #'Weather_Condition',   # in categorical
    'Carrier_Reliability_Score', 
    #'Route',               # in categorical
    #'Route_Mode',          # in categorical
    #'Year', 
    #'Month', 
    #'Quarter', 
    #'Day', 
    'Day_of_Week',
    'Week_of_Year', 
    'Is_Month_Start', 
    'Is_Month_End', 
    'Month_Sin',
    'Month_Cos', 
    'Day_of_Week_Sin', 
    'Day_of_Week_Cos', 
    #'Distance_km_Lag1',
    #'Distance_km_Lag2', 
    #'Distance_km_Lag3', 
    #'Weight_MT_Lag1',
    #'Weight_MT_Lag2', 
    #'Weight_MT_Lag3', 
    'Fuel_Price_Index_Lag1',
    #'Fuel_Price_Index_Lag2', 
    #'Fuel_Price_Index_Lag3',
    #'Geopolitical_Risk_Score_Lag1', 
    #'Geopolitical_Risk_Score_Lag2',
    #'Geopolitical_Risk_Score_Lag3', 
    'Carrier_Reliability_Score_Lag1',
    #'Carrier_Reliability_Score_Lag2', 
    #'Carrier_Reliability_Score_Lag3',
    #'Lead_Time_Days_Lag1', 
    #'Lead_Time_Days_Lag2', 
    #'Lead_Time_Days_Lag3',
    'Disruption_Occurred_Lag1', 
    #'Disruption_Occurred_Lag2',
    #'Disruption_Occurred_Lag3', 
    #'Lead_Time_Days_Rolling3_Mean',
    #'Lead_Time_Days_Rolling5_Mean', 
    #'Lead_Time_Days_Rolling3_Std',
    #'Disruption_Occurred_Rolling3_Mean',
    #'Disruption_Occurred_Rolling5_Mean', 
    #'Disruption_Occurred_Rolling3_Std',
    #'Fuel_Price_Index_Rolling3_Mean', 
    #'Fuel_Price_Index_Rolling5_Mean',
    #'Fuel_Price_Index_Rolling3_Std',
    #'Geopolitical_Risk_Score_Rolling3_Mean',
    #'Geopolitical_Risk_Score_Rolling5_Mean',
    #'Geopolitical_Risk_Score_Rolling3_Std',
    #'Carrier_Reliability_Score_Rolling3_Mean',
    #'Carrier_Reliability_Score_Rolling5_Mean',
    #'Carrier_Reliability_Score_Rolling3_Std', 
    'Historical_Disruption_Rate',
    #'Historical_Avg_Lead_Time', 
    'Days_Since_Previous_Shipment',
]

# Regression dataset setting
REGRESSION_LSTM_INPUT_SHAPE = (5, 19) 
REGRESSION_EMBEDDING_VOCAB_SIZES = { # dataset[column].nunique() for categorical columns
    #'Origin_Port': 8, 
    'Destination_Port': 9, 
    'Transport_Mode': 4, 
    'Product_Category': 5, 
    'Weather_Condition': 5, 
    #'Route': 64, 
    #'Route_Mode': 256
}
regression_numerical_features = [ # For LSTM Feature Importance plotting - Note: they are in order as numerical features are not labeled once embedded
    #'Origin_Port',         # in categorical
    #'Destination_Port',    # in categorical
    #'Transport_Mode',      # in categorical
    #'Product_Category',    # in categorical
    'Distance_km', 
    #'Weight_MT', 
    'Fuel_Price_Index',
    'Geopolitical_Risk_Score', 
    #'Weather_Condition',   # in categorical
    'Carrier_Reliability_Score', 
    #'Route',               # in categorical
    #'Route_Mode',          # in categorical
    #'Year', 
    #'Month', 
    #'Quarter', 
    #'Day', 
    'Day_of_Week',
    'Week_of_Year', 
    'Is_Month_Start', 
    'Is_Month_End', 
    #'Month_Sin',
    #'Month_Cos', 
    'Day_of_Week_Sin', 
    'Day_of_Week_Cos', 
    'Distance_km_Lag1',
    #'Distance_km_Lag2', 
    #'Distance_km_Lag3', 
    #'Weight_MT_Lag1',
    #'Weight_MT_Lag2', 
    #'Weight_MT_Lag3', 
    'Fuel_Price_Index_Lag1',
    #'Fuel_Price_Index_Lag2', 
    #'Fuel_Price_Index_Lag3',
    #'Geopolitical_Risk_Score_Lag1', 
    #'Geopolitical_Risk_Score_Lag2',
    #'Geopolitical_Risk_Score_Lag3', 
    'Carrier_Reliability_Score_Lag1',
    #'Carrier_Reliability_Score_Lag2', 
    #'Carrier_Reliability_Score_Lag3',
    #'Lead_Time_Days_Lag1', 
    #'Lead_Time_Days_Lag2', 
    #'Lead_Time_Days_Lag3',
    'Disruption_Occurred_Lag1', 
    #'Disruption_Occurred_Lag2',
    #'Disruption_Occurred_Lag3', 
    #'Lead_Time_Days_Rolling3_Mean',
    #'Lead_Time_Days_Rolling5_Mean', 
    #'Lead_Time_Days_Rolling3_Std',
    #'Disruption_Occurred_Rolling3_Mean',
    #'Disruption_Occurred_Rolling5_Mean', 
    #'Disruption_Occurred_Rolling3_Std',
    #'Fuel_Price_Index_Rolling3_Mean', 
    #'Fuel_Price_Index_Rolling5_Mean',
    #'Fuel_Price_Index_Rolling3_Std',
    #'Geopolitical_Risk_Score_Rolling3_Mean',
    #'Geopolitical_Risk_Score_Rolling5_Mean',
    #'Geopolitical_Risk_Score_Rolling3_Std',
    #'Carrier_Reliability_Score_Rolling3_Mean',
    #'Carrier_Reliability_Score_Rolling5_Mean',
    #'Carrier_Reliability_Score_Rolling3_Std', 
    'Historical_Disruption_Rate',
    #'Historical_Avg_Lead_Time', 
    #'Days_Since_Previous_Shipment',
    #'Disruption_Occurred',            # uncomment only for regression 2 stage
]



def plot_overall_data_analysis(dataset):
    plot_sin_cos_distributions(dataset)
    plot_target_distributions(dataset)
    plot_numerical_feature_distributions(dataset)
    plot_categorical_feature_distributions(dataset)



def plot_sin_cos_distributions(dataset):
    plot_Month_Cos_data(dataset)
    plot_Month_Sin_data(dataset)
    plot_Day_of_Week_Cos_data(dataset)
    plot_Day_of_Week_Sin_data(dataset)

def plot_Month_Cos_data(dataset):
    months = np.arange(1, 13)
    # columns 'Month' and 'Month_Cos'
    avg_cos = dataset.groupby('Month')['Month_Cos'].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(avg_cos.index, avg_cos.values, label='Average Month_Cos', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Month_Cos')
    plt.title('Average Month_Cos by Month')
    plt.xticks(months)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_Month_Sin_data(dataset):
    months = np.arange(1, 13)
    # columns 'Month' and 'Month_Sin'
    avg_sin = dataset.groupby('Month')['Month_Sin'].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(avg_sin.index, avg_sin.values, label='Average Month_Sin', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Month_Sin')
    plt.title('Average Month_Sin by Month')
    plt.xticks(months)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_Day_of_Week_Cos_data(dataset):
    days = np.arange(1, 8)
    # columns 'Day_of_Week' and 'Day_of_Week_Cos'
    avg_cos = dataset.groupby('Day_of_Week')['Day_of_Week_Cos'].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(avg_cos.index, avg_cos.values, label='Average Day_of_Week_Cos', marker='o')
    plt.xlabel('Day_of_Week')
    plt.ylabel('Day_of_Week_Cos')
    plt.title('Average Day_of_Week_Cos by Day_of_Week')
    plt.xticks(days)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_Day_of_Week_Sin_data(dataset):
    days = np.arange(1, 8)
    # columns 'Day_of_Week' and 'Day_of_Week_Sin'
    avg_sin = dataset.groupby('Day_of_Week')['Day_of_Week_Sin'].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(avg_sin.index, avg_sin.values, label='Average Day_of_Week_Sin', marker='o')
    plt.xlabel('Day_of_Week')
    plt.ylabel('Day_of_Week_Sin')
    plt.title('Average Day_of_Week_Sin by Day_of_Week')
    plt.xticks(days)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_target_distributions(dataset):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(dataset['Lead_Time_Days'], bins=30, color='blue', alpha=0.7)
    plt.xlabel('Lead Time (Days)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lead Time (Regression Target)')

    plt.subplot(1, 2, 2)
    disruption_counts = dataset['Disruption_Occurred'].value_counts()
    disruption_counts.plot(kind='bar', color='orange', alpha=0.7)
    plt.xlabel('Disruption Occurred')
    plt.ylabel('Count')
    plt.title('Distribution of Disruption Occurred (Binary Target)')

    plt.tight_layout()
    plt.show()

def plot_numerical_feature_distributions(dataset):
    numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
    num_plots = len(numerical_cols)
    cols = 4
    total_rows = np.ceil(num_plots / cols).astype(int)
    rows = int(total_rows / 4)

    plt.figure(figsize=(15, rows * 3))
    plt_counter = 0
    for _, col in enumerate(numerical_cols):
        plt.subplot(rows, cols, plt_counter + 1)
        plt.hist(dataset[col], bins=30, color='green', alpha=0.7)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {col}')
        plt_counter += 1
        if plt_counter >= rows * cols:
            plt_counter = 0
            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(15, rows * 3))
    plt.tight_layout()
    plt.show()

def plot_categorical_feature_distributions(dataset, task_type):
    if task_type == 'binary':
        EMBEDDING_SIZES = BINARY_EMBEDDING_VOCAB_SIZES
    elif task_type == 'regression':
        EMBEDDING_SIZES = REGRESSION_EMBEDDING_VOCAB_SIZES
    
    categorical_cols = EMBEDDING_SIZES.keys()
    num_plots = len(categorical_cols)
    cols = 4
    rows = np.ceil(num_plots / cols).astype(int)

    plt.figure(figsize=(15, rows * 3))
    plt_counter = 0
    for _, col in enumerate(categorical_cols):
        plt.subplot(rows, cols, plt_counter + 1)
        value_counts = dataset[col].value_counts()
        value_counts.plot(kind='bar', color='purple', alpha=0.7)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Distribution of {col}')
        plt_counter += 1
        if plt_counter >= 2:
            plt_counter = 0
            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(15, rows * 3))
    plt.tight_layout()
    plt.show()



def display_shape_and_first_entries(input_data, labels):
    print("\n \n########## Input data shape ##########")
    print(input_data.shape)
    print(input_data.head())
    print("\n \n########## Labels shape ##########")
    print(labels.shape)
    print(labels.head())

def display_split_input_shapes(input_train, input_val, input_test):
    print("\n \n########## Dataset splits ##########")
    print("Train:", input_train.shape, "Val:", input_val.shape, "Test:", input_test.shape)

def plot_xgb_feature_importances(opt, model_name):
    model = opt.best_estimator_.named_steps[model_name]
    xgb.plot_importance(model)
    plt.title("XGBoost Feature Importance")
    #plt.title("XGBoost with Extra Feature: Feature Importance")
    plt.show()

def plot_mlp_feature_importances(opt, input_val, label_val):
    result = permutation_importance(
        opt, input_val, label_val, n_repeats=10, random_state=42, n_jobs=-1
    )

    # Get feature names if input_val is a DataFrame
    feature_names = input_val.columns if hasattr(input_val, 'columns') else [f'feature_{i}' for i in range(input_val.shape[1])]
    
    # Sort importances
    sorted_idx = np.argsort(result.importances_mean)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), result.importances_mean[sorted_idx])
    plt.xticks(range(len(feature_names)), np.array(feature_names)[sorted_idx], rotation=90)
    plt.title("MLP Feature Importance")
    #plt.title("MLP with Extra Feature: Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_lstm_feature_importances(model, input_val, label_val, task_type):
    rng = np.random.default_rng(42)

    label_val = label_val.reshape(-1,) # ensure shape is (samples,)
    baseline_preds = model.predict(input_val, verbose=0).reshape(-1)
    print("label_val shape:", label_val.shape)
    print("baseline_preds shape:", baseline_preds.shape)

    if task_type.lower() == 'binary':
        baseline_score = roc_auc_score(label_val, baseline_preds)
        numerical_feat = binary_numerical_features
    elif task_type.lower() == 'regression':
        baseline_score = -mean_squared_error(label_val, baseline_preds)
        numerical_feat = regression_numerical_features

    importances = {}

    # numeric input: shape (samples, timesteps, n_num_features)
    X_num = input_val["numerical_inputs"]
    n_num_features = X_num.shape[2]

    for j in range(n_num_features):
        X_perm = {k: v.copy() for k, v in input_val.items()}

        # shuffle this numeric feature across samples, preserve timestep pattern
        perm_idx = rng.permutation(X_num.shape[0])
        X_perm["numerical_inputs"][:, :, j] = X_num[perm_idx, :, j]

        preds = model.predict(X_perm, verbose=0).reshape(-1)
        print("label_val shape:", label_val.shape)
        print("preds shape:", preds.shape)
        if task_type.lower() == 'binary':
            perm_score = roc_auc_score(label_val, preds)
        elif task_type.lower() == 'regression':
            perm_score = -mean_squared_error(label_val, preds)
        
        feat_name = numerical_feat[j]
        importances[feat_name] = baseline_score - perm_score

    # categorical inputs: each is treated as one feature block
    for key in input_val:
        if key == "numerical_inputs":
            continue

        X_perm = {k: v.copy() for k, v in input_val.items()}
        perm_idx = rng.permutation(input_val[key].shape[0])
        X_perm[key] = input_val[key][perm_idx]

        preds = model.predict(X_perm, verbose=0).reshape(-1)
        print("label_val shape:", label_val.shape)
        print("preds shape:", preds.shape)
        if task_type.lower() == 'binary':
            perm_score = roc_auc_score(label_val, preds)
        elif task_type.lower() == 'regression':
            perm_score = -mean_squared_error(label_val, preds)
        
        importances[key] = baseline_score - perm_score
    
    # Convert to Series and sort
    imp = pd.Series(importances).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    imp.sort_values().plot(kind="barh")  # horizontal
    plt.title("LSTM Feature Importance")
    #plt.title("LSTM with Extra Feature: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()



def plot_feature_importances(model, model_name, model_type, input_val, label_val, task_type):
    if model_type.lower() == 'xgb':
        plot_xgb_feature_importances(model, model_name)
    elif model_type.lower() == 'mlp':
        plot_mlp_feature_importances(model, input_val, label_val)
    elif model_type.lower() == 'lstm':
        plot_lstm_feature_importances(model, input_val, label_val, task_type)

def get_model_name(model):
    model_name = list(model.best_estimator_.named_steps.keys())[-1]
    return model_name

def get_model_type(model_filename_path):
    filename = os.path.basename(model_filename_path)
    model_type = filename.split('_')[0]
    verify_model_type(model_type)
    return model_type.lower()

def verify_model_type(model_type):
    if model_type.lower() not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model type '{model_type}'. Valid types are: {VALID_MODEL_TYPES}")
    
def verify_task_type(filename_path, method):
    filename = os.path.basename(filename_path)
    task_type = filename.split('_')[1]
    if task_type.lower() not in VALID_TASK_TYPES:
        raise ValueError(f"Invalid task type '{task_type}'. Valid types are: {VALID_TASK_TYPES}")
    if task_type.lower() != method.lower():
        raise ValueError(f"The provided model file '{filename}' does not match the expected task type '{method}'.")


def parse_for_embeddings(dataset, task_type):
    dataset_dict = {}
    labels_dict = [] # will return np.array for labels
    EMBEDDING_SIZES = BINARY_EMBEDDING_VOCAB_SIZES if task_type.lower() == 'binary' else REGRESSION_EMBEDDING_VOCAB_SIZES

    # Note: numerical columns are combined into a single input for LSTM and must be first in the dictionary
    # add numerical columns and remove labels and Date
    numerical_columns = []
    for col in dataset.columns:
        drop_cols = ['Date', 'Lead_Time_Days', 'Disruption_Occurred']
        if task_type.lower() == 'regression2':
            drop_cols = ['Date', 'Lead_Time_Days']
        if col not in list(EMBEDDING_SIZES.keys()) + drop_cols:
            numerical_columns.append(col)
    
    dataset_dict['numerical_inputs'] = add_timesteps(dataset[numerical_columns].values)

    # categorical columns , note: the _input suffix is required for the LSTM model input names
    for col in EMBEDDING_SIZES.keys():
        input_col = dataset[col].values
        dataset_dict[f"{col}_input"] = add_timesteps(input_col.reshape(-1, 1)) # reshape from (samples,) to (samples, 1) before adding timesteps
    
    if task_type.lower() == 'binary':
        labels_dict = add_timesteps(dataset['Disruption_Occurred'].values)
        # if last layer in model is return_sequences=False, flatten labels to (samples,)
        #labels_dict = labels_dict[:, -1] # take last time step
        #labels_dict = labels_dict.reshape(-1,)  # reshape from (samples, 1) to (samples,)
    elif task_type.lower() == 'regression' or task_type.lower() == 'regression2':
        labels_dict = add_timesteps(dataset['Lead_Time_Days'].values)
        #labels_dict = labels_dict[:, -1] # take last time step
        #labels_dict = labels_dict.reshape(-1,)  # reshape from (samples, 1) to (samples,)
    
    return dataset_dict, labels_dict

def add_timesteps(input_data, timesteps=5): # both LSTM_INPUT_SHAPE have a timestep of 5
    # convert from (samples, features) to (samples, timesteps, features)
    sequences = []
    for i in range(len(input_data) - timesteps + 1):
            sequences.append(input_data[i:i+timesteps])

    return np.array(sequences)

def is_stage2_regression_model(model_filename_path):
    filename = os.path.basename(model_filename_path)
    is_stage2 = filename.split('_')[3]
    return is_stage2.lower() == 'stage2'
