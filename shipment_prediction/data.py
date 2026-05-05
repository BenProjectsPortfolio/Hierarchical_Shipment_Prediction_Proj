import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(model_type, task_type):
    dataset_path = '.\dataset\global_supply_chain_risk_2026.csv'
    if sys.platform == "linux" or sys.platform == "darwin":
        dataset_path = './dataset/global_supply_chain_risk_2026.csv'
        
    column_names = [
        "Shipment_ID",                  # Unique alphanumeric identifier for each cargo movement.
        "Date",                         # Dispatch date of the shipment (Range: 2024-2026).
        "Origin_Port",                  # The starting point of the shipment (Major global hub).
        "Destination_Port",             # The intended arrival port for the cargo.
        "Transport_Mode",               # Method of transport: Sea, Air, Rail, or Road.
        "Product_Category",             # Type of goods being moved (e.g., Electronics, Pharmaceuticals).
        "Distance_km",                  # Total travel distance between ports in kilometers.
        "Weight_MT",                    # Weight of the shipment in Metric Tons.
        "Fuel_Price_Index",             # Normalized fuel cost multiplier at the time of departure.
        "Geopolitical_Risk_Score",      # Risk index (0-10) based on regional stability (0=Stable, 10=High Conflict).
        "Weather_Condition",            # Atmospheric conditions during transit (Clear, Rain, Storm, Hurricane, etc.)
        "Carrier_Reliability_Score",    # Historical performance score of the carrier (0.5 to 1.0).
        "Lead_Time_Days",               # Actual time taken for delivery (Target for Regression).
        "Disruption_Occurred",          # Binary flag: 1 if the shipment was delayed or cancelled, 0 otherwise.
    ]

    dataset = pd.read_csv(dataset_path, names=column_names, header=0)
    dataset = dataset[column_names]

    dataset = generate_new_features(dataset, model_type, task_type)

    return dataset

def generate_new_features(dataset, model_type, task_type): 
    # Sort by Date, then group by Route_Mode to create grouping key
    dataset = dataset.sort_values(by='Date')
    dataset['Route'] = dataset['Origin_Port'] + "_" + dataset['Destination_Port']
    dataset['Route_Mode'] = dataset['Route'] + "_" + dataset['Transport_Mode']
    dataset = dataset.sort_values(['Route_Mode', 'Date', 'Shipment_ID']).reset_index(drop=True)

    # Add time features
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.month
    dataset['Quarter'] = dataset['Date'].dt.quarter
    dataset['Day'] = dataset['Date'].dt.day
    dataset['Day_of_Week'] = dataset['Date'].dt.dayofweek # 0=Monday, 6=Sunday
    dataset['Week_of_Year'] = dataset['Date'].dt.isocalendar().week.astype(int)
    dataset['Is_Month_Start'] = dataset['Date'].dt.is_month_start.astype(int)
    dataset['Is_Month_End'] = dataset['Date'].dt.is_month_end.astype(int)
    dataset['Month_Sin'] = np.sin(2 * np.pi * dataset['Month'] / 12)
    dataset['Month_Cos'] = np.cos(2 * np.pi * dataset['Month'] / 12)
    dataset['Day_of_Week_Sin'] = np.sin(2 * np.pi * dataset['Day_of_Week'] / 7)
    dataset['Day_of_Week_Cos'] = np.cos(2 * np.pi * dataset['Day_of_Week'] / 7)

    # Lag features - past features in the same Route_Mode group
    group = dataset.groupby('Route_Mode', group_keys=False)
    lag_columns = ['Distance_km', 'Weight_MT', 'Fuel_Price_Index', 'Geopolitical_Risk_Score',
                   'Carrier_Reliability_Score', 'Lead_Time_Days', 'Disruption_Occurred']
    for column in lag_columns: # shifted to aviod data leakage
        dataset[f'{column}_Lag1'] = group[column].shift(1)
        dataset[f'{column}_Lag2'] = group[column].shift(2)
        dataset[f'{column}_Lag3'] = group[column].shift(3)
    
    # Rolling features - past rolling mean and std averages in the same Route_Mode group
    rolling_columns = ['Lead_Time_Days', 'Disruption_Occurred', 'Fuel_Price_Index', 'Geopolitical_Risk_Score', 'Carrier_Reliability_Score']
    for column in rolling_columns: # shifted to aviod data leakage
        dataset[f'{column}_Rolling3_Mean'] = (group[column].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()))
        dataset[f'{column}_Rolling5_Mean'] = (group[column].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()))
        dataset[f'{column}_Rolling3_Std'] = (group[column].transform(lambda x: x.shift(1).rolling(window=3, min_periods=2).std()))
    
    # Historical averages per Route_Mode group
    dataset['Historical_Disruption_Rate'] = (group['Disruption_Occurred'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean()))
    dataset['Historical_Avg_Lead_Time'] = (group['Lead_Time_Days'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean()))

    # Days since previous shipment in the same Route_Mode group
    dataset['Previous_Shipment_Date'] = group['Date'].shift(1)
    dataset['Days_Since_Previous_Shipment'] = (dataset['Date'] - dataset['Previous_Shipment_Date']).dt.days

    # Create categorical features into numerical codes
    categorical_columns = ['Origin_Port', 'Destination_Port', 'Transport_Mode', 'Product_Category', 'Weather_Condition', 'Route', 'Route_Mode']
    for col in categorical_columns:
        dataset[col] = dataset[col].astype('category').cat.codes

    # Keep only rows where Lag1 columns is not NaN - otherwise the shipment has no historical data
    has_content = dataset.filter(like='_Lag1').notna().any(axis=1)
    dataset = dataset.loc[has_content].reset_index(drop=True)

    # Drop unused features
    drop_columns = ['Shipment_ID', 'Date', 'Previous_Shipment_Date']
    #drop_columns = ['Shipment_ID', 'Previous_Shipment_Date']
    feature_columns = [c for c in dataset.columns if c not in drop_columns]
    dataset = dataset[feature_columns]
    
    # Handle NaN values in Lag and Rolling columns by filling with 0
    lag_columns = [col for col in dataset.columns if '_Lag' in col]
    dataset[lag_columns] = dataset[lag_columns].fillna(0)
    rolling_std_columns = [col for col in dataset.columns if '_Rolling3_Std' in col]
    dataset[rolling_std_columns] = dataset[rolling_std_columns].fillna(0)

    # For testing, print dataset and generate dataset to csv file
    #print(dataset.head())
    #dataset.to_csv('test_new_features_dataset.csv')

    # Drop features based on feature selection experiments - This is the Final Feature Set for all models
    #   update features number in LSTM_INPUT_SHAPE[1] in helper.py 
    #   subtract total amount minus uncommented features below. ie: 63 - n = LSTM_INPUT_SHAPE[1]
    #       also, comment out the uncomment features below, in helper.py
    #       this is for tracking numerical features when performing feature importance analysis
    #          Note: drop_columns1 is for binary classification, drop_columns2 is for regression (with disruption flag for 2 stage regression)
    drop_columns1 = [
        #'Origin_Port', 
        'Destination_Port', 
        'Transport_Mode',
        'Product_Category', 
        'Distance_km', 
        'Weight_MT', 
        #'Fuel_Price_Index',
        #'Geopolitical_Risk_Score', 
        #'Weather_Condition',
        #'Carrier_Reliability_Score', 
        'Route', 
        'Route_Mode', 
        'Year', 
        'Month', 
        'Quarter', 
        'Day', 
        #'Day_of_Week',
        #'Week_of_Year', 
        #'Is_Month_Start', 
        #'Is_Month_End', 
        #'Month_Sin',
        #'Month_Cos', 
        #'Day_of_Week_Sin', 
        #'Day_of_Week_Cos', 
        'Distance_km_Lag1',
        'Distance_km_Lag2', 
        'Distance_km_Lag3', 
        'Weight_MT_Lag1',
        'Weight_MT_Lag2', 
        'Weight_MT_Lag3', 
        #'Fuel_Price_Index_Lag1',
        'Fuel_Price_Index_Lag2', 
        'Fuel_Price_Index_Lag3',
        'Geopolitical_Risk_Score_Lag1', 
        'Geopolitical_Risk_Score_Lag2',
        'Geopolitical_Risk_Score_Lag3', 
        #'Carrier_Reliability_Score_Lag1',
        'Carrier_Reliability_Score_Lag2', 
        'Carrier_Reliability_Score_Lag3',
        'Lead_Time_Days_Lag1', 
        'Lead_Time_Days_Lag2', 
        'Lead_Time_Days_Lag3',
        #'Disruption_Occurred_Lag1', 
        'Disruption_Occurred_Lag2',
        'Disruption_Occurred_Lag3', 
        'Lead_Time_Days_Rolling3_Mean',
        'Lead_Time_Days_Rolling5_Mean', 
        'Lead_Time_Days_Rolling3_Std',
        'Disruption_Occurred_Rolling3_Mean',
        'Disruption_Occurred_Rolling5_Mean', 
        'Disruption_Occurred_Rolling3_Std',
        'Fuel_Price_Index_Rolling3_Mean', 
        'Fuel_Price_Index_Rolling5_Mean',
        'Fuel_Price_Index_Rolling3_Std',
        'Geopolitical_Risk_Score_Rolling3_Mean',
        'Geopolitical_Risk_Score_Rolling5_Mean',
        'Geopolitical_Risk_Score_Rolling3_Std',
        'Carrier_Reliability_Score_Rolling3_Mean',
        'Carrier_Reliability_Score_Rolling5_Mean',
        'Carrier_Reliability_Score_Rolling3_Std', 
        #'Historical_Disruption_Rate',
        'Historical_Avg_Lead_Time', 
        #'Days_Since_Previous_Shipment',
    ]
    drop_columns2 = [
        'Origin_Port', 
        #'Destination_Port', 
        #'Transport_Mode',
        #'Product_Category', 
        #'Distance_km', 
        'Weight_MT', 
        #'Fuel_Price_Index',
        #'Geopolitical_Risk_Score', 
        #'Weather_Condition',
        #'Carrier_Reliability_Score', 
        'Route', 
        'Route_Mode', 
        'Year', 
        'Month', 
        'Quarter', 
        'Day', 
        #'Day_of_Week',
        #'Week_of_Year', 
        #'Is_Month_Start', 
        #'Is_Month_End', 
        'Month_Sin',
        'Month_Cos', 
        #'Day_of_Week_Sin', 
        #'Day_of_Week_Cos', 
        #'Distance_km_Lag1',
        'Distance_km_Lag2', 
        'Distance_km_Lag3', 
        'Weight_MT_Lag1',
        'Weight_MT_Lag2', 
        'Weight_MT_Lag3', 
        #'Fuel_Price_Index_Lag1',
        'Fuel_Price_Index_Lag2', 
        'Fuel_Price_Index_Lag3',
        'Geopolitical_Risk_Score_Lag1', 
        'Geopolitical_Risk_Score_Lag2',
        'Geopolitical_Risk_Score_Lag3', 
        #'Carrier_Reliability_Score_Lag1',
        'Carrier_Reliability_Score_Lag2', 
        'Carrier_Reliability_Score_Lag3',
        'Lead_Time_Days_Lag1', 
        'Lead_Time_Days_Lag2', 
        'Lead_Time_Days_Lag3',
        #'Disruption_Occurred_Lag1', 
        'Disruption_Occurred_Lag2',
        'Disruption_Occurred_Lag3', 
        'Lead_Time_Days_Rolling3_Mean',
        'Lead_Time_Days_Rolling5_Mean', 
        'Lead_Time_Days_Rolling3_Std',
        'Disruption_Occurred_Rolling3_Mean',
        'Disruption_Occurred_Rolling5_Mean', 
        'Disruption_Occurred_Rolling3_Std',
        'Fuel_Price_Index_Rolling3_Mean', 
        'Fuel_Price_Index_Rolling5_Mean',
        'Fuel_Price_Index_Rolling3_Std',
        'Geopolitical_Risk_Score_Rolling3_Mean',
        'Geopolitical_Risk_Score_Rolling5_Mean',
        'Geopolitical_Risk_Score_Rolling3_Std',
        'Carrier_Reliability_Score_Rolling3_Mean',
        'Carrier_Reliability_Score_Rolling5_Mean',
        'Carrier_Reliability_Score_Rolling3_Std', 
        #'Historical_Disruption_Rate',
        'Historical_Avg_Lead_Time', 
        'Days_Since_Previous_Shipment',
    ]

    if task_type.lower() == 'binary':
        drop_columns = drop_columns1
    else: # regression or regression2
        drop_columns = drop_columns2

    feature_columns = [c for c in dataset.columns if c not in drop_columns]
    dataset = dataset[feature_columns]

    return dataset


def get_binary_inputs(dataset, task_type):
    x_binary = dataset.drop(['Lead_Time_Days', 'Disruption_Occurred'], axis=1)
    y_binary = dataset['Disruption_Occurred']
    return x_binary, y_binary

def get_regression_inputs(dataset, task_type):
    if task_type.lower() == 'regression':
        x_regression = dataset.drop(['Lead_Time_Days', 'Disruption_Occurred'], axis=1)
    elif task_type.lower() == 'regression2':
        x_regression = dataset.drop(['Lead_Time_Days'], axis=1) # train regression model stage 2 with disruption flag for 2 stage prediction
    y_regression = dataset['Lead_Time_Days']
    return x_regression, y_regression

def get_dataset_splits(dataset, task_type, val_and_test_size=0.2, random_state=42):
    # get inputs and labels based on training type
    if task_type.lower() == 'binary':
        input_data, labels = get_binary_inputs(dataset, task_type)
    else: # regression or regression2
        input_data, labels = get_regression_inputs(dataset, task_type)

    # Split datasets into training, validation, and testing sets
    ##
    # Default Split: 80% train, 20% temp (10% val, 10% test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        input_data, labels, test_size=val_and_test_size, random_state=random_state
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=random_state
    )

    return x_train, y_train, x_val, y_val, x_test, y_test
