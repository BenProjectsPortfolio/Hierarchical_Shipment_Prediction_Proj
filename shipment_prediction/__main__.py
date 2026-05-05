from datetime import datetime
import random
import numpy as np
import tensorflow as tf
import joblib
import argparse
from shipment_prediction.data import get_dataset_splits, load_dataset
from shipment_prediction.train import binary_prediction, full_pipeline_prediction, regression_prediction, train_lstm_binary_model, train_lstm_regression_model, train_mlp_binary_model, train_mlp_regression_model, train_xgb_binary_model, train_xgb_regression_model
from shipment_prediction.utils.helper import get_model_type, is_stage2_regression_model, verify_task_type, plot_overall_data_analysis

# Set random seeds for reproducibility
random.seed(19)
np.random.seed(19)
tf.random.set_seed(19)


def begin_training_xgb_binary():
    print("Training XGB Binary Model")

    dataset = load_dataset(model_type='xgb', task_type='binary')
    binary_input_train, binary_label_train, binary_input_val, binary_label_val, binary_input_test, binary_label_test = get_dataset_splits(dataset, task_type='binary')

    binary_classifier = train_xgb_binary_model(
        binary_input_train, binary_label_train, 
        binary_input_val, binary_label_val)
    
    binary_prediction(binary_input_test, binary_label_test, binary_classifier, model_type='xgb')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(binary_classifier, f'./models/xgb_binary_model_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING XGB BINARY CLASSIFIER ##########")

def begin_training_xgb_regression():
    print("Training XGB Regression Model")

    dataset = load_dataset(model_type='xgb', task_type='regression')
    rg_input_train, rg_label_train, rg_input_val, rg_label_val, rg_input_test, rg_label_test = get_dataset_splits(dataset, task_type='regression')

    regression_model = train_xgb_regression_model(
        rg_input_train, rg_label_train,
        rg_input_val, rg_label_val)
    
    regression_prediction(rg_input_test, rg_label_test, regression_model, model_type='xgb')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(regression_model, f'./models/xgb_regression_model_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING XGB REGRESSION MODEL ##########")

def begin_training_xgb_regression2():
    print("Training XGB Regression Model for Stage 2")

    dataset = load_dataset(model_type='xgb', task_type='regression2')
    rg_input_train, rg_label_train, rg_input_val, rg_label_val, rg_input_test, rg_label_test = get_dataset_splits(dataset, task_type='regression2')

    regression_model = train_xgb_regression_model(
        rg_input_train, rg_label_train,
        rg_input_val, rg_label_val)
    
    regression_prediction(rg_input_test, rg_label_test, regression_model, model_type='xgb')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(regression_model, f'./models/xgb_regression_model_stage2_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING XGB REGRESSION MODEL FOR STAGE 2 ##########")

def begin_training_mlp_binary():
    print("Training MLP Binary Model")
    
    dataset = load_dataset(model_type='mlp', task_type='binary')
    binary_input_train, binary_label_train, binary_input_val, binary_label_val, binary_input_test, binary_label_test = get_dataset_splits(dataset, task_type='binary')

    binary_classifier = train_mlp_binary_model(
        binary_input_train, binary_label_train, 
        binary_input_val, binary_label_val)
    
    binary_prediction(binary_input_test, binary_label_test, binary_classifier, model_type='mlp')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(binary_classifier, f'./models/mlp_binary_model_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING MLP BINARY MODEL ##########")

def begin_training_mlp_regression():
    print("Training MLP Regression Model")
    
    dataset = load_dataset(model_type='mlp', task_type='regression')
    rg_input_train, rg_label_train, rg_input_val, rg_label_val, rg_input_test, rg_label_test = get_dataset_splits(dataset, task_type='regression')

    regression_model = train_mlp_regression_model(
        rg_input_train, rg_label_train,
        rg_input_val, rg_label_val)
    
    regression_prediction(rg_input_test, rg_label_test, regression_model, model_type='mlp')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(regression_model, f'./models/mlp_regression_model_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING MLP REGRESSION MODEL ##########")

def begin_training_mlp_regression2():
    print("Training MLP Regression Model for Stage 2")
    
    dataset = load_dataset(model_type='mlp', task_type='regression2')
    rg_input_train, rg_label_train, rg_input_val, rg_label_val, rg_input_test, rg_label_test = get_dataset_splits(dataset, task_type='regression2')

    regression_model = train_mlp_regression_model(
        rg_input_train, rg_label_train,
        rg_input_val, rg_label_val)
    
    regression_prediction(rg_input_test, rg_label_test, regression_model, model_type='mlp')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(regression_model, f'./models/mlp_regression_model_stage2_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING MLP REGRESSION MODEL FOR STAGE 2 ##########")

def begin_training_lstm_binary():
    print("Training LSTM Binary Model")
    
    dataset = load_dataset(model_type='lstm', task_type='binary')
    binary_input_train, binary_label_train, binary_input_val, binary_label_val, binary_input_test, binary_label_test = get_dataset_splits(dataset, task_type='binary')

    binary_classifier = train_lstm_binary_model(
        binary_input_train, binary_label_train, 
        binary_input_val, binary_label_val)
    
    binary_prediction(binary_input_test, binary_label_test, binary_classifier, model_type='lstm')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(binary_classifier, f'./models/lstm_binary_model_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING LSTM BINARY MODEL ##########")

def begin_training_lstm_regression():
    print("Training LSTM Regression Model")
    
    dataset = load_dataset(model_type='lstm', task_type='regression')
    rg_input_train, rg_label_train, rg_input_val, rg_label_val, rg_input_test, rg_label_test = get_dataset_splits(dataset, task_type='regression')

    regression_model = train_lstm_regression_model(
        rg_input_train, rg_label_train,
        rg_input_val, rg_label_val)
    
    regression_prediction(rg_input_test, rg_label_test, regression_model, model_type='lstm')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(regression_model, f'./models/lstm_regression_model_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING LSTM REGRESSION MODEL ##########")

def begin_training_lstm_regression2():
    print("Training LSTM Regression Model for Stage 2")
    
    dataset = load_dataset(model_type='lstm', task_type='regression2')
    rg_input_train, rg_label_train, rg_input_val, rg_label_val, rg_input_test, rg_label_test = get_dataset_splits(dataset, task_type='regression2')

    regression_model = train_lstm_regression_model(
        rg_input_train, rg_label_train,
        rg_input_val, rg_label_val,
        task_type='regression2')
    
    regression_prediction(rg_input_test, rg_label_test, regression_model, model_type='lstm', task_type='regression2')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(regression_model, f'./models/lstm_regression_model_stage2_{timestamp}.joblib')

    print("\n \n########## ENDED TRAINING LSTM REGRESSION MODEL FOR STAGE 2 ##########")

def test_both_models(binary_model_path, regression_model_path):
    verify_task_type(binary_model_path, 'binary')
    binary_model_type = get_model_type(binary_model_path)
    binary_model = joblib.load(binary_model_path)
    binary_dataset = load_dataset(binary_model_type, task_type='binary')

    verify_task_type(regression_model_path, 'regression')
    regression_model_type = get_model_type(regression_model_path)
    regression_model = joblib.load(regression_model_path)
    regression_dataset = load_dataset(regression_model_type, task_type='regression')

    # Get the test set input data for prediction
    _,_,_,_,binary_input_data, binary_labels = get_dataset_splits(binary_dataset, task_type='binary')
    task_type = 'regression'
    is_stage2 = is_stage2_regression_model(regression_model_path)
    if is_stage2: task_type = 'regression2'
    _,_,_,_,regression_input_data, regression_labels = get_dataset_splits(regression_dataset, task_type=task_type)

    full_pipeline_prediction(
        binary_input_data, binary_labels,
        regression_input_data, regression_labels,
        binary_model, binary_model_type,
        regression_model, regression_model_type,
        is_regression_stage2=is_stage2
    )

def test_binary_model(binary_model_path):
    verify_task_type(binary_model_path, 'binary')
    binary_model_type = get_model_type(binary_model_path)
    binary_model = joblib.load(binary_model_path)
    dataset = load_dataset(binary_model_type, task_type='binary')

    # Get the test set input data for prediction
    _,_,_,_,input_data, labels = get_dataset_splits(dataset, task_type='binary')
    
    binary_prediction(
        input_data, labels, 
        binary_model, binary_model_type
    )

def test_regression_model(regression_model_path):
    verify_task_type(regression_model_path, 'regression')
    regression_model_type = get_model_type(regression_model_path)
    regression_model = joblib.load(regression_model_path)
    dataset = load_dataset(regression_model_type, task_type='regression')

    # Get the test set input data for prediction
    _,_,_,_,input_data, labels = get_dataset_splits(dataset, task_type='regression')

    task_type = 'regression'
    if is_stage2_regression_model(regression_model_path):
        task_type = 'regression2'
        _,_,_,_,input_data, labels = get_dataset_splits(dataset, task_type=task_type)

    regression_prediction(
        input_data, labels, 
        regression_model, regression_model_type,
        task_type=task_type
    )


def select_training_model_type(train_flag, model_type):
    if train_flag.lower() == 'xgb' and model_type.lower() == 'binary':
        begin_training_xgb_binary()
    elif train_flag.lower() == 'xgb' and model_type.lower() == 'regression':
        begin_training_xgb_regression()
    elif train_flag.lower() == 'xgb' and model_type.lower() == 'regression2':
        begin_training_xgb_regression2()
    elif train_flag.lower() == 'xgb' and model_type.lower() == 'both':
        begin_training_xgb_binary()
        begin_training_xgb_regression()
    elif train_flag.lower() == 'mlp' and model_type.lower() == 'binary':
        begin_training_mlp_binary()
    elif train_flag.lower() == 'mlp' and model_type.lower() == 'regression':
        begin_training_mlp_regression()
    elif train_flag.lower() == 'mlp' and model_type.lower() == 'regression2':
        begin_training_mlp_regression2()
    elif train_flag.lower() == 'mlp' and model_type.lower() == 'both':
        begin_training_mlp_binary()
        begin_training_mlp_regression()
    elif train_flag.lower() == 'lstm' and model_type.lower() == 'binary':
        begin_training_lstm_binary()
    elif train_flag.lower() == 'lstm' and model_type.lower() == 'regression':
        begin_training_lstm_regression()
    elif train_flag.lower() == 'lstm' and model_type.lower() == 'regression2':
        begin_training_lstm_regression2()
    elif train_flag.lower() == 'lstm' and model_type.lower() == 'both':
        begin_training_lstm_binary()
        begin_training_lstm_regression()
    elif train_flag.lower() == 'all' and model_type.lower() == 'both':
        begin_training_xgb_binary()
        begin_training_xgb_regression()
        begin_training_mlp_binary()
        begin_training_mlp_regression()
        begin_training_lstm_binary()
        begin_training_lstm_regression()
    else:
        print("Unsupported training configuration. Please specify --train (XGB/MLP/LSTM/all) and --type (binary/regression/both).")



if __name__ == "__main__":
    print("\n\n##### Shipment Prediction Package #####\n")
    parser = argparse.ArgumentParser(description="Shipment Prediction Training and Testing")
    parser.add_argument('--train', type=str, help='Model to train: XGB, MLP, LSTM, or all')
    parser.add_argument('--type', type=str, help='Type of model: binary, regression, regression2, or both')
    parser.add_argument('--binary_model_path', type=str, help='Path to the trained binary model for prediction')
    parser.add_argument('--regression_model_path', type=str, help='Path to the trained regression model for prediction')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print("Parameters passed:", args) # Namespace object with arguments

    if args.debug:
        print("Debug mode is enabled.")
        print("Area for testing.....\n")
        ###############################
        plot_overall_data_analysis(load_dataset("lstm", "binary"))
        plot_overall_data_analysis(load_dataset("lstm", "regression"))
        ###############################

    if args.train and not args.type:
        print("Please specify the type of model to train using --type (binary/regression/regression2/both).")
    elif not args.train and args.type:
        print("Please specify the model to train using --train (XGB/MLP/LSTM/all).")

    if args.train and args.type:
        select_training_model_type(args.train, args.type)

    if args.binary_model_path or args.regression_model_path:
        if args.binary_model_path and args.regression_model_path:
            test_both_models(
                args.binary_model_path,
                args.regression_model_path
            )
        elif args.binary_model_path:
            test_binary_model(
                args.binary_model_path
            )
        elif args.regression_model_path:
            test_regression_model(
                args.regression_model_path
            )

    if not args.train and not args.type and not args.binary_model_path and not args.regression_model_path:
        print("No arguments provided. Use --help for usage information.")
