import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from shipment_prediction.utils.helper import BINARY_EMBEDDING_VOCAB_SIZES, BINARY_LSTM_INPUT_SHAPE, REGRESSION_EMBEDDING_VOCAB_SIZES, REGRESSION_LSTM_INPUT_SHAPE, get_model_name, plot_feature_importances, parse_for_embeddings
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate
from keras.optimizers import Adam


def create_lstm_binary_model(
    hidden_layer=64,
    learning_rate=0.001,
    dropout_rate=0.2,
    activation="relu",
    **kwargs
):
    inputs = []
    numerical_inputs = Input(shape=(BINARY_LSTM_INPUT_SHAPE[0], BINARY_LSTM_INPUT_SHAPE[1] - len(BINARY_EMBEDDING_VOCAB_SIZES)), name='numerical_inputs')
    inputs.append(numerical_inputs)

    sequence_parts = [numerical_inputs]

    for col, vocab_size in BINARY_EMBEDDING_VOCAB_SIZES.items():
        categorical_input = Input(shape=(BINARY_LSTM_INPUT_SHAPE[0],), dtype='int32', name=f"{col}_input")
        inputs.append(categorical_input)

        embedding_dim = min(50, max(4, (vocab_size + 1) // 2))  # Ensure at least dimension of 4
        embedding_layer = Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            name=f"{col}_embedding"
        )(categorical_input)

        sequence_parts.append(embedding_layer)
    
    x = Concatenate(axis=-1, name=f"concatenate_features")(sequence_parts)

    x = LSTM(hidden_layer, return_sequences=True, name='lstm_1')(x)
    x = LSTM(hidden_layer // 2, return_sequences=True, name='lstm_2')(x)
    x = Dense(hidden_layer ** 2, activation=activation, name='dense_1')(x)
    x = Dropout(dropout_rate, name='dropout_1')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

def create_lstm_regression_model(
    hidden_layer=64,
    learning_rate=0.001,
    dropout_rate=0.2,
    activation="relu",
    numerical_input_shape=(REGRESSION_LSTM_INPUT_SHAPE[0], REGRESSION_LSTM_INPUT_SHAPE[1] - len(REGRESSION_EMBEDDING_VOCAB_SIZES)),
    **kwargs
):
    inputs = []
    numerical_inputs = Input(shape=numerical_input_shape, name='numerical_inputs')
    inputs.append(numerical_inputs)

    sequence_parts = [numerical_inputs]

    for col, vocab_size in REGRESSION_EMBEDDING_VOCAB_SIZES.items():
        categorical_input = Input(shape=(REGRESSION_LSTM_INPUT_SHAPE[0],), dtype='int32', name=f"{col}_input")
        inputs.append(categorical_input)

        embedding_dim = min(50, max(4, (vocab_size + 1) // 2))  # Ensure at least dimension of 4
        embedding_layer = Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            name=f"{col}_embedding"
        )(categorical_input)

        sequence_parts.append(embedding_layer)
    
    x = Concatenate(axis=-1, name=f"concatenate_features")(sequence_parts)

    x = LSTM(hidden_layer, return_sequences=True, name='lstm_1')(x)
    x = LSTM(hidden_layer // 2, return_sequences=True, name='lstm_2')(x)
    x = Dense(hidden_layer ** 2, activation=activation, name='dense_1')(x)
    x = Dropout(dropout_rate, name='dropout_1')(x)
    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, 
        loss='mean_squared_error', 
        metrics=['mse']
    )
    return model


def train_xgb_binary_model(binary_input_train, binary_label_train, binary_input_val, binary_label_val):
    pipeline = Pipeline([
        ('xgb_binary', XGBClassifier(enable_categorical=True, random_state=42))
    ])

    # Define the hyperparameter search space to find optimal hyperparameters, scikit-optimize
    search_space = {
        'xgb_binary__max_depth': Integer(2, 8),
        'xgb_binary__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'xgb_binary__subsample': Real(0.5, 1.0),
        'xgb_binary__colsample_bytree': Real(0.5, 1.0),
        'xgb_binary__colsample_bylevel': Real(0.5, 1.0),
        'xgb_binary__colsample_bynode': Real(0.5, 1.0),
        'xgb_binary__reg_alpha': Real(0.0, 10.0),
        'xgb_binary__reg_lambda': Real(0.0, 10.0),
        'xgb_binary__gamma': Real(0.0, 10.0),
        'xgb_binary__n_estimators': Integer(50, 500),
        'xgb_binary__max_iter': Integer(100, 500)
    }
    opt = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        scoring='roc_auc'
    )

    print(binary_input_train.columns)
    print(binary_label_train.name)

    eval_set = [(binary_input_train, binary_label_train), (binary_input_val, binary_label_val)]
    opt.fit(
        binary_input_train,
        binary_label_train,
        xgb_binary__eval_set=eval_set,
        xgb_binary__verbose=1
    )
    print("\nBest Hyperparameters: ", opt.best_params_)
    print("\nBest Estimator: ", opt.best_estimator_)
    print("\nBest AUC Score: ", opt.best_score_)
    print("Accuracy (using .score()):", opt.score(binary_input_val, binary_label_val))

    return opt

def train_xgb_regression_model(rg_input_train, rg_label_train, rg_input_val, rg_label_val):
    pipeline = Pipeline([
        ('xgb_regression', XGBRegressor(enable_categorical=True, random_state=42))
    ])

    # Define the hyperparameter search space to find optimal hyperparameters, scikit-optimize
    search_space = {
        'xgb_regression__max_depth': Integer(2, 8),
        'xgb_regression__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'xgb_regression__subsample': Real(0.5, 1.0),
        'xgb_regression__colsample_bytree': Real(0.5, 1.0),
        'xgb_regression__colsample_bylevel': Real(0.5, 1.0),
        'xgb_regression__colsample_bynode': Real(0.5, 1.0),
        'xgb_regression__reg_alpha': Real(0.0, 10.0),
        'xgb_regression__reg_lambda': Real(0.0, 10.0),
        'xgb_regression__gamma': Real(0.0, 10.0),
        'xgb_regression__n_estimators': Integer(50, 500),
        'xgb_regression__max_iter': Integer(100, 500)
    }
    opt = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        scoring='neg_root_mean_squared_error'
    )

    print(rg_input_train.columns)
    print(rg_label_train.name)

    eval_set = [(rg_input_train, rg_label_train), (rg_input_val, rg_label_val)]
    opt.fit(
        rg_input_train,
        rg_label_train,
        xgb_regression__eval_set=eval_set,
        xgb_regression__verbose=1
    )
    print("\nBest Hyperparameters: ", opt.best_params_)
    print("\nBest Estimator: ", opt.best_estimator_)
    print("\nBest Negative Root Mean Squared Error Score: ", opt.best_score_)
    print("Accuracy (using .score()):", opt.score(rg_input_val, rg_label_val))

    return opt

def train_mlp_binary_model(binary_input_train, binary_label_train, binary_input_val, binary_label_val):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp_binary', MLPClassifier(random_state=42))
    ])

    # Define the hyperparameter search space to find optimal hyperparameters, scikit-optimize
    search_space = {
        'mlp_binary__hidden_layer_sizes': Categorical([ 32, 64, 128, 256 ]),
        'mlp_binary__activation': Categorical(['relu', 'tanh', 'logistic']),
        'mlp_binary__solver': Categorical(['adam', 'sgd']),
        'mlp_binary__alpha': Real(0.0001, 0.1, prior='log-uniform'),
        'mlp_binary__learning_rate_init': Real(0.001, 0.1, prior='log-uniform'),
        'mlp_binary__max_iter': Integer(100, 500)
    }
    opt = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        scoring='roc_auc'
    )

    print(binary_input_train.columns)
    print(binary_label_train.name)
    
    opt.fit(
        binary_input_train,
        binary_label_train,
    )
    print("\nBest Hyperparameters: ", opt.best_params_)
    print("\nBest Estimator: ", opt.best_estimator_)
    print("\nBest AUC Score: ", opt.best_score_)
    print("Accuracy (using .score()):", opt.score(binary_input_val, binary_label_val))

    return opt

def train_mlp_regression_model(regression_input_train, regression_label_train, regression_input_val, regression_label_val):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp_regression', MLPRegressor(random_state=42))
    ])

    # Define the hyperparameter search space to find optimal hyperparameters, scikit-optimize
    search_space = {
        'mlp_regression__hidden_layer_sizes': Categorical([ 32, 64, 128, 256 ]),
        'mlp_regression__activation': Categorical(['relu', 'tanh']),
        'mlp_regression__solver': Categorical(['adam', 'sgd']),
        'mlp_regression__alpha': Real(0.0001, 0.1, prior='log-uniform'),
        'mlp_regression__learning_rate_init': Real(1e-5, 1e-2, prior='log-uniform'),
        'mlp_regression__max_iter': Integer(100, 500)
    }
    opt = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        scoring='neg_root_mean_squared_error'
    )

    print(regression_input_train.columns)
    print(regression_label_train.name)

    opt.fit(
        regression_input_train,
        regression_label_train,
    )
    print("\nBest Hyperparameters: ", opt.best_params_)
    print("\nBest Estimator: ", opt.best_estimator_)
    print("\nBest Neg Root Mean Squared Error Score: ", opt.best_score_)
    print("Accuracy (using .score()):", opt.score(regression_input_val, regression_label_val))

    return opt

def train_lstm_binary_model(binary_input_train, binary_label_train, binary_input_val, binary_label_val):
    data_train = pd.concat([binary_input_train, binary_label_train], axis=1)
    data_val = pd.concat([binary_input_val, binary_label_val], axis=1)
    binary_input_train, binary_label_train = parse_for_embeddings(data_train, task_type="binary")
    binary_input_val, binary_label_val = parse_for_embeddings(data_val, task_type="binary")

    print(data_train.columns)
    print("Note: label is included in the printed columns.")

    model = create_lstm_binary_model()
    model.fit(
        binary_input_train,
        binary_label_train,
        epochs=1000,
        batch_size=16,
        validation_data=(binary_input_val, binary_label_val)
    )
    
    y_pred = model.predict(binary_input_val)  # shape (n_samples, 1)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to class labels

    # If y_test shape is (n_samples, 1), flatten it:
    y_true = binary_label_val.flatten()
    accuracy = accuracy_score(y_true, y_pred_class)
    roc_auc = roc_auc_score(y_true, y_pred.flatten())

    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)

    return model

def train_lstm_regression_model(regression_input_train, regression_label_train, regression_input_val, regression_label_val, task_type='regression'):
    data_train = pd.concat([regression_input_train, regression_label_train], axis=1)
    data_val = pd.concat([regression_input_val, regression_label_val], axis=1)
    regression_input_train, regression_label_train = parse_for_embeddings(data_train, task_type=task_type)
    regression_input_val, regression_label_val = parse_for_embeddings(data_val, task_type=task_type)

    print(data_train.columns)
    print("Note: label is included in the printed columns.")

    model = create_lstm_regression_model()
    if task_type.lower() == 'regression2': # Add 1 for the additional feature in regression 2 stage
        numerical_input_shape = (REGRESSION_LSTM_INPUT_SHAPE[0], REGRESSION_LSTM_INPUT_SHAPE[1] - len(REGRESSION_EMBEDDING_VOCAB_SIZES) + 1)
        model = create_lstm_regression_model(numerical_input_shape=numerical_input_shape)
    model.fit(
        regression_input_train,
        regression_label_train,
        epochs=1000,
        batch_size=16,
        validation_data=(regression_input_val, regression_label_val)
    )

    y_pred = model.predict(regression_input_val)
    # If y_pred shape is (samples, timesteps, 1)
    y_pred_last = y_pred[:, -1].flatten()
    # If regression_label_val shape is (samples, timesteps)
    y_true_last = regression_label_val[:, -1].flatten()

    r2 = r2_score(y_true_last, y_pred_last)
    mae = mean_absolute_error(y_true_last, y_pred_last)
    mse = mean_squared_error(y_true_last, y_pred_last)
    
    print("R^2 Score:", r2)
    print("MAE:", mae)
    print("MSE:", mse)

    return model


def binary_prediction(input_data, labels, binary_model, model_type):
    if model_type == 'lstm':
        data_train = pd.concat([input_data, labels], axis=1)
        input_data, labels = parse_for_embeddings(data_train, task_type="binary")

        y_pred = binary_model.predict(input_data)  # shape (n_samples, 1)
        y_pred_class = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to class labels
        # If y_test shape is (n_samples, 1), flatten it:
        y_true = labels.flatten()
        accuracy = accuracy_score(y_true, y_pred_class)
        roc_auc = roc_auc_score(y_true, y_pred.flatten())
        precision = precision_score(y_true, y_pred_class)
        recall = recall_score(y_true, y_pred_class)
        f1 = f1_score(y_true, y_pred_class)
        print("Accuracy:", accuracy)
        print("ROC AUC:", roc_auc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # ROC values
        fpr, tpr, thresholds = roc_curve(y_true, y_pred.flatten())
        roc_auc = auc(fpr, tpr)
        # plot
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("LSTM ROC Curve")
        plt.legend()
        plt.show()

        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred_class)
        plt.title("LSTM Confusion Matrix")
        plt.show()
        
    else:
        # Predict on the test set
        y_pred = binary_model.predict(input_data)
        y_proba = binary_model.predict_proba(input_data)[:, 1]  # Probability for ROC_AUC

        # Accuracy
        acc = accuracy_score(labels, y_pred)
        print("Test Accuracy:", acc)
        # ROC_AUC Score
        auc_score = roc_auc_score(labels, y_proba)
        print("Test ROC_AUC Score:", auc_score)
        # Precision
        precision = precision_score(labels, y_pred)
        print("Precision:", precision)
        # Recall
        recall = recall_score(labels, y_pred)
        print("Recall:", recall)
        # F1 Score
        f1 = f1_score(labels, y_pred)
        print("F1 Score:", f1)
        # Accuracy score
        print("Test set .score() (accuracy):", binary_model.score(input_data, labels))

        # ROC values
        fpr, tpr, thresholds = roc_curve(labels, y_proba)
        roc_auc = auc(fpr, tpr)
        # plot
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_type.upper()} ROC Curve") # xgboost or mlp
        plt.legend()
        plt.show()

        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(labels, y_pred)
        plt.title(f"{model_type.upper()} Confusion Matrix") # xgboost or mlp
        plt.show()
    
    model_name = ""
    if model_type != 'lstm':
        model_name = get_model_name(binary_model)
    
    plot_feature_importances(binary_model, model_name, model_type, input_data, labels, task_type="binary")

def regression_prediction(input_data, labels, regression_model, model_type, task_type='regression'):
    if model_type == 'lstm':
        data_train = pd.concat([input_data, labels], axis=1)
        input_data, labels = parse_for_embeddings(data_train, task_type=task_type)

        y_pred = regression_model.predict(input_data)
        # If y_pred shape is (samples, timesteps, 1)
        y_pred_last = y_pred[:, -1].flatten()
        # If regression_label_val shape is (samples, timesteps)
        y_true_last = labels[:, -1].flatten()

        r2 = r2_score(y_true_last, y_pred_last)
        mae = mean_absolute_error(y_true_last, y_pred_last)
        mse = mean_squared_error(y_true_last, y_pred_last)
        
        # Score metrics
        print("R^2 Score:", r2)
        print("MAE:", mae)
        print("MSE:", mse)

        # Scatter plot of actual vs predicted values
        plt.scatter(y_true_last, y_pred_last)
        # Plot y=x reference line
        min_val = min(min(y_true_last), min(y_pred_last))
        max_val = max(max(y_true_last), max(y_pred_last))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("LSTM Regression: Actual vs Predicted")
        #plt.title("LSTM Regression with Additional Feature: Actual vs Predicted") # uncomment if using stage 2 regression with the additional feature
        plt.show()

    else:
        # Predict on the test set
        y_pred = regression_model.predict(input_data)

        # Score metrics
        print("R^2 Score:", r2_score(labels, y_pred))
        print("MAE:", mean_absolute_error(labels, y_pred))
        print("MSE:", mean_squared_error(labels, y_pred))

        # Scatter plot of actual vs predicted values
        plt.scatter(labels, y_pred)
        # Plot y=x reference line
        min_val = min(min(labels), min(y_pred))
        max_val = max(max(labels), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_type.upper()} Regression: Actual vs Predicted") # xgboost or mlp
        #plt.title(f"{model_type.upper()} Regression with Additional Feature: Actual vs Predicted") # uncomment if using stage 2 regression with the additional feature
        plt.show()

    model_name = ""
    if model_type != 'lstm':
        model_name = get_model_name(regression_model)
    
    plot_feature_importances(regression_model, model_name, model_type, input_data, labels, task_type="regression")

def full_pipeline_prediction(
        binary_input_data,  binary_labels, 
        regression_input_data, regression_labels, 
        binary_model, binary_model_type, 
        regression_model, regression_model_type,
        is_regression_stage2=False
    ):

    if binary_model_type == 'lstm':
        binary_input_data, binary_labels = parse_for_embeddings(pd.concat([binary_input_data, binary_labels], axis=1), task_type="binary")
    if regression_model_type == 'lstm':
        task_type = "regression2" if is_regression_stage2 else "regression"
        regression_input_data, regression_labels = parse_for_embeddings(pd.concat([regression_input_data, regression_labels], axis=1), task_type=task_type)

    # Look for mismatch in sample sizes
    num_binary_samples = len(binary_labels)
    num_regression_samples = len(regression_labels)

    num_samples = min(num_binary_samples, num_regression_samples)

    ### Stage 1: Predict Disruption ###
    if binary_model_type != 'lstm':
        disruption_pred = binary_model.predict(binary_input_data)[-num_samples:]
    else:
        disruption_raw = (binary_model.predict(binary_input_data) > 0.5).astype(int)
        disruption_pred = disruption_raw[-num_samples:, -1, 0]

    ### Stage 2: Predict Lead Time only for disrupted shipments ###
    if regression_model_type == 'lstm':
        n_samples = next(iter(regression_input_data.values())).shape[0]
        lead_time_pred = np.full(n_samples, np.nan)
    else:
        lead_time_pred = np.full(len(regression_input_data), np.nan)[-num_samples:]

    disrupted_indices = np.where(disruption_pred == 1)[0]

    if regression_model_type == 'lstm':
        # Batch prediction for all disrupted samples
        disrupted_input = {k: v[disrupted_indices] for k, v in regression_input_data.items()}
        y_pred = regression_model.predict(disrupted_input)
        y_pred_last = y_pred[:, -1].flatten() # Flatten to 1D array and take last timestep
        lead_time_pred[disrupted_indices] = y_pred_last
    else:
        lead_time_pred[disrupted_indices] = regression_model.predict(regression_input_data.iloc[disrupted_indices])

    results = pd.DataFrame({
        'Disruption_Prediction': disruption_pred,
        'Lead_Time_Prediction': lead_time_pred
    }, index=pd.RangeIndex(num_samples))

    # Evaluate disruption prediction accuracy
    print("Disruption Prediction Distribution:")
    print(results['Disruption_Prediction'].value_counts())
    if binary_model_type == 'lstm':
        binary_labels_subset = binary_labels[-num_samples:, -1]
        disruption_pred_subset = np.array(disruption_pred).reshape(-1)  # Ensure it's a 1D array
        accuracy = accuracy_score(binary_labels_subset, disruption_pred_subset)
    else:
        accuracy = accuracy_score(binary_labels[-num_samples:], disruption_pred)
    print(f"Disruption Prediction Accuracy: {accuracy:.4f}")

    # Evaluate lead time prediction accuracy only for disrupted samples
    if regression_model_type == "lstm":
        reg_labels_subset = regression_labels[-num_samples:, -1]
        reg_predictions = np.array(lead_time_pred).reshape(-1)  # Ensure it's a 1D array
        mask = ~np.isnan(reg_predictions)
    else:
        reg_labels_subset = pd.Series(regression_labels[-num_samples:]).reset_index(drop=True)
        reg_predictions = pd.Series(lead_time_pred).reset_index(drop=True)
        mask = reg_predictions.notna()
    y_true_valid = reg_labels_subset[mask]
    y_pred_valid = reg_predictions[mask]
    accuracy_lead_time = r2_score(y_true_valid, y_pred_valid)
    print(f"Lead Time Prediction R^2 Score (only disrupted samples): {accuracy_lead_time:.4f}")

    # Score metrics
    print("R^2 Score:", r2_score(y_true_valid, y_pred_valid))
    print("MAE:", mean_absolute_error(y_true_valid, y_pred_valid))
    print("MSE:", mean_squared_error(y_true_valid, y_pred_valid))

    # Scatter plot of actual vs predicted values
    plt.scatter(y_true_valid, y_pred_valid)
    # Plot y=x reference line
    min_val = min(min(y_true_valid), min(y_pred_valid))
    max_val = max(max(y_true_valid), max(y_pred_valid))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Stage 2 {regression_model_type.upper()} Regression: Actual vs Predicted") # xgboost or mlp or lstm
    #plt.title(f"Stage 2 {regression_model_type.upper()} Regression with Additional Feature: Actual vs Predicted") # uncomment if using stage 2 regression with the additional feature
    plt.show()

    results_clean = pd.DataFrame({
        'Disruption_Prediction': disruption_pred,
        'Lead_Time_Prediction': lead_time_pred
    }, index=pd.RangeIndex(num_samples)).dropna(subset=['Lead_Time_Prediction'])

    # Prediction results
    print("\n\nFull Pipeline Prediction Results:")
    print(results)
    print("\n\nFull Pipeline Prediction Results (only disrupted samples with lead time predictions):")
    print(results_clean)

    # Plot feature importances for both models
    # binary_model_name = ""
    # if binary_model_type != 'lstm':
    #     binary_model_name = get_model_name(binary_model)
    # plot_feature_importances(binary_model, binary_model_name, binary_model_type, binary_input_data, binary_labels, task_type="binary")

    # regression_model_name = ""
    # if regression_model_type != 'lstm':
    #     regression_model_name = get_model_name(regression_model)
    # plot_feature_importances(regression_model, regression_model_name, regression_model_type, regression_input_data, regression_labels, task_type="regression")
