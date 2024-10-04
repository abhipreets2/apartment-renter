"""
This module creates the pipeline for building, training and saving ML model.

It includes the process of data preparation, model training using
RandomForestRegressor, hyperparameter tuning with GridSearchCV,
model evaluation, and serialization of the trained model.
"""

import pickle as pk

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from config import model_settings
from model.pipeline.preparation import prepare_data


def build_model() -> None:
    """
    Build, evaluate and save a RandomForestRegressor model.

    This function orchestrates the model building pipeline.
    It starts by preparing the data, followed by defining feature names
    and splitting the dataset into features and target varibles.
    The dataset is then divided into training and testing sets.
    The model's performance is evaluated on the test set, and
    finally, the model is saved for future use.

    Return:
        None
    """
    logger.info('starting up model building pipeline')
    # 1. load preprocessed dataset
    df = prepare_data()
    # 2. identity X and y
    feature_names = [
            'area',
            'construction_year',
            'bedrooms', 'garden',
            'balcony_yes',
            'parking_yes',
            'furnished_yes',
            'garage_yes',
            'storage_yes'
    ]
    X, y = _get_X_y(
            df,
            col_X=feature_names
    )
    # 3. split the dataset
    X_train, X_test, y_train, y_test = _split_train_test(
            X,
            y
    )
    # 4. train the model
    rf = _train_model(
            X_train,
            y_train
    )
    # 5. save the model in a configuration file
    _save_model(rf)


def _get_X_y(data: pd.DataFrame,
             col_X: list,
             col_y: str = 'rent'
             ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits the dataset into features and target variable.

    Args:
        data (pd.DataFrame): The dataset to be split.
        col_X (list[str]): List of column names for features.
        col_y (str): Name of the target variable column.
    """
    logger.info('defining X and y variables.\n'
                f'X vars: {col_X}\ny var: {col_y}')
    return data[col_X], data[col_y]


def _split_train_test(
        X: pd.DataFrame,
        y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variables.

    Returns:
        tuple: Training and testing sets for features and target.
    """
    logger.info('splitting data into train and test sets')
    return train_test_split(
            X,
            y,
            test_size=0.2,
    )


def _train_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> RandomForestRegressor:
    """
    Train the RandomForestRegressor model with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training set features.
        y_train (pd.Series): Training set target.

    Returns:
        RandomForestRegressor: The best estimator after GridSearch.
    """
    logger.info('training a model with hyperparameters')
    grid_space = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9, 12]
    }

    logger.debug(f'grid_space = {grid_space}')
    grid = GridSearchCV(
            RandomForestRegressor(),
            param_grid=grid_space,
            cv=5,
            scoring='r2'
    )

    model_grid = grid.fit(
            X_train,
            y_train
    )
    return model_grid.best_estimator_


def _evaluate_model(
        model: RandomForestRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
) -> float:
    """
    Evaluate the trained model's performance.

    Args:
        model (RandomForestRegressor): The trained model.
        X_test (pd.DataFrame): Testing set features.
        y_test (pd.Series): Testing set target.

    Returns:
        float: The model's score.
    """
    model_score = model.score(
            X_test,
            y_test
    )
    logger.info(f'evaluating model performance, SCORE={model_score}')
    return model_score


def _save_model(model):
    """
    Save the trained model to a specified directory/

    Args:
        model (RandomForestRegressor): The model to save.

    Returns:
        None
    """
    model_path = f'{model_settings.model_path}/{model_settings.model_name}'
    logger.info(f'saving model to a directory: {model_path}')
    with open(model_path, 'wb') as model_file:
        pk.dump(
                model,
                model_file
                )
