"""
This module provide functionality for managing a ML model.

It contains the ModelService class, which handles loading and using
a pre-trained ML model. The class offers methods to load a model
from a file, building it if it doesn't exist, and to make predictions
using the loaded model.
"""

import pickle as pk
from pathlib import Path

from loguru import logger

from config import model_settings


class ModelInferenceService:
    """
    A service class for making predicitions.

    This class provides functionalities to load a ML model from
    a specified path, build it if it doesn't exist, and make
    predictions using the loaded model

    Attributes:
        model: ML model managed by this service. Initially set to None.
        model_path: Directort to extract the model from.
        model_name: Name of the saved model to use.

    Methods:
        __init__: Constructor that initializes the ModelService.
        load_model: Loads the model from file or build it if it doesn't exist.
        predict: Makes a prediction using the loaded model.
    """

    def __init__(self) -> None:
        """Initializes the ModelInferenceService."""
        self.model = None
        self.model_path = model_settings.model_path
        self.model_name = model_settings.model_name

    def load_model(self) -> None:
        """
        Loads the model from a specified path

        Raises:
            FileNotFoundError: If the model file does
            not exist at specified dir.
        """
        logger.info(f'checking the existance of model config file at '
                    f'{self.model_path}/{self.model_name}',
                    )
        model_path = Path(
                f'{self.model_path}/{self.model_name}'
                )

        if not model_path.exists():
            raise FileNotFoundError("Model file does not exist!")

        logger.info(
                f'model {self.model_name} exists -> '
                'loading model configuration file',
                )

        with open(model_path, 'rb') as file:
            self.model = pk.load(file)

    def predict(self, input_parameters: list) -> list:
        """"
        Makes a predicition using the loaded model.

        Takes the input parameters and passes it to the model, which
        was loaded using a pickle file

        Args:
            input_parameters (list): The input data for making a prediction.

        Returns:
            list: The prediction result from the model.
        """
        logger.info('making prediction')
        return self.model.predict([input_parameters])
