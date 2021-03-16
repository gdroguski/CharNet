from abc import ABC, abstractmethod

import numpy as np


class AbstractPreprocess(ABC):
    """
    Abstract class for working with defined preprocess instance given specified document
    """

    @abstractmethod
    def __init__(self):
        """
        Constructor in which the internal preprocess type must be defined
        """
        pass

    @property
    @abstractmethod
    def p_type(self) -> str:
        """
        Property defining internal preprocess type

        Returns
        ----------
        str
            Preprocess type
        """
        pass

    @abstractmethod
    def run(self, img_path: str) -> np.ndarray:
        """
        Abstract method to be implemented running a preprocess

        Parameters
        ----------
        img_path: str
            path to preprocessed-to-be image

        Returns
        ----------
        np.ndarray
            preprocessed image
        """
        pass
