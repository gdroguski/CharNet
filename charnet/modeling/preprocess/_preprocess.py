import inspect
import sys
from typing import Dict

import cv2
import numpy as np

from ._abstract_preprocess import AbstractPreprocess

__all__ = [
    'EmptyPreprocess',
    'SamplePreprocess',
    'get_all_preprocesses'
]


class EmptyPreprocess(AbstractPreprocess):
    """
    Empty preprocess made for configs without specified documents' preprocess
    """

    __slots__ = '__p_type'

    def __init__(self):
        """
        Constructor for setting preprocess type to "Empty"
        """
        super().__init__()
        self.__p_type: str = 'Empty'

    @property
    def p_type(self) -> str:
        """
        Property defining internal preprocess type

        Returns
        ----------
        str
            Preprocess type "Empty"
        """
        return self.__p_type

    def run(self, img_path: str) -> np.ndarray:
        """
        Empty Preprocess pipeline

        Parameters
        ----------
        img_path: str
            path to image to be preprocessed

        Returns
        -------
        img: np.ndarray
            loaded image without any preprocessing
        """
        img = cv2.imdecode(
            np.fromfile(f'{img_path}'.encode('utf-8'),
                        np.uint8), cv2.COLOR_RGBA2BGRA)

        return img


class SamplePreprocess(AbstractPreprocess):
    """
    Simple preprocess for de-skewing and frame-deletion
    """
    __slots__ = '__p_type'

    def __init__(self):
        super().__init__()
        self.__p_type: str = 'Sample'

    @property
    def p_type(self) -> str:
        """
        Property defining internal preprocess type

        Returns
        ----------
        str
            Preprocess type "ZUS"
        """
        return self.__p_type

    def run(self, img_path: str) -> np.ndarray:
        """
        SamplePreprocess pipeline:
            1. Load image
            2. Deskew it
            3. Remove horizontal and vertical lines

        Parameters
        ----------
        img_path: str
            path to image to be preprocessed

        Returns
        -------
        img: np.ndarray
            preprocessed image
        """
        img = cv2.imdecode(
            np.fromfile(f'{img_path}'.encode('utf-8'),
                        np.uint8), cv2.COLOR_RGBA2BGRA)

        img = rotate_image(img, detect_angle(img))

        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 2))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(img, [c], -1, (255, 255, 255), 9)

        # remove horizontal lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 25))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(img, [c], -1, (255, 255, 255), 9)

        return img


def get_all_preprocesses() -> Dict:
    """
    Function returning all defined here preprocesses for images within the dictionary: Dict[p_type, cls_instance]

    Returns
    -------
    preprocesses: Dict
        dictionary with preprocess type name and class instance
    """
    preprocesses: Dict = {}

    cls_members = inspect.getmembers(sys.modules[__name__], __is_cls_here)
    for cls in cls_members:
        cls_inst = cls[1]()
        preprocesses.update({cls_inst.p_type: cls_inst})

    return preprocesses


def __is_cls_here(cls) -> bool:
    """
    Lambda wrapper for checking whether the specified class is within this particular file/module

    Parameters
    ----------
    cls
        class reference

    Returns
    -------
    bool:
        whether the class is within this particular file/module
    """
    return inspect.isclass(cls) and cls.__module__ == __is_cls_here.__module__


def detect_angle(image: np.ndarray) -> float:
    """
    Method for detecting angle of the skewed document

    Parameters
    ----------
    image: np.ndarray
        image to be analyzed

    Returns
    -------
    angle: float
        angle of skewing of this image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_contour = contours[0]
    min_area_rect = cv2.minAreaRect(largest_contour)

    angle = min_area_rect[-1]
    if angle < -45:
        angle = 90 + angle

    return angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate given image by specified angle

    Parameters
    ----------
    image: np.ndarray
        image to be rotated
    angle
        how much to rotate the image

    Returns
    -------
    rotated_img: np.ndarray
        rotated image
    """
    rotated_img: np.ndarray = image.copy()
    (h, w) = rotated_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img: np.ndarray = cv2.warpAffine(
        rotated_img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE)

    return rotated_img
