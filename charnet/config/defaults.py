from yacs.config import CfgNode as CN


_C = CN()

_C.INPUT_SIZE = 2280
_C.SIZE_DIVISIBILITY = 1
_C.WEIGHT = ""

_C.CHAR_DICT_FILE = ""
_C.WORD_LEXICON_PATH = ""
_C.USE_CPP = False
_C.PARSE_CHARS = False
_C.DLL_PATH = ""
_C.PREPROCESS_TYPE = "Empty"

_C.WORD_MIN_SCORE = 0.95
_C.WORD_NMS_IOU_THRESH = 0.15
_C.CHAR_MIN_SCORE = 0.01
_C.CHAR_NMS_IOU_THRESH = 0.3
_C.MAGNITUDE_THRESH = 0.2

_C.WORD_STRIDE = 4
_C.CHAR_STRIDE = 4
_C.NUM_CHAR_CLASSES = 68

_C.WORD_DETECTOR_DILATION = 1
_C.RESULTS_SEPARATOR = chr(31)
