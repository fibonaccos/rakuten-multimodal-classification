import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config


MODELS_CONFIG = get_config("MODELS")


import os
from cnn_utils import make_CNN_dataset


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


make_CNN_dataset(df_train=MODELS_CONFIG["PATHS"]["cleanTrainLabels"],
                 df_test=MODELS_CONFIG["PATHS"]["cleanTestLabels"],
                 src_train=MODELS_CONFIG["PATHS"]["cleanImageTrainFolder"],
                 src_test=MODELS_CONFIG["PATHS"]["cleanImageTestFolder"],
                 dst_root=MODELS_CONFIG["CNN"]["DATASET"]["folderPath"],
                 max_workers=MODELS_CONFIG["CNN"]["numThreads"])
