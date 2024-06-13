import os
import pathlib

HOME = str(pathlib.Path(__file__).parent.parent)

DATASET_DIR_PATH = os.path.join(HOME, "dataset")

DATA_MERGED_DIR_PATH = os.path.join(DATASET_DIR_PATH, "Merged_Dataset")  # No

DATA_AUMENTED_DIR_PATH = os.path.join(DATASET_DIR_PATH, "Augmented_Dataset")

DATA_YAML_PATH = os.path.join(DATASET_DIR_PATH, "Augmented_Dataset", "data.yaml")

DATA_ZIP_PATH = os.path.join(DATASET_DIR_PATH, "data.zip")

AWS_BUCKET = "datasetsymodelos"

AWS_FILE_PATH = "Pilar_productos_en_mano/Pilar_productos_en_mano_YOLOV8_704x576.zip"
