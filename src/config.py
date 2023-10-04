import os
import pathlib

HOME = str(pathlib.Path(__file__).parent.parent)

DATASET_DIR_PATH = os.path.join(HOME, "dataset")

DATA_MERGED_DIR_PATH = os.path.join(DATASET_DIR_PATH, "Merged_Dataset")  # No

DATA_AUMENTED_DIR_PATH = os.path.join(DATASET_DIR_PATH, "Augmented_Dataset")

DATA_YAML_PATH = os.path.join(DATASET_DIR_PATH, "Augmented_Dataset", "data.yaml")

DATA_ZIP_PATH = os.path.join(DATASET_DIR_PATH, "data.zip")

AWS_BUCKET = "probando-entrenar-una-red-en-aws"

AWS_FILE_PATH = "Train_Datasets_MiniGO_products/Merged_Dataset.zip"
