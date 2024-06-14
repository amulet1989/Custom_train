from boto3.session import Session
import os
import zipfile
import yaml
from src import config
from dotenv import load_dotenv
import requests

import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

session = Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)


def update_data_yaml(data_yaml_path, train_dir, val_dir):
    # Leer el archivo data.yaml
    with open(data_yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Actualizar las rutas train y val
    data["train"] = train_dir
    data["val"] = val_dir

    # Escribir el archivo data.yaml con las rutas actualizadas
    with open(data_yaml_path, "w") as file:
        yaml.dump(data, file)


def download_from_s3(
    bucket_name=config.AWS_BUCKET,
    bucket_path=config.AWS_FILE_PATH,
    local_dir=config.DATASET_DIR_PATH,
):
    # create local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # download the zip file
    logging.info("Downloading the zip file from S3...")
    s3_file = bucket_path.split("/")[-1]
    s3 = session.resource("s3")
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(bucket_path, os.path.join(local_dir, s3_file))

    # unzip the file
    logging.info("Unzipping the zip file ...")
    with zipfile.ZipFile(os.path.join(local_dir, s3_file), "r") as zip_ref:
        zip_ref.extractall(local_dir)
    # delete the zip file
    os.remove(os.path.join(local_dir, s3_file))

    # updated train and val directories
    # get unzipped folder name
    folder_name = os.listdir(local_dir)[0]
    update_data_yaml(
        os.path.join(local_dir, folder_name, "data.yaml"),
        os.path.join(local_dir, folder_name, "train"),
        os.path.join(local_dir, folder_name, "valid"),
    )
    logging.info(f"Downloaded {s3_file} from S3")


def download_from_local(input_dir: str, output_dir: str):
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # download the zip file
    logging.info(f"Downloading the zip file from {input_dir} to {output_dir}")

    with zipfile.ZipFile(input_dir, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    # os.system(f"unzip {input_dir} -d {output_dir}")
    # updated train and val directories
    # get unzipped folder name
    folder_name = os.listdir(output_dir)[0]
    update_data_yaml(
        os.path.join(output_dir, folder_name, "data.yaml"),
        os.path.join(output_dir, folder_name, "train"),
        os.path.join(output_dir, folder_name, "valid"),
    )

    logging.info(f"Downloaded {input_dir} at {output_dir}")


def download_model(model_link: str, output_dir: str = config.MODEL_DIR_PATH):
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define the output path
    output_path = os.path.join(output_dir, "model.pt")

    # download the model
    logging.info(f"Downloading the model from {model_link} to {output_path}")

    try:
        response = requests.get(model_link, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logging.info(f"Downloaded model from {model_link} to {output_path}")
    except requests.RequestException as e:
        logging.error(f"Failed to download the model: {e}")
        return None

    return output_path
