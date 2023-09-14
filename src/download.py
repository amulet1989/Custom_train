import boto3
import os
import zipfile

import logging

logging.basicConfig(level=logging.INFO)

s3 = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)


def download_from_s3(bucket_name: str, s3_folder: str, local_dir: str):
    # create local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # download the zip file
    logging.info("Downloading the zip file from S3...")
    s3_file = f"{s3_folder}.zip"
    s3.Object(bucket_name, s3_file).download_file(os.path.join(local_dir, s3_file))
    # unzip the file
    logging.info("Unzipping the zip file ...")
    with zipfile.ZipFile(os.path.join(local_dir, s3_file), "r") as zip_ref:
        zip_ref.extractall(local_dir)
    # delete the zip file
    os.remove(os.path.join(local_dir, s3_file))
    logging.info(f"Downloaded {s3_file} from S3")


# Ejemplo de uso:
download_from_s3("mi_bucket", "mi_carpeta", "mi_directorio_local")


def download_from_local(input_dir: str, output_dir: str):
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # download the zip file
    logging.info(f"Downloading the zip file from {input_dir} to {output_dir}")

    with zipfile.ZipFile(input_dir, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    # os.system(f"unzip {input_dir} -d {output_dir}")

    logging.info(f"Downloaded {input_dir} at {output_dir}")
