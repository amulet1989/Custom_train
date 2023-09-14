import comet_ml
from ultralytics import YOLO
import argparse
from src import config, data_augmentation, download
import os


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline para procesar videos y detección de objetos"
    )

    parser.add_argument(
        "--data_zip_path",
        default=config.DATA_ZIP_PATH,
        type=str,
        help="Ruta al archivo zip",
    )

    parser.add_argument(
        "--dataset_dir_path",
        default=config.DATASET_DIR_PATH,
        type=str,
        help="Ruta al Dataser de entrenamiento",
    )

    parser.add_argument(
        "--augmented_dir_path",
        default=config.DATA_AUMENTED_DIR_PATH,
        type=str,
        help="Ruta al Dataset de entrenamiento",
    )

    parser.add_argument(
        "--aumented_for",
        default=5,
        type=int,
        help="Veces que se aplicaran las transformaciones",
    )

    args = parser.parse_args()

    # Download datset
    print("downloading datset")
    download.download_from_local(args.data_zip_path, args.dataset_dir_path)

    # Augment data
    print("augmenting data")
    data_augmentation.augment_dataset(
        os.path.join(args.dataset_dir_path, "Merged_Dataset"),
        args.augmented_dir_path,
        augmented_for=args.aumented_for,
    )

    # Train the model
    # Read API key from the .environment variables
    api_key = os.environ.get("COMET_API_KEY")

    comet_ml.init(api_key=api_key)

    model = YOLO("yolov5su.pt")
    model.train(
        data=config.DATA_YAML_PATH,
        epochs=3,
        batch=-1,
        project="MiniGO",
        name="YoloV5su_1",
    )


if __name__ == "__main__":
    main()
