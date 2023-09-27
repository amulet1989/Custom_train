import comet_ml
import wandb
from ultralytics import YOLO
import argparse
from src import config, data_augmentation, download
import os
from ray import tune


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline para procesar videos y detección de objetos"
    )

    parser.add_argument(
        "--not_download",
        default=True,
        action="store_false",
        help="Si se desea descargar el dataset",
    )

    parser.add_argument(
        "--not_augment",
        default=True,
        action="store_false",
        help="Si se desea aplicar las transformaciones a los datos",
    )

    parser.add_argument(
        "--not_train",
        default=True,
        action="store_false",
        help="Si no desesa entrenar el modelo",
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
    if args.not_download:
        # download.download_from_local(args.data_zip_path, args.dataset_dir_path)
        download.download_from_s3()

        # Augment data
        if args.not_augment:
            print("augmenting data")
            data_augmentation.augment_dataset(
                os.path.join(args.dataset_dir_path, "Merged_Dataset"),
                args.augmented_dir_path,
                augmented_for=args.aumented_for,
            )

    # Train the model
    # Read API key from the .environment variables
    if args.not_train:
        # api_key = os.environ.get("COMET_API_KEY")
        # comet_ml.init(api_key=api_key)
        # Cargar la clave API desde una variable de entorno

        api_key = os.environ.get("WANDB_API_KEY")
        # Iniciar sesión en W&B
        wandb.login(key=api_key)

        model = YOLO("yolov8s.pt")

        result_grid = model.tune(
            data=os.path.join(args.dataset_dir_path, "Augmented_Dataset", "data.yaml"),
            project="YoloV8m_gest_fila",
            use_ray=True,
            iterations=10,
            epochs=50,
        )


if __name__ == "__main__":
    main()
