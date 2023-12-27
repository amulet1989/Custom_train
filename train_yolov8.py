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
        "--dataset_name",
        default="Merged_Dataset",
        type=str,
        help="Nombre del Datset de entrenamiento",
    )

    parser.add_argument(
        "--bucket_name",
        default=config.AWS_BUCKET,
        type=str,
        help="Nombre del bucket en AWS",
    )

    parser.add_argument(
        "--bucket_path",
        default=config.AWS_FILE_PATH,
        type=str,
        help="Path del dataset dentro del bucket en AWS",
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
        download.download_from_s3(
            bucket_name=args.bucket_name, bucket_path=args.bucket_path
        )

        # Augment data
        if args.not_augment:
            print("augmenting data")
            data_augmentation.augment_dataset(
                os.path.join(args.dataset_dir_path, args.dataset_name),
                args.augmented_dir_path,
                augmented_for=args.aumented_for,
            )

    # Train the model
    # Read API key from the .environment variables
    if args.not_train:
        api_key = os.environ.get("COMET_API_KEY")

        comet_ml.init(api_key=api_key)

        model = YOLO(
            "trained/yolov8m_cf_4cam_verano_pies_v2.pt"
        )  # yolov8m trained/yolov8m_6cam_augm - yolov8n_6cam

        model.train(
            data=os.path.join(
                args.dataset_dir_path,
                args.dataset_name,
                "data.yaml",
            ),  # Merged_Dataset/Augmented_Dataset - "Gestion_de_filas_4_camaras_v1i_yolov8",
            cfg="cfgs/cfg_y8s.yaml",
            project="Gestion_fila_Yolov8m_4_cam",  # Gestion_fila_Yolov8m_4_cam - Gestion_fila_Yolov8n_4_cam -CF_Pilar_tracking_Yolov8m_11_cam
            name="Yolov8m_",  # Yolov8m_ - Yolov8n_
        )


if __name__ == "__main__":
    main()
