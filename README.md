# Custom_train
Train a custom Yolo model version 5, 8, 9 or a DETR using Ultralytics. Alternatively we can connect to an AWS bucket to download the dataset and perform data augmentation if necessary. All experiments can be tracked by CometML if you set your API_KEY inside an `.env`. 

## Instructions
```bash
git clone https://github.com/amulet1989/Custom_train.git
cd Custom_train
```
## Create venv
```bash
python3 -m venv .venv
.venv/bin/activate
```
## Install dependencies
```bash
pip install -r requirements.txt
```
## Create `.env` in the root directory to set environment variables
Inside `.env` put:

`COMET_API_KEY=<your api key>`

And if you will use a AWS bucket

`AWS_ACCESS_KEY = <your AWS public acces key>`

`AWS_SECRET_KEY = <your AWS secret acces key>`

## Run `train_yolov8.py`,  `train_yolov9.py`, `train_yolov5.py` or `train_detr.py`
```bash
python train_yolov9.py --not_download --dataset_dir_path ./dataset --dataset_name your_custom_dataset
```

## Some arguments to use:
```python
    parser.add_argument(
        "--not_download",
        default=True,
        action="store_false",
        help="Si no se desea descargar el dataset",
    )

    parser.add_argument(
        "--not_augment",
        default=True,
        action="store_false",
        help="Si se desea aplicar data aumentación a los datos",
    )

    parser.add_argument(
        "--not_train",
        default=True,
        action="store_false",
        help="Si no se desesa entrenar el modelo",
    )

    parser.add_argument(
        "--data_zip_path",
        default=config.DATA_ZIP_PATH,
        type=str,
        help="Ruta al archivo zip que se descarga de AWS",
    )

    parser.add_argument(
        "--dataset_dir_path",
        default=config.DATASET_DIR_PATH,
        type=str,
        help="Ruta al Dataset de entrenamiento",
    )

    parser.add_argument(
        "--dataset_name",
        default="Merged_Dataset",
        type=str,
        help="Nombre del Dataset de entrenamiento",
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
        help="Ruta al Dataset de entrenamiento aumentado",
    )

    parser.add_argument(
        "--aumented_for",
        default=5,
        type=int,
        help="Veces que se aplicará la aumentación de datos",
    )
    parser.add_argument(
        "--model_link",
        default="https://www.comet.com/api/asset/download...",
        type=str,
        help="Link del modelo a descargar",
    )
    parser.add_argument(
        "--model_path",
        default="",
        type=str,
        help="path del modelo de ultralytics",
    )
    parser.add_argument(
        "--project_name",
        default="pruebas",
        type=str,
        help="Nombre del proyecto en Comet",
    )
```
## Ejemplos de uso
### RUN local
python3 train_yolov8.py --not_download --dataset_dir_path /media/minigo/DATA/Hands_Dataset_dump --dataset_name final_hand_dataset_704x576 --not_augment --model_path "yolov8m.pt"

### Run AWS
- descargar modelo de un link

python3 train_yolov8.py  --not_augment --model_link "https://www.comet.com/api/asset/download?assetId=7c94626e68db40ba82c5a845de851147&experimentKey=7d1a10bcbb9a4e6287464baf9926ff7e" --bucket_name datasetsymodelos --bucket_path CF_Vicente_Lopez_9cam/CF_Vicente_Lopez_9cam_640x480_v5_YOLOV8.zip --dataset_name CF_Vicente_Lopez_9cam_640x480_v5_YOLOV8 --project_name Gestion_fila_Yolov8m_9_cam

- descargar un modleo base de ultralytics

python3 train_yolov8.py  --not_augment --model_path "yolov8m.pt" --bucket_name datasetsymodelos --bucket_path CF_Vicente_Lopez_9cam/CF_Vicente_Lopez_9cam_640x480_v5_YOLOV8.zip --dataset_name CF_Vicente_Lopez_9cam_640x480_v5_YOLOV8 --project_name Gestion_fila_Yolov8m_9_cam


    