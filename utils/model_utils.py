import os
from gdown import download as drive_download

google_drive_paths = {
    'shape_predictor_68_face_landmarks.dat': 'https://drive.google.com/uc?id=17kwWXLN9fA6acrBWqfuQCBdcc1ULmBc9',
    'e4e_ffhq_encode.pt': 'https://drive.google.com/uc?id=1O8ipkyMYHwCRmuaZBaO-KYZ9FYuH8Xnc',
    'ffhq.pt': 'https://drive.google.com/uc?id=1H9-wdGHMu0E6H-QXOmRXLZVy_E_D0mf0',
    'titan_erwin.pt': 'https://drive.google.com/uc?id=1AvSrWkIxgoxXtjPuogKiP45BkKJvFQbI',
    'titan_armin.pt': 'https://drive.google.com/uc?id=1o9yhTmW8voeCi6dNrOY3sSVQqMawftRB',
    'titan_historia.pt': 'https://drive.google.com/uc?id=1MqMmdcCGXutoDV8wxP31K2iyvA4SgJx7',
    'pocahontas.pt': 'https://drive.google.com/uc?id=1jRcWh7lQ-28abiSOVBVUi_iBDGPj7Esl',
    'moana.pt': 'https://drive.google.com/uc?id=19kjijHa_G2B3UrNGHXYVe5izaNdi8ABr',
    'doc_brown.pt': 'https://drive.google.com/uc?id=1fQJYUE9a9DSoRupOxllslA-dIm-8-D72',
    'brave.pt': 'https://drive.google.com/uc?id=1wD3xoGgrmbN74npUAlrmkk97_DXF6NWR',
    'sketch.pt': 'https://drive.google.com/uc?id=1YbFyukh6n9l6UFtsqOESbhwGH7BupJtb',
    'jojo.pt': 'https://drive.google.com/uc?id=1VLLmh7f-vcS2MB3CXET1lFRC-_Sq-p8J',
    'detroit.pt': 'https://drive.google.com/uc?id=1cYSX9oLkhv6vosIAKnEQZoDFQS5siT-u',
    'picasso.pt': 'https://drive.google.com/uc?id=1C7pCKIFdqFFrK9diFZKGLXrYlmrPP2Ui',
    'anastasia.pt': 'https://drive.google.com/uc?id=1iXxaxKG0EJ_C1Jr5QrBogq9yvMzyhcth',
    'room_girl.pt': 'https://drive.google.com/uc?id=16F1oCrv8UNnhlqFFUwJy49OpW5d-kA-m',
    'speed_paint.pt': 'https://drive.google.com/uc?id=1uB2uQnAF8pghXNlTnGRTduiwpAtcDiqo',
    'digital_painting_jing.pt': 'https://drive.google.com/uc?id=168bfp7FvN_VF1pOT7uEYzhtj2oFGIsQ_',
    'mermaid.pt': 'https://drive.google.com/uc?id=1LO3UdMHPKfwjaaxgVgBbLquei8n8q5P8',
    'zbrush_girl.pt': 'https://drive.google.com/uc?id=1YPQSDW-_utOEu5A9nbq2832jTTJK2ECb',
    'joker.pt': 'https://drive.google.com/uc?id=1Ptv-EjYAKngxpf9lY5cpypAmyoX863Nh',
}


def download_weight(weight_path):
    if not os.path.isfile(weight_path) and (
            os.path.basename(weight_path) in google_drive_paths
    ):

        gdrive_url = google_drive_paths[os.path.basename(weight_path)]
        try:
            # drive_download(gdrive_url, weight_path, fuzzy=True)
            drive_download(gdrive_url, weight_path, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

