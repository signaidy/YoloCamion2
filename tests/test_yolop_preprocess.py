import cv2
import numpy as np

from src.percepcion.yolop_inference import InferenciaYOLOP


def test_clahe_recupera_contraste_de_marcas_de_carril():
    img = np.full((160, 320, 3), 128, dtype=np.uint8)
    mask_lineas = np.zeros((160, 320), dtype=np.uint8)

    cv2.line(img, (90, 0), (110, 159), (145, 145, 145), 4)
    cv2.line(mask_lineas, (90, 0), (110, 159), 255, 4)
    cv2.line(img, (210, 0), (190, 159), (145, 145, 145), 4)
    cv2.line(mask_lineas, (210, 0), (190, 159), 255, 4)

    inferencia = InferenciaYOLOP(device="cpu")
    realzada = inferencia._realzar_contraste_local(img)

    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_realzada = cv2.cvtColor(realzada, cv2.COLOR_BGR2GRAY)
    fondo = mask_lineas == 0
    lineas = mask_lineas > 0

    delta_original = gray_original[lineas].mean() - gray_original[fondo].mean()
    delta_realzada = gray_realzada[lineas].mean() - gray_realzada[fondo].mean()

    assert delta_realzada > delta_original


def test_preprocesar_guarda_debug_letterbox_con_contraste_local():
    img = np.full((108, 192, 3), 128, dtype=np.uint8)
    img[:, 80:84] = (145, 145, 145)

    inferencia = InferenciaYOLOP(imgsz=64, device="cpu")
    tensor, _, _, shape_original = inferencia.preprocesar(img)

    assert shape_original == (108, 192)
    assert tuple(tensor.shape) == (1, 3, 64, 64)
    assert inferencia.ultima_imagen_debug is not None
    assert inferencia.ultima_imagen_debug.shape == (64, 64, 3)
