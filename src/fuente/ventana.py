"""Captura el contenido de una ventana específica por título usando Win32 API.

Funciona incluso cuando la ventana está detrás de otras (terminal, etc.)
porque usa PrintWindow que accede al buffer de la aplicación, no a la pantalla.

Requisito: ETS2 en modo Ventana sin bordes (Borderless Windowed).
"""
import ctypes
import ctypes.wintypes
import logging
import time
from typing import Optional

import cv2
import numpy as np

from src.tipos import Cuadro
from src.fuente.base import FuenteCuadros

logger = logging.getLogger(__name__)

# Flags de PrintWindow
_PW_CLIENTONLY         = 0x1
_PW_RENDERFULLCONTENT  = 0x2  # necesario para apps DirectX en Windows 10+


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize",          ctypes.c_uint32),
        ("biWidth",         ctypes.c_int32),
        ("biHeight",        ctypes.c_int32),
        ("biPlanes",        ctypes.c_uint16),
        ("biBitCount",      ctypes.c_uint16),
        ("biCompression",   ctypes.c_uint32),
        ("biSizeImage",     ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed",       ctypes.c_uint32),
        ("biClrImportant",  ctypes.c_uint32),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 3),
    ]


def _capturar_hwnd(hwnd: int, ancho: int, alto: int) -> Optional[np.ndarray]:
    user32 = ctypes.windll.user32
    gdi32  = ctypes.windll.gdi32

    screen_dc = user32.GetDC(hwnd)
    mem_dc    = gdi32.CreateCompatibleDC(screen_dc)
    hbmp      = gdi32.CreateCompatibleBitmap(screen_dc, ancho, alto)
    gdi32.SelectObject(mem_dc, hbmp)

    # PrintWindow captura el contenido aunque la ventana esté tapada
    ok = user32.PrintWindow(hwnd, mem_dc,
                            _PW_CLIENTONLY | _PW_RENDERFULLCONTENT)

    if not ok:
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, screen_dc)
        return None

    bmp_info = _BITMAPINFO()
    bmp_info.bmiHeader.biSize        = ctypes.sizeof(_BITMAPINFOHEADER)
    bmp_info.bmiHeader.biWidth       = ancho
    bmp_info.bmiHeader.biHeight      = -alto   # negativo = top-down
    bmp_info.bmiHeader.biPlanes      = 1
    bmp_info.bmiHeader.biBitCount    = 32
    bmp_info.bmiHeader.biCompression = 0       # BI_RGB

    buf = (ctypes.c_byte * (ancho * alto * 4))()
    gdi32.GetDIBits(mem_dc, hbmp, 0, alto, buf, ctypes.byref(bmp_info), 0)

    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, screen_dc)

    frame = np.frombuffer(buf, dtype=np.uint8).reshape(alto, ancho, 4)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def buscar_ventana(titulo_parcial: str) -> Optional[int]:
    """Busca una ventana cuyo título contenga el texto dado."""
    encontrados: list[int] = []

    def _callback(hwnd, _):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
            if titulo_parcial.lower() in buf.value.lower():
                encontrados.append(hwnd)
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    ctypes.windll.user32.EnumWindows(WNDENUMPROC(_callback), 0)
    return encontrados[0] if encontrados else None


class FuenteVentana(FuenteCuadros):
    """Captura una ventana de Windows por título, sin importar si está oculta.

    Ideal para capturar ETS2 con un solo monitor mientras la terminal
    está en primer plano.
    """

    def __init__(
        self,
        titulo: str = "Euro Truck Simulator 2",
        escalar_a: Optional[tuple[int, int]] = (1920, 1080),
    ):
        self._titulo = titulo
        self._escalar_a = escalar_a
        self._hwnd: Optional[int] = None
        self._ancho = 0
        self._alto  = 0
        self._indice = 0
        self._t_inicio = 0.0
        self._activa = False

    def iniciar(self) -> None:
        self._hwnd = buscar_ventana(self._titulo)
        if not self._hwnd:
            raise RuntimeError(
                f"No se encontró ventana con título '{self._titulo}'. "
                "Asegúrate de que ETS2 está abierto en modo Ventana sin bordes."
            )

        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(self._hwnd, ctypes.byref(rect))
        self._ancho = rect.right
        self._alto  = rect.bottom

        if self._ancho <= 0 or self._alto <= 0:
            raise RuntimeError(
                f"Ventana encontrada (hwnd={self._hwnd}) pero tamaño inválido "
                f"({self._ancho}x{self._alto}). ¿Está minimizada?"
            )

        self._indice   = 0
        self._t_inicio = time.monotonic()
        self._activa   = True
        logger.info(
            "FuenteVentana: '%s' hwnd=%d tamaño=%dx%d",
            self._titulo, self._hwnd, self._ancho, self._alto
        )

    def siguiente(self) -> Optional[Cuadro]:
        if not self._activa or self._hwnd is None:
            return None

        frame = _capturar_hwnd(self._hwnd, self._ancho, self._alto)
        if frame is None:
            return None

        if self._escalar_a is not None:
            w, h = self._escalar_a
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        ahora   = time.monotonic()
        elapsed = ahora - self._t_inicio
        fps     = self._indice / elapsed if elapsed > 0 else 0.0
        self._indice += 1

        return Cuadro(imagen=frame, timestamp=ahora,
                      indice=self._indice, fps_instantaneo=fps)

    def cerrar(self) -> None:
        self._activa = False

    @property
    def esta_activa(self) -> bool:
        return self._activa
