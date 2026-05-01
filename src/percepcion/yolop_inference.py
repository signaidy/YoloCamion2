import cv2
import numpy as np
import torch
import torchvision
from typing import Tuple, List
from src.tipos import Clase, Deteccion

# Mapeo de YOLOP (COCO) a clases propias del proyecto
_COCO_A_CLASE: dict[int, Clase] = {
    0: Clase.PEATON,
    2: Clase.VEHICULO,      # car
    3: Clase.MOTOCICLETA,
    5: Clase.VEHICULO,      # bus
    7: Clase.VEHICULO,      # truck
    9: Clase.SEMAFORO,
    11: Clase.SENAL_ALTO,
}

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Realiza NMS sobre las inferencias de YOLOP.
    prediction: [batch, num_anchors, 85]
    """
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        # Filtro de confianza
        x = x[x[:, 4] > conf_thres]
        if not x.shape[0]:
            continue

        # Calcular cajas de [x_center, y_center, w, h] a [x1, y1, x2, y2]
        box = xywh2xyxy(x[:, :4])

        # Encontrar la mejor clase
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if not x.shape[0]:
            continue

        # Batched NMS
        c = x[:, 5:6] * 4096  # offset por clase
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]

    return output

class InferenciaYOLOP:
    """Carga YOLOP (Panoptic Driving Perception) y procesa objetos, area manejable y lineas."""

    def __init__(self, conf_min: float = 0.35, imgsz: int = 640, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_min = conf_min
        self.imgsz = imgsz
        self.modelo = None
        self._ultima_imagen_debug: np.ndarray | None = None  # BGR uint8, antes de normalizar
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def cargar(self):
        """Carga el modelo YOLOP preentrenado usando PyTorch Hub."""
        print("[YOLOP] Cargando modelo Panóptico (puede tardar en la primera ejecución)...")
        import warnings
        warnings.filterwarnings("ignore", message="You are about to download and run code from an untrusted repository")

        self.modelo = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)
        self.modelo.to(self.device)
        self.modelo.eval()
        print("[YOLOP] Modelo cargado exitosamente en", self.device)

    @property
    def ultima_imagen_debug(self) -> np.ndarray | None:
        """Último frame tal como entró al modelo: BGR uint8, letterboxed, CLAHE + sharpened.
        None hasta la primera inferencia. Útil para visualizar qué ve YOLOP."""
        return self._ultima_imagen_debug

    def _letterbox_img(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int, int, int]:
        """Resize preservando aspect ratio y rellena con gris (114,114,114) hasta cuadrado.

        YOLOP fue entrenado con letterbox, no con resize directo squash. Usar resize
        squash (1920x1080 → 640x640) distorsiona los ángulos de las líneas de carril
        ~1.5x verticalmente, confundiendo al modelo en curvas.

        Returns: (img_letterboxed, scale, pad_left, pad_top, new_w, new_h)
        """
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        new_w, new_h = round(w * r), round(h * r)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_l = (self.imgsz - new_w) // 2
        pad_t = (self.imgsz - new_h) // 2
        pad_r = self.imgsz - new_w - pad_l
        pad_b = self.imgsz - new_h - pad_t
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, pad_l, pad_t, new_w, new_h

    def _realzar_contraste_local(self, img: np.ndarray) -> np.ndarray:
        """Aplica CLAHE solo en luminancia para recuperar marcas sin alterar color."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = self._clahe.apply(l_chan)
        lab = cv2.merge((l_chan, a_chan, b_chan))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def preprocesar(self, imagen: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int, int, int], Tuple[int, int]]:
        """Prepara la imagen para el modelo con CLAHE + letterbox + sharpening suave.

        Returns:
            img_tensor: tensor normalizado (1, 3, imgsz, imgsz)
            r:          factor de escala uniforme aplicado antes del padding
            (pad_l, pad_t, new_w, new_h): geometría del letterbox para deshacer en postproceso
            shape_original: (H, W) del frame original
        """
        shape_original = imagen.shape[:2]  # (H, W)

        img = self._realzar_contraste_local(imagen)

        # Unsharp mask leve: amplifica líneas de carril finas antes del downscale ~2x.
        # Pesos 1.2/-0.2 evitan clipping en zonas brillantes (asfalto claro, sol).
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = cv2.addWeighted(img, 1.2, blurred, -0.2, 0)

        # Letterbox: preserva aspect ratio 16:9 → 864x486 centrado en 864x864
        img, r, pad_l, pad_t, new_w, new_h = self._letterbox_img(img)

        # Guardar copia BGR uint8 para debug — exactamente lo que entra al modelo
        # (antes de normalizar a float32 con media/std de ImageNet)
        self._ultima_imagen_debug = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).float().to(self.device)
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Normalización ImageNet requerida por YOLOP
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, r, (pad_l, pad_t, new_w, new_h), shape_original

    def procesar_frame(self, imagen: np.ndarray) -> Tuple[list[Deteccion], np.ndarray, np.ndarray]:
        """
        Infiere todo en una sola pasada.
        Retorna:
            - detecciones: Lista de objetos (Deteccion)
            - area_manejable: Mascara binaria (shape_original)
            - lineas_carril: Mascara binaria (shape_original)
        """
        if self.modelo is None:
            raise RuntimeError("Llama a cargar() antes de procesar_frame()")

        img_tensor, r, (pad_l, pad_t, new_w, new_h), shape_orig = self.preprocesar(imagen)

        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.modelo(img_tensor)

        # 1. Postprocesar Detecciones (Bounding Boxes)
        inf_out, _ = det_out  # YOLOP retorna (inference_out, training_out)
        pred = non_max_suppression(inf_out, conf_thres=self.conf_min, iou_thres=0.45)

        detecciones: list[Deteccion] = []
        if len(pred) > 0 and pred[0] is not None:
            for *xyxy, conf, cls in pred[0]:
                id_coco = int(cls)
                clase_proyect = _COCO_A_CLASE.get(id_coco, Clase.DESCONOCIDO)

                # Deshacer letterbox: restar padding y dividir por escala
                x1 = int((float(xyxy[0]) - pad_l) / r)
                y1 = int((float(xyxy[1]) - pad_t) / r)
                x2 = int((float(xyxy[2]) - pad_l) / r)
                y2 = int((float(xyxy[3]) - pad_t) / r)

                x1 = max(0, min(x1, shape_orig[1]))
                y1 = max(0, min(y1, shape_orig[0]))
                x2 = max(0, min(x2, shape_orig[1]))
                y2 = max(0, min(y2, shape_orig[0]))

                area = (x2 - x1) * (y2 - y1)

                if area > 0:
                    detecciones.append(
                        Deteccion(
                            clase=clase_proyect,
                            caja=(x1, y1, x2, y2),
                            confianza=float(conf),
                            area=area,
                        )
                    )

        # 2. Postprocesar Area Manejable — recortar padding antes de upscale
        da_predict = da_seg_out[:, 1, :, :] > da_seg_out[:, 0, :, :]
        da_seg_mask = da_predict.byte().cpu().numpy()[0]
        da_seg_mask = da_seg_mask[pad_t:pad_t + new_h, pad_l:pad_l + new_w]
        da_seg_mask = cv2.resize(da_seg_mask, (shape_orig[1], shape_orig[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # 3. Postprocesar Lineas de Carril — ídem
        ll_predict = ll_seg_out[:, 1, :, :] > ll_seg_out[:, 0, :, :]
        ll_seg_mask = ll_predict.byte().cpu().numpy()[0]
        ll_seg_mask = ll_seg_mask[pad_t:pad_t + new_h, pad_l:pad_l + new_w]
        ll_seg_mask = cv2.resize(ll_seg_mask, (shape_orig[1], shape_orig[0]),
                                 interpolation=cv2.INTER_NEAREST)

        return detecciones, da_seg_mask, ll_seg_mask
