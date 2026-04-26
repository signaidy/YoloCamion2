from abc import ABC, abstractmethod
from typing import Optional

from src.tipos import Cuadro


class FuenteCuadros(ABC):
    @abstractmethod
    def iniciar(self) -> None: ...

    @abstractmethod
    def siguiente(self) -> Optional[Cuadro]: ...

    @abstractmethod
    def cerrar(self) -> None: ...

    @property
    @abstractmethod
    def esta_activa(self) -> bool: ...
