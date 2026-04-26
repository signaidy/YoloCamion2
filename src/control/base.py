from abc import ABC, abstractmethod

from src.tipos import ComandoControl


class Controlador(ABC):
    @abstractmethod
    def aplicar(self, cmd: ComandoControl) -> None: ...

    @abstractmethod
    def liberar(self) -> None:
        """Suelta todos los controles. Se llama en emergencia o al cerrar."""

    @abstractmethod
    def cerrar(self) -> None: ...
