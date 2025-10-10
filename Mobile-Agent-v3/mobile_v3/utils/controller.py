from abc import ABC, abstractmethod

class Controller(ABC):
    @abstractmethod
    def get_screenshot(self, save_path):
        pass

    @abstractmethod
    def tap(self, x, y):
        pass

    @abstractmethod
    def type(self, text):
        pass

    @abstractmethod
    def slide(self, x1, y1, x2, y2):
        pass

    @abstractmethod
    def back(self):
        pass

    @abstractmethod
    def home(self):
        pass
