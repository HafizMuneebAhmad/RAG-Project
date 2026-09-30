from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query, k=3):
        pass