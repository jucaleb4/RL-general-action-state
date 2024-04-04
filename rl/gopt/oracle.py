from abc import ABC
from abc import abstractmethod

class FOO(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def f(x):
        raise NotImplemented

    @abstractmethod
    def df(x):
        raise NotImplemented
