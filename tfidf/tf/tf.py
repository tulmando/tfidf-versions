import math
from abc import ABC, abstractmethod


class AbstractTF(ABC):
    def __init__(self, n_w, N):
        # _n_w = how many time the specific word appears in the doc
        # _N = total number of words in doc

        self._n_w = n_w
        self._N = N

    @property
    def n_w(self):
        return self._n_w

    @n_w.setter
    def n_w(self, new_n_w):
        self._n_w = new_n_w

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, new_N):
        self._N = new_N

    @abstractmethod
    def compute_tf(self):
        pass


class LinearTF(AbstractTF):
    def compute_tf(self):
        res = self._n_w / float(self._N)
        print(f'{self.__class__.__name__}, compute_tf(). res: {res}')
        return res


class LogTF(AbstractTF):
    def compute_tf(self):
        res = math.log10(1 + self._n_w / float(self._N))
        print(f'{self.__class__.__name__}, compute_tf(). res: {res}')
        return res
