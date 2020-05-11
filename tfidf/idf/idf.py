import math
from abc import ABC, abstractmethod


# using versions for idf according to: https://en.wikipedia.org/wiki/Tf%E2%80%93idf


class AbstractIDF(ABC):
    def __init__(self, n_t, N):
        # _n_t = number of documents containing the specific word
        # _N = total docs in corpus
        self._n_t = n_t
        self._N = N

    @property
    def n_t(self):
        return self._n_t

    @n_t.setter
    def n_t(self, new_n_t):
        self._n_t = new_n_t

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, new_N):
        self._N = new_N

    @abstractmethod
    def compute_idf(self):
        pass


# inverse document frequency
class IDF(AbstractIDF):
    def compute_idf(self):
        res = math.log10(self._N / float(self._n_t))
        print(f'{self.__class__.__name__}, compute_idf(). res: {res}')
        return res


# inverse document frequency smooth
class IDFSmooth(AbstractIDF):
    def compute_idf(self):
        res = math.log10(self._N / float(1 + self._n_t)) + 1
        print(f'{self.__class__.__name__}, compute_idf(). res: {res}')
        return res


# probabilistic inverse document frequency
class ProbIDF(AbstractIDF):
    def compute_idf(self):
        res = math.log10((self._N - self._n_t) / float(self._n_t))
        print(f'{self.__class__.__name__}, compute_idf(). res: {res}')
        return res
