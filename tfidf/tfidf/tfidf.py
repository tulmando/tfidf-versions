# a class of tfidf.
# it persists attributes of tf & idf objects.
# Those object could be changed dynamically.


class TFIDF:
    def __init__(self, tf, idf):
        self._tf = tf
        self._idf = idf

    def compute_tfidf(self):
        tf_result = self._tf.compute_tf()
        idf_result = self._idf.compute_idf()
        print(
            f'TFIDF, compute_tfidf(). tf_result: {tf_result}, idf_result: {idf_result}. result: {tf_result * idf_result}')
        return tf_result * idf_result

    @property
    def tf(self):
        return self._tf

    @tf.setter
    def tf(self, new_tf):
        self._tf = new_tf

    @property
    def idf(self):
        return self._idf

    @idf.setter
    def idf(self, new_idf):
        self._idf = new_idf
