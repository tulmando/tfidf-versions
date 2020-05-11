from tfidf.tf.tf import LinearTF, LogTF
from tfidf.idf.idf import IDF, IDFSmooth, ProbIDF


# TODO use factory method design pattern instead
def factory(type, tup0, tup1):
    if type == 'LinearTF':
        return LinearTF(tup0, tup1)
    elif type == 'LogTF':
        return LogTF(tup0, tup1)
    elif type == 'IDF':
        return IDF(tup0, tup1)
    elif type == 'IDFSmooth':
        return IDFSmooth(tup0, tup1)
    elif type == 'ProbIDF':
        return ProbIDF(tup0, tup1)


def create_tf_vector(vec, words_in_doc, class_type):
    res = []
    for val in vec:
        instance = factory(class_type, val, words_in_doc)
        res.append(instance.compute_tf())
    return res


def create_idf_vector(vec, idf_N, class_type):
    res = []
    for val in vec:
        instance = factory(class_type, val, idf_N)
        res.append(instance.compute_idf())
    return res
