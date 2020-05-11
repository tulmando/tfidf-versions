from tfidf.simulation.vector_gen import *
import matplotlib.pyplot as plt
import numpy as np

idf_N = 100000000  # assume 100 Million docs in the corpus

# build vectors - tf
words_in_doc = 100  # for the simulation of TF
v1_tf_n_w = np.linspace(0, words_in_doc, 1000)
v_tf1 = create_tf_vector(v1_tf_n_w, words_in_doc,  'LinearTF')
v_tf2 = create_tf_vector(v1_tf_n_w, words_in_doc, 'LogTF')

# build vectors - idf
# assumption  - 0 < idf_n_t < idf_N
v1_idf_n_t = np.linspace(100, idf_N - 100, 100)
v_idf1 = create_idf_vector(v1_idf_n_t, idf_N, 'IDF')
v_idf2 = create_idf_vector(v1_idf_n_t, idf_N, 'IDFSmooth')
v_idf3 = create_idf_vector(v1_idf_n_t, idf_N, 'ProbIDF')

# draw tf+idf as subplots
f, (ax1, ax2) = plt.subplots(1, 2)
# tf
ax1.plot(v1_tf_n_w, v_tf1, color='blue', label='LinearTF')
ax1.plot(v1_tf_n_w, v_tf2, color='red', label='LogTF')
ax1.set_title('TF variations')
ax1.set_xlabel("how many time the specific word appears in the doc")
ax1.set_ylabel("TF")
ax1.legend(loc="upper right")

# idf
ax2.plot(v1_idf_n_t, v_idf1, color='blue', label='IDF')
ax2.plot(v1_idf_n_t, v_idf2, color='red', label='IDFSmooth')
ax2.plot(v1_idf_n_t, v_idf3, color='orange', label='ProbIDF')
ax2.set_title('IDF variations')
ax2.set_xlabel("number of documents containing the specific word")
ax2.set_ylabel("IDF")
ax2.legend(loc="upper right")
plt.show()
