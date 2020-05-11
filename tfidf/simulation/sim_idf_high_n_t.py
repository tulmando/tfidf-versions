from tfidf.simulation.vector_gen import *
import matplotlib.pyplot as plt
import numpy as np

idf_N = 100000000  # assume 100 Million docs in the corpus

# build vectors - idf
v1_idf_n_t = np.linspace(idf_N - 1000, idf_N - 100, 1000)
print(v1_idf_n_t)
v_idf1 = create_idf_vector(v1_idf_n_t, idf_N, 'IDF')
v_idf2 = create_idf_vector(v1_idf_n_t, idf_N, 'IDFSmooth')
v_idf3 = create_idf_vector(v1_idf_n_t, idf_N, 'ProbIDF')

# v_idf1
plt.scatter(v1_idf_n_t, v_idf1, color='blue')
idf_blue = plt.plot(v1_idf_n_t, v_idf1, color='blue', label='IDF')
# v_idf2
plt.scatter(v1_idf_n_t, v_idf2, color='red')
idf_red = plt.plot(v1_idf_n_t, v_idf2, color='red', label='IDFSmooth')
# v_idf3
plt.scatter(v1_idf_n_t, v_idf3, color='orange')
idf_orange = plt.plot(v1_idf_n_t, v_idf3, color='orange', label='ProbIDF')

plt.xlabel("number of documents containing the specific word")
plt.ylabel("IDF")
plt.title('IDF variations - n_t ~= N')
plt.legend(loc="upper right")
plt.show()
