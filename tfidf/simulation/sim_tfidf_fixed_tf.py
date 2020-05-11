from tfidf.simulation.vector_gen import *
import matplotlib.pyplot as plt
import numpy as np

idf_N = 100000000  # assume 100 Million docs in the corpus

# build vectors - idf
# assumption  - 0 < idf_n_t < idf_N
v1_idf_n_t = np.linspace(100, idf_N - 100, 100)
v_idf1 = create_idf_vector(v1_idf_n_t, idf_N, 'IDF')
v_idf2 = create_idf_vector(v1_idf_n_t, idf_N, 'IDFSmooth')
v_idf3 = create_idf_vector(v1_idf_n_t, idf_N, 'ProbIDF')

v_tfidf1_lowTF = np.array(v_idf1) * 0.1
v_tfidf1_mediumTF = np.array(v_idf1) * 0.5
v_tfidf1_highTF = np.array(v_idf1) * 0.9
v_tfidf2_lowTF = np.array(v_idf2) * 0.1
v_tfidf2_mediumTF = np.array(v_idf2) * 0.5
v_tfidf2_highTF = np.array(v_idf2) * 0.9
v_tfidf3_lowTF = np.array(v_idf3) * 0.1
v_tfidf3_mediumTF = np.array(v_idf3) * 0.5
v_tfidf3_highTF = np.array(v_idf3) * 0.9

print(f'v_tfidf1_lowTF: {v_tfidf1_lowTF}')
print(f'v_tfidf2_lowTF: {v_tfidf2_lowTF}')
print(f'v_tfidf3_lowTF: {v_tfidf3_lowTF}')
f2, (w1, w2, w3) = plt.subplots(1, 3)
w1.plot(v1_idf_n_t, v_tfidf1_lowTF, color='blue', label='IDF')
w1.plot(v1_idf_n_t, v_tfidf2_lowTF, color='red', label='IDFSmooth')
w1.plot(v1_idf_n_t, v_tfidf3_lowTF, color='orange', label='ProbIDF')
w1.set_title('low fixed TF = 0.1, 3 versions IDF')
w1.set_xlabel("number of documents containing the specific word")
w1.set_ylabel("TF*IDF")
w1.legend(loc="upper right")

w2.plot(v1_idf_n_t, v_tfidf1_mediumTF, color='blue', label='IDF')
w2.plot(v1_idf_n_t, v_tfidf2_mediumTF, color='red', label='IDFSmooth')
w2.plot(v1_idf_n_t, v_tfidf3_mediumTF, color='orange', label='ProbIDF')
w2.set_title('medium fixed TF = 0.5, 3 versions IDF')
w2.set_xlabel("number of documents containing the specific word")
w2.set_ylabel("TF*IDF")
w2.legend(loc="upper right")

w3.plot(v1_idf_n_t, v_tfidf1_highTF, color='blue', label='IDF')
w3.plot(v1_idf_n_t, v_tfidf2_highTF, color='red', label='IDFSmooth')
w3.plot(v1_idf_n_t, v_tfidf3_highTF, color='orange', label='ProbIDF')
w3.set_title('high fixed TF = 0.9, 3 versions IDF')
w3.set_xlabel("number of documents containing the specific word")
w3.set_ylabel("TF*IDF")
w3.legend(loc="upper right")

plt.show()
