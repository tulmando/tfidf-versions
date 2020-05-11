# tfidf-versions

## definitions
- n_t = number of documents containing the specific word
- N = total docs in corpus
- Wiki IDF variations
![](./images/wiki-idf-variations.png)
    - I will use IDF, SmoothIDF, ProbIDF
    
    
## conclusions
- tf idf variations (code is here: [sim_tf_idf_variations.py](./tfidf/simulation/sim_tf_idf_variations.py))
![](images/res_sim_tf_idf_variations.png) 
- For comparing between common words in the corpus, the ProbIDF is better (right side of the graph):
    - idf variations with high n_t (code is here: [sim_idf_high_n_t.py](./tfidf/simulation/sim_idf_high_n_t.py))
        - Notice, it's a zoom in on the right part of ""IDF Variations" graph.  
        ![](images/res_sim_idf_high_n_t.png)
        - For example: for 2 words w1, w2 with n_t =~ N (appears in almost all docs in the corpus):
            - Assume w1_tf = w2_tf
            - When using IDF / SmoothIDF: w1_idf =~ w2_idf ---> w1_tfidf =~ w2_tfidf
                 - It is hard to tell the difference!
            - When using ProbIDF: w1_idf != w2_idf ---> w1_tfidf != w2_tfidf.
                - you can tell the difference!
        - Conclusion: when analyzing common words, consider use the ProbeIDF.

- For new words (not in the corpus yet), don't use normal IDF (it's IDF could not be calculated, due to division by zero)

- SmoothID is better than IDF in cases we want to ensure that the words with an IDF score of zero don’t get suppressed entirely (thanks to the 1 added to the log in the IDF).
    - For example, in machine learning which want to use tf-idf as a pre-processing step. The goal is to increase the weight of rare words and reduce the weight of high frequency function words (just reduce, not make it 0...).
- For giving more weight to the IDF part (of the TF*IDF formula) we can use the second variation of the TF formula (with the log).

## notes
- sklearn can be used for calculating tfidf (I avoided it here).
    - For a real world example, I will use it.        