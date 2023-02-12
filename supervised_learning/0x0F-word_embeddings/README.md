# 0x0F. Natural Language Processing - Word Embeddings #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is natural language processing?
- What is a word embedding?
- What is bag of words?
- What is TF-IDF?
- What is CBOW?
- What is a skip-gram?
- What is an n-gram?
- What is negative sampling?
- What is word2vec, GloVe, fastText, ELMo?

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Bag Of Words | Write a function def bag_of_words(sentences, vocab=None): that creates a bag of words embedding matrix | 0-bag_of_words.py |
| 1. TF-IDF | Write a function def tf_idf(sentences, vocab=None): that creates a TF-IDF embedding | 1-tf_idf.py |
| 2. Train Word2Vec | Write a function def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a gensim word2vec model | 2-word2vec.py |
| 3. Extract Word2Vec | Write a function def gensim_to_keras(model): that converts a gensim word2vec model to a keras Embedding layer | 3-gensim_to_keras.py |
| 4. FastText | Write a function def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a genism fastText model | 4-fasttext.py |
| 5. ELMo | When training an ELMo embedding model, you are training | 5-elmo |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
