# BiLSTM-Based Models for Metaphor Detection

This repository is the [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html) implementation for the network presented in:

> Shichao Sun and Zhipeng Xie,
> [BiLSTM-Based Models for Metaphor Detection](https://link.springer.com/chapter/10.1007/978-3-319-73618-1_36)
> NLPCC 2017 

Concat: [scsun17@fudan.edu.cn](mailto:scsun17@fudan.edu.cn)

## Requirements
- python 2.7
- [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html) 0.2.dev1
- [gensim](https://radimrehurek.com/gensim/) 1.0.1
- [scikit-learn](http://scikit-learn.org/stable/) 0.18.1
- [NLTK](https://www.nltk.org/) 3.2.2
- [Theano](http://deeplearning.net/software/theano/index.html) 0.8.0
- [Numpy](http://www.numpy.org/) 1.12.1

## Train and Test
- Stage 1: Download the `word2vec` pre-trained **Google News** corpus 
    [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
- Stage 2: run the code to train and test
```
python metaphor.py
```

## Citation

    @inproceedings{sun2017bilstm,
    title={BiLSTM-Based Models for Metaphor Detection},
    author={Sun, Shichao and Xie, Zhipeng},
    booktitle={National CCF Conference on Natural Language Processing and Chinese Computing},
    pages={431--442},
    year={2017},
    organization={Springer}
    }
