# Sequence To Sequence model implemeted on Chainer 

This repository hosts the implementation of simple sequence-to-sequence (seq2seq) model (no attention mechanism)
on top of [chainer][chainer].
This program generates a sentence in response to an input, through learning patterns in a huge amount of sentences pairs 
(e.g. conversation corpus or parallel corpus used in the field of machine translation). 

__Caution__: This model is not used NStepLSTM. 
If you only want to use seq2seq model in chainer, you should use [chainer's formal example][chainer_seq2seq]. 

[chainer]: https://github.com/pfnet/chainer "chainer"
[chainer_seq2seq]: https://github.com/pfnet/chainer/blob/seq2seq/examples/seq2seq/seq2seq.py "chainer_seq2seq"



## Description

***DEMO:***

![Demo](https://github.com/OnizukaLab/SimpleSeq2Seq/blob/master/data/s2s_sample.gif?raw=true)

In this experiment, we train the seq2seq model with movie dialogs 
from the [Cornell Movie-Dialogs Corpus][cornell].
You can get this conversation corpus using `cornell_corpus.py`.
The data will be placed in `./data/pair_corpus.txt` when you run its script.

If you want to train the model using your own corpus, you should create file with the following format.
    
    <post_sentence><TAB><comment_sentence>

A sentence in the second column is the reply to a sentence in the first column 
when you use it as a conversation corpus.
If you use it as a parallel corpus, 
a sentence in the second column is a translation of the sentence in the first column.
These sentences have to be separated by TAB.
This corpus should be placed in `./data/pair_corpus.txt` (i.e. the same name). 

[cornell]: https://people.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html "cornell"



## Features

- `seq2seq.py`
  - Seq2seq model implemented on top of [chainer][chainer].

- `util.py`
  - Module to process dictionary and corpus.

- `train.py`
  - Train seq2seq model.

- `interpreter.py`
  - Run the model trained by `train.py`.
  - You can interactevely talk with your model like a ChatBot or Translator.


## Requirement

You can easily get Python and its packages for data science by installing [Anaconda][anaconda].
Packages of chainer, gensim, and nltk in python packages are required.
I also use nkf command to change encoding of a file into UTF-8.

Required packages:
- anaconda3-2.4.0
- chainer (1.5 ~ latest)
- gensim
- nltk
- nkf

[anaconda]: https://www.continuum.io/ "anaconda"


## Usage

1. Run `cornell_corpus.py` to prepare a training corpus (named `pair_corpus.txt`).
   
   ~~~
    $ python cornell_corpus.pypython cornell_corpus.py
   ~~~
   
   If you type this command and run it, 
   python script start to download cornell corpus into own your PC.
   
2. Train the seq2seq model using the corpus.
   You can train the model by `train.py`.

   ~~~
    $ python train.py
   ~~~
   
   This script does not use GPU by default.
   If you want to use GPU, use the parameter; `--gpu`.
   
   ~~~
    $ python train.py --gpu 1
   ~~~
   
   This script use a GPU when you set the GPU flag to 1 like above.
   
   You can also set the epochs, the dimension of the hidden and word embedding layer, and the batch size
   by writing the following command.
   
   ~~~
    $ python train.py --epoch 500 --feature_num 1000 --hidden_num 1000 --batchsize 100
   ~~~

3. Run `interpreter.py` to interactively talk with the trained model like a ChatBot.
   You can choose the model by setting `MODEL_PATH` in File `interpreter.py`, line `19`.
   `train.py` saves the models in `./data/hoge.model` per epoch ("hoge" means the number of epochs),
   so you should set the path of the model which you want to see outputs from.
   As you set the hidden and feature parameters which are not default values in the training, 
   you have to teach this script its values like the following command.

   ~~~
    $ python interpreter.py --feature_num 1000 --hidden_num 1000
   ~~~ 
   
   If you set `--bar` parameter to 1, you can see the loss graphs.
   
   ~~~
    $ python interpreter.py --bar 1 --feature_num 1000 --hidden_num 1000
   ~~~ 
   

## Example

I'll show you the result of learning from Cornell Movie-Dialogs Corpus.
This is the plot of loss.


***Model Loss:***

![Model Loss](https://github.com/OnizukaLab/SimpleSeq2Seq/blob/master/data/train160epochs.png?raw=true)


Finally, here, I show some conversations with the trained model using Cornell Movie-Dialogs Corpus. 

***Sample of Talking:***

    $ python interpreter.py
      Vocabulary Size (number of words) : 796
      The system is ready to run, please talk to me!
      ( If you want to end a talk, please type "exit". )
      
      Hello!
      
      -> hello
      
      Do you want to go to school?
      
      -> No
      
      I'm very hungry.
      
      -> Then I can't help you
      
      exit
      
      -> See you!


## Reference 

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.
[Sequence to sequence learning with neural networks.][s2s_paper]
In Advances in Neural Information Processing Systems (NIPS 2014).

[s2s_paper]: http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf "s2s_paper"

## Author

[@KChikai](https://github.com/KChikai)

