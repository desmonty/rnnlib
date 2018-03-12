# rnnlib
A library for recurrent neural networks in [Dlang](https://dlang.org/)

The aim of this project is to build a RNN factory such that one can construct
RNN architecture using simple layers and any non linear function.

I choose to train the neural networks using [Evolutionnary Algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm).
The reason why we do not use any optimization based on gradient descent is to avoid many issues it gives us when optimizing
a rnn like vanishing/exploding gradients (e.g. see _[On the difficulty of training recurrent neural networks](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)_).

One of the auxiliary goal of this project is to cut any dependencies between the evolutionary algorithms
and the recurrent neural netwroks to enable anyone to use it independently.

# TODO
* RNN
  * _Matrices_
    * ~~Diagonal~~
    * ~~Reflection~~
    * ~~Permutation~~
    * ~~Fourier~~
    * Unitary
    * Block
    * Unittest
    * Optimization
  * _Vectors_
    * Unittest
    * Optimization
  * _Layers_
    * Linear
    * Functional
    * Recurrent
* EVO
  * Evolution Strategy
  * Genetic Algorithm
  * Particle Swarm optimization
  * Ant colony optimization
