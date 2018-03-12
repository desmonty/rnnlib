# rnnlib
A library for recurrent neural networks in [Dlang](https://dlang.org/)

The aim of this project is to build a RNN factory such that one can construct
RNN architecture using simple layers and any non linear function.

I choose to train the neural networks using [Evolutionnary Algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm).
The reason why we do not use any optimization based on gradient descent is to avoid many issues it gives us when optimizing
a rnn like vanishing/exploding gradients (e.g. see _[On the difficulty of training recurrent neural networks]_(http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)).

One of the auxiliary goal of this project is to cut any dependencies between the evolutionary algorithms
and the recurrent neural netwroks to enable anyone to use it independently.

# TODO
1. RNN
  2. _Matrices_
    3. Diagonal
    3. Reflection
    3. Permutation
    3. Fourier
    3. Unitary
    3. Block
    3. Unittest
    3. Optimization
  2. _Vectors_
    3. Unittest
    3. Optimization
  2. _Layers_
    3. Linear
    3. Functional
    3. Recurrent
1. EVO
  2. Evolution Strategy
  2. Genetic Algorithm
  2. Particle Swarm optimization
  2. Ant colony optimization
