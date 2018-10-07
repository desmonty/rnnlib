[![Build Status](https://travis-ci.org/desmonty/rnnlib.svg?branch=master)](https://travis-ci.org/desmonty/rnnlib)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/desmonty/rnnlib?svg=true)](https://ci.appveyor.com/api/projects/status/github/desmonty/rnnlib)
[![codecov](https://codecov.io/gh/desmonty/rnnlib/branch/master/graph/badge.svg)](https://codecov.io/gh/desmonty/rnnlib)

# rnnlib
A library for (recurrent) neural networks in [Dlang](https://dlang.org/)

The aim of this project is to build a RNN factory such that one can construct
RNN architecture using simple layers and possibly any functions.

Firstly, a focus will be put on gradient-based algorithm (e.g. SGD) to train the nns.
SGD will then be used as a baseline against which we will test gradient-free algorithms (e.g. [Evolutionnary Algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm)).

https://docs.google.com/document/d/1YrIUGw8XOrTrF7r6YTlCyMAbhDv9xIM5qTRsI9WwcdI/edit

## Bonus
- [ ] static nn
- [ ] staticly trained nn

## Research Goals
This library will be used to try to answer the following questions:

* _Optimizations_
  * How good are gradient free algorithms in training (r)nn ? Metrics: Training Time / Precision / Generalization
  * Benchmark Several different algorithms against a set of problems.
* _Weights Augmentation_
  * Gradient-free optimizations algorithms can be much more time consuming than SGD. We could remove this issue by applying dimensionality augmentation technics to generate the weights of the NN with a small intial weigths vector (This can be compared to Genetic Algorithm). Hence the goal would be to test different technics to augment small vectors.
  * Random Sampling
  * Random Matrix
  * Random NN
  * Random * with constraint (e.g. distribution of final weigths).
* _Sparsification_
  * Weights augmentation allow us to work with potentially huge neural networks (Billion of weigths). The resulting efficiency is still to be studied but what we can be sure of is that it will be energy consuming to use a huge RNN or FFNN on ligthweigth device (iot, mobile, ...). Hence we would like to come up with a way to reduce drastically the number of nonzero element in every matrices. There is several ways to do this.
    1) Spectral sparsification (between learning phase ?).
    2) Use the fact we used weigths augmentation.
    3) Train sparse matrix (e.g. block matrices).
* _Connection Optimization_
  * The use of gradient-free optimization allow us to optimize the way nodes in the neural networks are connected. Discussion: We could either want to modify the connection of the neural network viewed as a graph or the connection between computational component of the nn.  
* _Scientific paper generation_
  * Because why not ? And also this could be a nice POC that rnnlib works well for image generation (let's implement GANs).
* _"Linear" Tropical Neural Networks_
  * mult -> plus & plus -> max/min
  * The use of gradient free optimizations algorithms allow us to train Neural Network defined over a tropical algebra. This could be nice because the dot product becomes a non-linear function of the weigths.
  * Questions: Non-linear functions useful in this context ? Some theory possible ? Does the Universal Approximation Theorem stands here ? What can we do with a "Linear Tropical Neural Network" (only Matrix operation). 
* _Compression_


# TODO
* RNN
  * _Matrices_
    * Optimization
      * Use a single "dot" product and optimize it.
      * Matrix abstract multiplication: mult(auto override this T)
  * _Vectors_
    * Optimization
      * alias array for vector ?
      * use arrayfire ? 
  * _Layers_
    * ~Linear~
    * ~Functional~
    * ~Recurrent~
  * _Neural Network_
    * 
* TRAINING
  * _Evolutionary Algorithms_
    * Evolution Strategy
    * Genetic Algorithm
    * Particle Swarm optimization
    * Ant colony optimization
  * _Other Gradient-free Optimization_
    * Nelder–Mead Simplex
    * DIRECT
    * DONE
    * Pattern Search
    * MCS
* TESTS
  * _Functional Tests_
    * [Some good examples](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
  * _Visualizable Tests_
    * test the optimization algorithm for drawing graphs: force directed layout gives a function.
  * _Machine Learning Tests_
    * Adding Problem
    * XOR
    * Copying memory
    * Pixel-by-pixel MNIST (+ permuted)
    * NLP ?
    * Music Generation ?
  * _Deep Reinforcement Learning Final Tests_
    * [Gym](https://gym.openai.com/)
    * ~~[Universe](https://github.com/openai/universe)~~ (too large a dependency)
    * Zbaghul
    
* GPU
 * OpenCL computing of graph
  - translate entire graph to openCL kernel?
  - optimization pass with similar system to https://github.com/CNugteren/cltune

* DOCUMENTATION
  * Vector
  * Matrix
  * Layer
  * Neural Network
  * Optimization Algorithms
    * EA
    * OGFO
* FORMAT
  * CodeCov
  * [For most of these](https://github.com/libmir/mir-algorithm)

