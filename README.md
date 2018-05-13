# rnnlib
A library for recurrent neural networks in [Dlang](https://dlang.org/)

The aim of this project is to build a RNN factory such that one can construct
RNN architecture using simple layers and any non linear function.

The neural networks will primarily be trained using [Evolutionnary Algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm).
The reason why we do not use any optimization based on gradient descent is to avoid many issues it gives us when optimizing
a rnn like vanishing/exploding gradients (e.g. see _[On the difficulty of training recurrent neural networks](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)_).

One of the auxiliary goal of this project is to cut any dependencies between the evolutionary algorithms
and the recurrent neural netwroks to enable anyone to use it independently.


# Research Goals
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


## Daily Todo
* Think about parameters shared between layers: can the layer be different and share the same parameters ?
* (optm) randclone weigths: serialized_data holds all weigths -> can be constructed using smaller array with random cloning.


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
    * [Universe](https://github.com/openai/universe)
    * Zbaghul
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
  * Travis CI ?
  * [Command Line git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) (+ .gitignore)
  * dub
  * [For most of these](https://github.com/libmir/mir-algorithm)

