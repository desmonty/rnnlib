module source.optimizers.loss_functions;

import source.NeuralNetwork;
import source.Parameter;

auto loss_squared_mean(T)(ref NeuralNetwork!T _nn,
                          in Vector!T[] _X,
                          in Vector!T[] _Y,
                          ref Vector!T[] _Y_tmp)
{   
    /+
     + Arguments:
     +
     +  - _parameters (Vector!T): The weights to give to the NN.
     +
     +  - _nn (NeuralNetwork!T): The NeuralNetwork to test.
     +
     +  - _X (Vector!T[]): The training data, inputs.
     +
     +  - _Y (Vector!T[]): The training data, outputs.
     +
     +  - _Y_tmp (Vector!T[]): This is used as placeholder for the nn(_X[i]).
     +                         Hence, there is no memory allocation in the computation
     +                         of the loss function.
     +
     +
     + Description:
     +
     +  This function take a neural networks together with its weights and the training set
     +  and return the loss, computed as the mean squared error between _Y and nn(_X).
     +/
    T delegate(in Vector!T _parameters) dg;
    dg = delegate(in Vector!T _parameters) {
        T loss_value = 0.0;

        // We equipe the neural network with the weigth given in parameters.
        _nn.set_parameters(_parameters);

        // We loop over all data points and compute the sum of squared errors.
        foreach(i; 0 .. _X.length) {
            _nn.apply(_X[i], _Y_tmp[i]);
            _Y_tmp[i] -= _Y[i];
            loss_value += _Y_tmp[i].norm!"L2";
        }

        return loss_value / _X.length;
    };
    return dg;
}
