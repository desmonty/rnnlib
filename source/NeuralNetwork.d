module source.NeuralNetwork;

import source.Layer;
import source.Matrix;
import source.Parameter;

version(unittest)
{
    import std.stdio : writeln, write;
    import core.exception;
}

/++ The Neural Network class hold all the logic of the function approximator.
 +  
 +  Args:
 +      T: Type of the element in the network.
 +
 +  Members:
 +      layers (Layer!T[string]): An associative array mapping the name of a
 +                                layer to itself.
 +     input_layers (string[][string]): An associative array mapping layer's name
 +                                       with a list of name of layer which send
 +                                       their the layer.
 +      output_layers (string[][string]): An associative array mapping layer's name
 +                                        with a list of name of layer to send 
 +                                        the output vector of the layer.
 +      serialized_data (T[]): Hold all the learnable data from the NeuralNetwork.
 +                             E.g. Matrix, bias, and so on.
 +                             The serialiwed data is constructed only once at
 +                             the user's demand only.
 +
 +  Note:
 +
 +      - Adding layers in the network is done by the use of the "addLayer..."
 +        functions. These functions can all take a "_from" and a "_to" as
 +        parameter. They will be used to know how to inscribe the new layer in
 +        the network as explained below:
 +
 +            _from: The layer will take the output of all the layers in "_from".
 +                   These layers should all gives vector of the same length.
 +                   The input of the new layer is computed using a reducer
 +                   function of type (T delegate(T,T)) which will be applied
 +                   successively to the vectors' value of same indices, e.g.
 +
 +                   input_vector[i] = reducer(reducer(v_1[i], v_2[i]), v_3[i]).
 +
 +
 +            _to: The output of the layer can be sent to several layers as input.
 +                 This means that the new layer will be added to the inputs of
 +                 the layers in "_to".
 +
 +/
class NeuralNetwork(T) {
    private {
        Layer!T[string] layers;
        string[][string] input_layers;
        string[][string] output_layers;

        T[] serialized_data;
    }

    auto
    addLinearLayer(in string[] _from=null, in string[] _to=null,
                   in bool use_bias=false)
    {



        return this;
    }


    /// Apply the NeuralNetwork to the vector and change the NN state if needed.
    Vector!T compute(in Vector!T _v)
    {

    }
}
unittest {
        write("Unittest: NeuralLayer ... ");

        writeln("Done");
}