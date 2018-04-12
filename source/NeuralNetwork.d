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
 +      layers (Layer!T[]): An array mapping the id of a layer to itself.
 +      input_layers (size_t[][]): An array mapping layer's id with a list 
 +                                 of id of layer which send their the layer.
 +      output_layers (size_t[][]): An array mapping layer's id with a list of
 +                                  id of layer to send the output vector of the
 +                                  layer.
 +      serialized_data (T[]): Hold all the learnable data from the NeuralNetwork.
 +                             E.g. Matrix, bias, and so on.
 +                             The serialiwed data is constructed only once at
 +                             the user's demand only.
 +
 +  Note:
 +
 +      - Adding layers in the network is done by the use of the "addLayer..."
 +        functions. These functions can all take a "_in" and a "_to" as
 +        parameter. They will be used to know how to inscribe the new layer in
 +        the network as explained below:
 +
 +            _in: The layer will take the output of all the layers in "_in".
 +                   These layers should all gives vector of the same length.
 +                   The input of the new layer is computed adding successively 
 +                   the output vectors of the layer.
 +
 +            _to: The output of the layer can be sent to several layers as input.
 +                 This means that the new layer will be added to the inputs of
 +                 the layers in "_to".
 +
 +/
class NeuralNetwork(T) {
    private {
        Layer!T[] layers;
        Vector!T[] results;
        Vector!T[] states;
        size_t[][] input_layers;
        size_t size;
        size_t[string] name_to_id;

        T[] serialized_data;
    }

    this(in size_t _dim_in)
    {
        // We set the first elements of these arrays to null because
        // the first layer is the "Input". 
        states = [null];
        results = [null];
        input_layers = [null];
        size = 1;
    }

    /++ Create a Linear Layer in the network and handle the logic for futur
     +  computation.
     +
     +  Args:
     +      _dim_out (size_t): Dimension of the resulting vector.
     +      _use_bias (bool): Add a bias vector to the output if true.
     +      _in (size_t[]): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[]): A list of layers' name for the layer to send its 
     +                        outpus. If empty, this is the end, my beautiful friend.
     +      _reducer (T deleagte(T,T)): Used to reduce all the inputs vectors
     +                                  into one only.
     +/
    auto
    addLinearLayer(in size_t _dim_out,
                   in bool _use_bias=false,
                   in string[] _in=null,
                   in string[] _to=null,
                   in string _name=null,
                   in Vector!T _init_state=null,
                   T delegate(T,T) _reducer=null)
    {
        // If the dimension of the output vector is zero.
        if (!_dim_out)
            throw new Exception("Cannot set output dimension to zero.");

        // If a name was given and if it is already used.
        if (_name && (_name in name_to_id))
            throw new Exception(_name~" is already used as a name.");

        // If the inputs are not given, we assume it is the last defined layer.
        size_t[] _inputs = _in;
        if (!_in)
            _inputs = [size - 1];
        input_layers ~= _inputs;

        // Linear Layers don't have any internal states vectors.
        states ~= [null];


        auto tmp_layer = new MatrixLayer!T();



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