module source.NeuralNetwork;

import std.algorithm: map;
import std.array: array;


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
 +  Note:
 +
 +      - The output of NeuralNetwork.compute will always be the output of the last
 +        layer only. If you want to ouput from another layer, you shouldn't, but
 +        you can, be smart.
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
    	// Array of the layers.
        Layer!T[] layers;

        // Save the result of the computation of a layer. Allow its multiple use.
        Vector!T[] results;

        // Used to know which layers to ask for the input of a specific layer.
        size_t[][] input_layers;
        
        // Number of layers in the neural network.
        size_t id;

        // Array of the input dimension for a specific layer.
        size_t[] arr_dim_in;

        // Array of the output dimension for a specific layer.
        size_t[] arr_dim_out;

        // Map the name of the layer to its id.
        size_t[string] name_to_id;

        // Map the id of the layer to its name.
        string[size_t] id_to_name;

        // Give access to all the learnable parameter of the neural network.
        T[] serialized_data;
    }

    this(in size_t _dim_in)
    {
        // We set the first elements of these arrays to null because
        // the first layer is the "Input". 
        results = [null];
        input_layers = [null];
        id = 1;

        id_to_name = ["input"];
        name_to_id["input"] = 0;

        arr_dim_in = [_dim_in];
        arr_dim_out = [_dim_in];
    }

    /++ Create a Linear Layer in the network and handle the logic for futur
     +  computation.
     +
     +  Args:
     +      _dim_out (size_t): Dimension of the resulting vector.
     +      _use_bias (bool): Add a bias vector to the output if true.
     +      _in (size_t[]): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +/
    auto
    addLinearLayer(in size_t _dim_out,
                   in bool _use_bias=false,
                   in Tc _randomBound=1.0,
                   in string _type="Matrix",
                   in string[] _in=null,
                   Vector!T _state=null,
                   in string _name=null)
    {
        // If the dimension of the output vector is zero.
        if (!_dim_out)
            throw new Exception("Cannot set output dimension to zero.");

        // If a name was given and if it is already used.
        if (_name && (_name in name_to_id))
            throw new Exception(_name~" is already used as a name.");

        // If a specific name is given, we remember its id to be able to retreive it.
        if (_name) {
            id_to_name[id] = _name;
            name_to_id[_name] = id;
        }

        // If the inputs are not given, we assume it is the last defined layer.
        size_t[] _inputs = [id - 1];
        if (_in)
            _inputs = _in.map(a => name_to_id[a]).array();

        input_layers ~= [_inputs];

        // Update dimension arrays.
        arr_dim_in ~= [arr_dim_in[_inputs[0]]];
        arr_dim_out ~= [_dim_out];

        // Result will be filled by NeuralNetwork.compute.
        if (_state)
            results ~= [_state.dup];

        // Create the Linear Layer.
        auto tmp_layer = new MatrixLayer!T(_type, [arr_dim_in[id], _dim_out], _use_bias, _randomBound);

        // Add the layer to the network.
        layers ~= [tmp_layer];

        ++id;
        return this;
    }

    /// Apply the NeuralNetwork to the vector and change the NN state if needed.
    Vector!T compute(in Vector!T _v)
    {
        results[0] = _v;
        Vector!T tmp_vec;

        foreach(cur_id; 1 .. id)
        {
            // If there is only one input, we just pass it to the layer for computation. 
            if (input_layers[cur_id].length == 1)
                results[cur_id] = layers[cur_id].compute(results[input_layers[cur_id][0]]);
            else {
                // Else we need to create a temporary vector to sum all the inputs.
                tmp_vec = new Vector!T(arr_dim_in[id], 0);
                foreach(tmp_id; input_layers[cur_id])
                    tmp_vec += results[tmp_id];

                // Finally, we compute the tmp vector using the layer.
                results[cur_id] = layers[cur_id].compute(tmp_vec);
            }
        }

        return results[$-1];
    }
}
unittest {
        write("Unittest: NeuralNetwork ... ");

        writeln("Done");
}