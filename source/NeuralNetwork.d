module source.NeuralNetwork;

import std.algorithm: map, sum;
import std.array: array;
import std.conv: to;
import std.exception: enforce;

import source.Layer;
import source.Matrix;
import source.Parameter;
import source.Utils;

version(unittest)
{
    import core.exception;

    import std.math: abs;
    import std.stdio : writeln, write;
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

    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

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

    @safe pure
    this(in size_t _dim_in)
    {
        // We set the first elements of these arrays to null because
        // the first layer is the "Input".
        layers = [null];
        results = [null];
        input_layers = [null];
        id = 1;

        id_to_name[0] = "input";
        name_to_id["input"] = 0;

        arr_dim_in = [_dim_in];
        arr_dim_out = [_dim_in];
    }


    /++ A general method to add a Layer.
     +  It should be sufficient for all the cases as it only
     +  handles the direction logic between the layers.
     +
     +  It use lazy evaluation to create create the layer at the right time,
     +  being lazy is the solution for a better generalization !
     +/
    private
    @safe
    auto addLayer(size_t _dim_out,
                  lazy Layer!T _create_layer,
                  in bool _use_bias,
                  in Tc _randomBound,
                  in string _name,
                  Vector!T _state,
                  in string[] _in,
                  in string[] _to)
    {
        // If a name was given and if it is already used.
        if (_name && (_name in name_to_id))
            throw new Exception(_name~" is already used as a name.");

        // If a specific name is given, we remember its id to be able to retreive it.
        if (_name) {
            id_to_name[id] = _name;
            name_to_id[_name] = id;
        }

        // Result will be filled by NeuralNetwork.compute.
        if (_state)
            results ~= _state.dup;
        else {
            results ~= null;
        }

        // If the inputs are not given, we assume it is the last defined layer.
        size_t[] _inputs = [id - 1];
        if (_in)
            _inputs = _in.map!(a => name_to_id[a]).array();

        input_layers ~= _inputs;


        // If the dimension of the output vector is zero, set it to the input dimension.
        if (!_dim_out)
            _dim_out = arr_dim_out[_inputs[0]];

        // If the layer points to other layers, we add it to their inputs.
        // And we initialize the state vecotr if this is not already done.
        if (_to) {
            foreach(tmp_id; _to) {
                enforce(tmp_id in name_to_id, tmp_id~": name not found.");
                input_layers[name_to_id[tmp_id]] ~= id;
            }

            if (results[$-1] is null)
                results[$-1] = new Vector!T(_dim_out, 0);
        }

        // Update dimension arrays.
        arr_dim_in ~= arr_dim_out[_inputs[0]];
        arr_dim_out ~= _dim_out;

        // Add the layer to the network.
        layers ~= _create_layer();

        ++id;
        return this;
    }


    /++ Create a Linear Layer in the network and handle the logic for futur
     +  computation.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _use_bias (bool, =false): Add a bias vector to the output if true.
     +      _randomBound (Tc, =1.0): Used for the generation of random values in the parameters.
     +      _type (string, ="Matrix"): The type of the matrix to create (e.g. Fourier, Unitary).
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto Linear(size_t _dim_out=0,
                in bool _use_bias=false,
                in Tc _randomBound=1.0,
                in string _type="Matrix",
                in string _name=null,
                Vector!T _state=null,
                in string[] _in=null,
                in string[] _to=null)
    {
        if (!_dim_out)
            _dim_out = arr_dim_out[$-1];

        return addLayer(_dim_out,
                        new MatrixLayer!T(_type, [_dim_out, arr_dim_in[id]],
                                           _use_bias, _randomBound),
                        _use_bias, _randomBound,
                        _name, _state, _in, _to);
    }


    /++ Recurrent Layer
     +
     +  It is simply constructed by adding two layers:
     +  - The first one is a FunctionLayer which input is the last layer's result
     +  - The second one is a MatrixLayer takes its input from the first and add a
     +    redirection of its output towards the function layer (the recurrence).
     +
     +  Args:
     +      _function (string, ="relu"): The name of the function to use as non-linearity.
     +      _type (string, ="Matrix"): The type of the matrix to create (e.g. Fourier, Unitary).
     +      _randomBound (Tc, =1.0): Used for the generation of random values in the parameters.
     +      _name_in (string, =null): Name of the first layer for futur redirection.
     +      _name_to (string, =null): Name of the last layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto Recurrent(in string _function="relu",
                   in string _type="Matrix",
                   in Tc _randomBound=1.0,
                   string _name_in=null,
                   in string _name_to=null,
                   Vector!T _state=null,
                   in string[] _in=null,
                   in string[] _to=null)
    {
        if (!_name_in)
            _name_in = "IN_RECURRENT_LAYER_" ~ to!string(id);

        auto tmp_to = _to ~ _name_in;

        if (!_state)
            _state = new Vector!T(arr_dim_out[$-1], _randomBound);

        this.Function(_function,  // Function name.
                      0,          // dim out = dim in.
                      _name_in,   // name for futur reference in the recurrent layer.
                      null,       // no state vector.
                      _in,        // 'in' references. 
                      null);      // no 'out' references.

        this.Linear(0,            // dim out = dim in.
                    false,        // no bias vector.
                    _randomBound, // random bound for the initialisation.
                    _type,        // Matrix type.
                    _name_to,    // Name for futur references out of the recurrent layer.
                    _state,       // intial state vector.
                    null,         // no 'in' references.
                    tmp_to);         // 'out' references.

        return this;
    }

    /++ Functional layer.
     +
     +  Args:
     +      _function (string): The name of the function to use as non-linearity.
     +                          This can also be a (Vector!T delegate(Vector!T) )
     +                          and a (Vector!T delegate(Vector!T, Parameter[])).
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +
     +/
    auto Function(in string _function,
                  in size_t _dim_out=0,
                  in string _name=null,
                  Vector!T _state=null,
                  in string[] _in=null,
                  in string[] _to=null)
    {
        return addLayer(_dim_out,
                        new FunctionalLayer!T(_function, arr_dim_out[$-1]),
                        false, 0.0,
                        _name, _state,
                        _in, _to);
    }

    auto Function(Vector!T delegate(Vector!T) _function,
                  in size_t _dim_out=0,
                  in string _name=null,
                  Vector!T _state=null,
                  in string[] _in=null,
                  in string[] _to=null)
    {
        return addLayer(_dim_out,
                        new FunctionalLayer!T(_function),
                        false, 0.0,
                        _name, _state,
                        _in, _to);
    }

    auto Function(Vector!T delegate(Vector!T, Parameter[]) _function,
                  in size_t _dim_out=0,
                  in string _name=null,
                  Vector!T _state=null,
                  in string[] _in=null,
                  in string[] _to=null)
    {
        return addLayer(_dim_out,
                        new FunctionalLayer!T(_function),
                        false, 0.0,
                        _name, _state,
                        _in, _to);
    }

    /++ Serialize all the parameters in the neural networks.
     +  This funciton should be called at the end of the construction of the neural network.
     +  It is a MANDATORY step for the optimization of the network.
     +
     +  This method will create an array of type 'T[]' so that all the parameters (Matrix and Vectors)
     +  should have their weights in it. In practice, what we want is to change the neural networks
     +  weights by changing only the array.
     +  This way of changing the weights of the neural networks should ease the work of the optimization
     +  algorithms.
     +/
    @property
    void serialize()
    {
        // First we want to know the size of the total array.
        // We sum the size of each layer.
        size_t total_size = 0;
        foreach(tmp_l; layers)
            if (!(tmp_l is null))
                total_size += tmp_l.params
                                   .map!(a => paramsToSize!T(a))
                                   .sum;

        serialized_data = new T[total_size];

        // TODO: Go through each layer, copy the data in serialized_data
        //       and then make the layers points to it.

        size_t _index = 0;
        foreach(tmp_l; layers)
            if (!(tmp_l is null))
                foreach(tmp_param; tmp_l.params)
                    takeOwnership!T(serialized_data, tmp_param, _index);
    }

    /// Apply the NeuralNetwork to the vector and change the NN state if needed.
    Vector!T compute(in Vector!T _v)
    {
        results[0] = _v.dup;
        Vector!T tmp_vec;

        foreach(cur_id; 1 .. id)
        {
            // If there is only one input,
            // we just pass it to the layer for computation. 
            if (input_layers[cur_id].length == 1)
                results[cur_id] = layers[cur_id].compute(results[input_layers[cur_id][0]]);
            else {
                // Else we need to create a temporary vector to sum all the inputs.
                tmp_vec = new Vector!T(arr_dim_in[cur_id], 0);
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

    // Neural Network 1: Simple linear layer neural network.
    // Neural Network 2: Two linear layer with a recurrence.
    
    {
        // Initialize the neural network.
        // At this point, we have the identity function.
        auto nn = new NeuralNetwork!float(4);

        // Vector of L2 norm = 1.
        auto v = new Vector!float([0.5, 0.0, -0.5, 0.7071068]);

        // w should be equal to v.
        auto w = nn.compute(v);
        w -= v;

        assert(w.norm!"L2" <= 0.0001);

        // We add a Linear Layer of shape (6, 4).
        nn.Linear(6, false, 1.0, "Matrix", "L1");
        w = nn.compute(v);


        // Hence, the resulting vector should have length 6.
        assert(w.length == 6);

        // We add some complexity: the layer take the user's input and the ouput of "L1"
        // and return its output to "L1" (And so create a rnn-like structure) and to
        // the output (by default, the result of the last layer).
        nn.Linear(4, false, 1.0, "Matrix", "L2", null, null, ["L1"]);
        w = nn.compute(v);
        auto z = nn.compute(v);


        // Now we reconstruct what we think the neural network should compute.
        // w
        auto w_bis = (cast(Matrix!float) nn.layers[1].params[0]) * v;
        w_bis *= (cast(Matrix!float) nn.layers[2].params[0]);

        auto hidden = w_bis.dup;
        w_bis -= w;

        assert(w_bis.norm!"L2" <= 0.0001);

        // z
        auto z_bis = hidden;
        z_bis += v;
        z_bis = (cast(Matrix!float) nn.layers[1].params[0]) * z_bis;
        z_bis = (cast(Matrix!float) nn.layers[2].params[0]) * z_bis;
        
        z_bis -= z;
        assert(z_bis.norm!"L2" <= 0.0001);
    }
    
    // Neural Network 1: Simple linear layer + softmax.
    // Neural Network 2: RNN: Linear + relu + Linear (+backlink) + Linear + softmax.
    {

        auto vec = new Vector!real(5, 1.0);

        // NN1
        auto nn1 = new NeuralNetwork!real(5);
        nn1.Linear(5)
           .Function("softmax", 5);



        auto w = nn1.compute(vec);

        assert(abs(1 - w.norm!"L1") <= 0.0001);


        // NN2
        auto nn2 = new NeuralNetwork!real(5);
        nn2.Linear(5, true)
           .Recurrent()
           .Linear(5, true)
           .Function("softmax");



        auto s = nn2.results[3].dup; 

        auto nn3 = new NeuralNetwork!real(5);
        nn3.Linear(5, true, 1.0, "Matrix")
           .Function("relu", 5, "Rec_in")
           .Linear(5, false, 1.0, "Matrix", null, s, null, ["Rec_in"])
           .Linear(5, true)
           .Function("softmax");



        // put same parameters in the second nn.
        nn3.layers[1].params[0] = (cast(Matrix!real) nn2.layers[1].params[0]).dup;
        nn3.layers[1].params[1] = (cast(Vector!real) nn2.layers[1].params[1]).dup;
        
        nn3.layers[3].params[0] = (cast(Matrix!real) nn2.layers[3].params[0]).dup;

        nn3.layers[4].params[0] = (cast(Matrix!real) nn2.layers[4].params[0]).dup;
        nn3.layers[4].params[1] = (cast(Vector!real) nn2.layers[4].params[1]).dup;



        auto a_1 = nn2.compute(vec);
        auto a_2 = nn2.compute(a_1);
        auto a_3 = nn2.compute(a_2);
        auto a_4 = nn2.compute(vec);
        auto me1 = a_4.dup;

        auto b_1 = nn3.compute(vec);
        auto b_2 = nn3.compute(b_1);
        auto b_3 = nn3.compute(b_2);
        auto b_4 = nn3.compute(vec);
        auto me2 = b_4.dup;

        // If this is a recurrent neural network, this should be ok.
        me1 -= a_1;
        assert(a_4.norm!"L2" >= 0.1);
        me2 -= b_1;
        assert(b_4.norm!"L2" >= 0.1);

        b_1-=a_1;
        b_2-=a_2;
        b_3-=a_3;
        b_4-=a_4;

        assert(b_1.norm!"L2" <= 0.0001);
        assert(b_2.norm!"L2" <= 0.0001);
        assert(b_3.norm!"L2" <= 0.0001);
        assert(b_4.norm!"L2" <= 0.0001);
    }

    // Neural Network: Linear + softmax
    {
        auto nn = new NeuralNetwork!real(4);
        nn.Linear(4, true)
          .Function("softmax")
          .serialize;

        auto v = new Vector!real([1.0, -1.0, 0.0, 1.0]);

        assert(nn.serialized_data.length == 4*4+4);

        foreach(i; 0 .. nn.serialized_data.length)
            nn.serialized_data[i] = 1.0;

        v = nn.compute(v);

        auto res = new Vector!real([0.25, 0.25, 0.25, 0.25]);
        res -= v;

        assert(res.norm!"L2" <= 0.0001);
    }

    writeln("TODO.");
}