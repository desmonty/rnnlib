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
    import std.exception: assertThrown;

    import std.stdio : writeln, write;
    import std.math;
}

/++ The Neural Network class hold all the logic of the func approximator.
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
 +        funcs. These funcs can all take a "_in" and a "_to" as
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

        // Save the buffers used to handle layer. Allow its multiple use.
        Vector!T[] buffers;

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
        results = [new Vector!T(_dim_in)];
        buffers = [new Vector!T(_dim_in)];
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
            throw new Exception("NeuralNetwork: Error: "~_name~" is already used as a name.");

        // If a specific name is given, we remember its id to be able to retreive it.
        if (_name) {
            id_to_name[id] = _name;
            name_to_id[_name] = id;
        }

        // If the inputs are not given, we assume it is the last defined layer.
        size_t[] _inputs = [id - 1];
        if (_in)
            _inputs = _in.map!(a => name_to_id[a]).array();

        input_layers ~= _inputs;

        // If the dimension of the output vector is zero, set it to the input dimension.
        if (!_dim_out)
            _dim_out = arr_dim_out[_inputs[0]];

        // Result will be filled by NeuralNetwork.compute.
        if (_state)
            results ~= _state.dup;
        else {
            results ~= new Vector!T(_dim_out);
        }
        buffers ~= new Vector!T(_dim_out);

        // If the layer points to other layers, we add it to their inputs.
        // And we initialize the state vector if this is not already done.
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


    /++ Create a linear Layer in the network and handle the logic for futur
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
    auto linear(Mtype = Matrix!T)(size_t _dim_out=0,
                               in bool _use_bias=false,
                               in Tc _randomBound=1.0,
                               in string _name=null,
                               Vector!T _state=null,
                               in string[] _in=null,
                               in string[] _to=null)
    {
        if (!_dim_out)
            _dim_out = arr_dim_out[$-1];

        return addLayer(_dim_out,
                        new MatrixLayer!Mtype([_dim_out, arr_dim_in[id]],
                            _use_bias, _randomBound),
                        _use_bias, _randomBound,
                        _name, _state, _in, _to);
    }


    /++ recurrent Layer
     +
     +  It is simply constructed by adding two layers:
     +  - The first one is a funcLayer which input is the last layer's result
     +  - The second one is a MatrixLayer takes its input from the first and add a
     +    redirection of its output towards the func layer (the recurrence).
     +
     +  Args:
     +      strfunc (string, ="relu"): The name of the func to use as non-linearity.
     +      _randomBound (Tc, =1.0): Used for the generation of random values in the parameters.
     +      _name_in (string, =null): Name of the first layer for futur redirection.
     +      _name_to (string, =null): Name of the last layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto recurrent(Mtype = Matrix!T, string strfunc="relu", TypeParameter...)
                  (in Tc _randomBound=1.0,
                   string _name_in=null,
                   in string _name_to=null,
                   Vector!T _state=null,
                   in string[] _in=null,
                   in string[] _to=null,
                   in size_t[] size_parameters=[],
                   in Tc[] randomBound_parameters=[])
    {
        if (!_name_in)
            _name_in = "IN_RECURRENT_LAYER_" ~ to!string(id);

        auto tmp_to = _to ~ _name_in;

        if (!_state)
            _state = new Vector!T(arr_dim_out[$-1], _randomBound);

        this.func!(strfunc, TypeParameter)  // func name.
                  (0,          // dim out = dim in.
                   _name_in,   // name for futur reference in the recurrent layer.
                   null,       // no state vector.
                   _in,        // 'in' references. 
                   null,       // no 'out' references.
                   size_parameters,
                   randomBound_parameters);

        this.linear!(Mtype)(0,            // dim out = dim in.
                            false,        // no bias vector.
                            _randomBound, // random bound for the initialisation.
                            _name_to,     // Name for futur references out of the recurrent layer.
                            _state,       // intial state vector.
                            null,         // no 'in' references.
                            tmp_to);         // 'out' references.

        return this;
    }

    /++ Functional layer.
     +
     +  Args:
     +      strfunc (string): The name of the func to use as non-linearity.
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
    auto func(string strfunc="", TypeParameter...)
             (in size_t _dim_out=0,
              in string _name=null,
              Vector!T _state=null,
              in string[] _in=null,
              in string[] _to=null,
              in size_t[] size_parameters=[],
              in Tc[] randomBound_parameters=[])
    {
        return addLayer(_dim_out,
                        new FunctionalLayer!(T, strfunc, TypeParameter)
                                           (size_parameters, randomBound_parameters),
                        false, 0.0,
                        _name, _state,
                        _in, _to);
    }

    /++ softmax layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto softmax(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"softmax"(_dim_out, _name, _state, _in, _to);
    }
    /++ relu layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto relu(in size_t _dim_out=0,
              in string _name=null,
              Vector!T _state=null,
              in string[] _in=null,
              in string[] _to=null)
    {
        return this.func!"relu"(_dim_out, _name, _state, _in, _to);
    }
    /++ binary layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto binary(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"binary"(_dim_out, _name, _state, _in, _to);
    }
    /++ logistic layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto logistic(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"logistic"(_dim_out, _name, _state, _in, _to);
    }
    /++ identity layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto identity(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"identity"(_dim_out, _name, _state, _in, _to);
    }
    /++ tanh layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto tanh(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"tanh"(_dim_out, _name, _state, _in, _to);
    }
    /++ arctan layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto arctan(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"arctan"(_dim_out, _name, _state, _in, _to);
    }
    /++ softsign layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto softsign(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"softsign"(_dim_out, _name, _state, _in, _to);
    }
    /++ softplus layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto softplus(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"softplus"(_dim_out, _name, _state, _in, _to);
    }
    /++ sin layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto sin(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"sin"(_dim_out, _name, _state, _in, _to);
    }
    /++ gaussian layer.
     +
     +  Args:
     +      _dim_out (size_t, =0): Dimension of the resulting vector.
     +      _name (string, =null): Name of the layer for futur redirection.
     +      _state (Vector!T, =null): Initial state of the result vector.
     +      _in (size_t[], =null): A list of layers' name for the layer to take its
     +                        inputs. If empty, the last known layer will be took.
     +      _to (size_t[], =null): A list of the layers which will take their input from this layer.
     +/
    auto gaussian(in size_t _dim_out=0,
                 in string _name=null,
                 Vector!T _state=null,
                 in string[] _in=null,
                 in string[] _to=null)
    {
        return this.func!"gaussian"(_dim_out, _name, _state, _in, _to);
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
        /++ This method allocate memory for the all neural network.
         +/

        // First we want to know the size of the total array.
        // We sum the size of each layer.
        size_t total_size = 0;
        foreach(tmp_l; layers)
            if (!(tmp_l is null))
                total_size += tmp_l.size;

        serialized_data = new T[total_size];

        // We copy the values in the new array and replace the old
        // ones by a reference to the new array. 
        size_t _index = 0;
        foreach(tmp_l; layers)
            if (!(tmp_l is null))
                tmp_l.takeOwnership(serialized_data, _index);
    }

    /// Apply the NeuralNetwork to the vector and change the NN state if needed.
    Vector!T compute(in Vector!T _v)
    {
        buffers[0].v[] = _v.v[];

        results[0].v[] = _v.v[];

        foreach(cur_id; 1 .. id)
        {
            // If there is only one input,
            // we just pass it to the layer for computation. 
            if (input_layers[cur_id].length == 1)
                layers[cur_id].apply(results[input_layers[cur_id][0]], results[cur_id]);
            else {
                // Else we need to create a temporary vector to sum all the inputs.
                buffers[cur_id].v[] = to!T(0);
                foreach(tmp_id; input_layers[cur_id])
                    buffers[cur_id] += results[tmp_id];

                // Finally, we compute the tmp vector using the layer.
                layers[cur_id].apply(buffers[cur_id], results[cur_id]);
            }
        }
        return results[$-1];
    }
}
unittest {
    write("Unittest: NeuralNetwork ... ");

    // Neural Network 1: Simple linear layer neural network.
    // Neural Network 2: Two linear layer with a recurrence.
    
    writeln("A");
    {
        // Initialize the neural network.
        // At this point, we have the identity func.
        auto nn = new NeuralNetwork!float(4);
        nn.identity;

        // Vector of L2 norm = 1.
        auto v = new Vector!float([0.5, 0.0, -0.5, 0.7071068]);

        // w should be equal to v.
        auto w = nn.compute(v);
        w -= v;

        assert(w.norm!"L2" <= 0.0001);

        // We add a linear Layer of shape (6, 4).
        nn.linear!(Matrix!float)(6, false, 1.0, "L1");
        w = nn.compute(v);

        writeln("A3");
        // Hence, the resulting vector should have length 6.
        assert(w.length == 6);

        // We add some complexity: the layer take the user's input and the ouput of "L1"
        // and return its output to "L1" (And so create a rnn-like structure) and to
        // the output (by default, the result of the last layer).
        nn.linear!(Matrix!float)(4, false, 1.0, "L2", null, null, ["L1"]);
        writeln("A3a");
        w = nn.compute(v);
        writeln("A3b");
        auto z = nn.compute(v);
        writeln("A4");

        // Now we reconstruct what we think the neural network should compute.
        // w
        auto w_bis = (cast(Matrix!float) nn.layers[1].params[0]) * v;
        w_bis *= (cast(Matrix!float) nn.layers[2].params[0]);

        auto hidden = w_bis.dup;
        w_bis -= w;

        assert(w_bis.norm!"L2" <= 0.0001);

        writeln("A5");
        // z
        auto z_bis = hidden;
        z_bis += v;
        z_bis = (cast(Matrix!float) nn.layers[1].params[0]) * z_bis;
        z_bis = (cast(Matrix!float) nn.layers[2].params[0]) * z_bis;
        
        z_bis -= z;
        assert(z_bis.norm!"L2" <= 0.0001);
    }

    writeln("B");
    // Neural Network 1: Simple linear layer + softmax.
    // Neural Network 2: RNN: linear + relu + linear (+backlink) + linear + softmax.
    {

        auto vec = new Vector!real(5, 1.0);

        // NN1
        auto nn1 = new NeuralNetwork!real(5);
        nn1.linear(5)
           .softmax(5);


        auto w = nn1.compute(vec);

        assert(abs(1 - w.norm!"L1") <= 0.0001);


        // NN2
        auto nn2 = new NeuralNetwork!real(5);
        nn2.linear(5, true)
           .recurrent()
           .linear(5, true)
           .softmax();


        auto s = nn2.results[3].dup; 

        auto nn3 = new NeuralNetwork!real(5);
        nn3.linear(5, true, 1.0)
           .relu(5, "Rec_in")
           .linear(5, false, 1.0, null, s, null, ["Rec_in"])
           .linear(5, true)
           .softmax();


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

    writeln("C");
    // Neural Network 1: linear + softmax
    // Neural Network 2: linear + softmax + linear + recurrent + linear + norm!"L2"^-1
    {
        auto nn1 = new NeuralNetwork!real(4);
        nn1.linear(4, true)
           .softmax()
           .serialize;

        auto v = new Vector!real([1.0, -1.0, 0.0, 1.0]);

        assert(nn1.serialized_data.length == 4*4+4);

        foreach(i; 0 .. nn1.serialized_data.length)
            nn1.serialized_data[i] = 1.0;

        v = nn1.compute(v);

        auto res = new Vector!real([0.25, 0.25, 0.25, 0.25]);
        res -= v;

        assert(res.norm!"L2" <= 0.0001);

        auto nn2 = new NeuralNetwork!real(4);
        nn2.linear(5)
           .softmax()
           .linear(5)
           .recurrent()
           .linear(4)
           .func!"
                b.v[] = _v.v[];
                b /= b.norm!\"L2\";"()
           .serialize;

        auto w = nn2.compute(v);

        // Test if the delegate works as needed.
        enforce(abs(1 - w.norm!"L2") <= 0.0001, "Norm of "~to!string(w.v)~" is not 1.0");

        // We set each weights to one.
        foreach(i; 0 .. nn2.serialized_data.length)
            nn2.serialized_data[i] = 1.0;

        w = nn2.compute(v);
        
        // We then apply each step separately with weights 1 using the fact that v = [1, -1, 0, 1].

        //Apply linear(5): [1, 1, 1, 1, 1]

        //Apply softmax: [0.2, 0.2, 0.2, 0.2, 0.2]

        //Apply recurrent (relu(1 + previous vector)): [1.2, 1.2, 1.2, 1.2, 1.2]

        //linear(4): [4.8, 4.8, 4.8, 4.8]

        //Divide by norm!"L2": [0.5, 0.5, 0.5, 0.5]

        auto true_res = new Vector!real([0.5, 0.5, 0.5, 0.5]);
        true_res -= w;

        assert(true_res.norm!"L2" <= 0.0001);
    }

    writeln("D");
    // Neural Networks: relu / logistic / gaussian / identity
    //                / tanh / arctan / softsign  / softplus / sin / binary 
    {
        auto vec = new Vector!real([-1.0, -0.5, 0.5, 1.0]);
        
        auto nn_relu = new NeuralNetwork!real(4);
        nn_relu.relu();
        auto res_relu = new Vector!real([0.0, 0.0, 0.5, 1.0]);
        res_relu -= nn_relu.compute(vec);
        assert(res_relu.norm!"L2" <= 0.000001);

        auto nn_logistic = new NeuralNetwork!real(4);
        nn_logistic.logistic();
        auto res_logistic = new Vector!real([1.0/(1.0 + exp( 1.0)), 1.0/(1.0 + exp( 0.5)),
                                             1.0/(1.0 + exp(-0.5)), 1.0/(1.0 + exp(-1.0))]);
        res_logistic -= nn_logistic.compute(vec);
        assert(res_logistic.norm!"L2" <= 0.000001);

        auto nn_gaussian = new NeuralNetwork!real(4);
        nn_gaussian.gaussian();
        auto res_gaussian = new Vector!real([exp(-1.0), exp(-0.25), exp(-0.25), exp(-1.0)]);
        res_gaussian -= nn_gaussian.compute(vec);
        assert(res_gaussian.norm!"L2" <= 0.000001);

        auto nn_identity = new NeuralNetwork!real(4);
        nn_identity.identity();
        auto res_identity = new Vector!real([-1.0, -0.5, 0.5, 1.0]);
        res_identity -= nn_identity.compute(vec);
        assert(res_identity.norm!"L2" <= 0.000001);

        auto nn_tanh = new NeuralNetwork!real(4);
        nn_tanh.tanh();
        auto res_tanh = new Vector!real([tanh(-1.0), tanh(-0.5), tanh(0.5), tanh(1.0)]);
        res_tanh -= nn_tanh.compute(vec);
        assert(res_tanh.norm!"L2" <= 0.000001);

        auto nn_arctan = new NeuralNetwork!real(4);
        nn_arctan.arctan();
        auto res_arctan = new Vector!real([atan(-1.0), atan(-0.5), atan(0.5), atan(1.0)]);
        res_arctan -= nn_arctan.compute(vec);
        assert(res_arctan.norm!"L2" <= 0.000001);

        auto nn_softsign = new NeuralNetwork!real(4);
        nn_softsign.softsign();
        auto res_softsign = new Vector!real([-0.5, -1.0/3.0, 1.0/3.0, 0.5]);
        res_softsign -= nn_softsign.compute(vec);
        assert(res_softsign.norm!"L2" <= 0.000001);

        auto nn_softplus = new NeuralNetwork!real(4);
        nn_softplus.softplus();
        auto res_softplus = new Vector!real([log(1+exp(-1.0)), log(1+exp(-0.5)), log(1 + exp(0.5)), log(1 + exp(1.0))]);
        res_softplus -= nn_softplus.compute(vec);
        assert(res_softplus.norm!"L2" <= 0.000001);

        auto nn_sin = new NeuralNetwork!real(4);
        nn_sin.sin();
        auto res_sin = new Vector!real([sin(-1.0), sin(-0.5), sin(0.5), sin(1.0)]);
        res_sin -= nn_sin.compute(vec);
        assert(res_sin.norm!"L2" <= 0.000001);

        auto nn_binary = new NeuralNetwork!real(4);
        nn_binary.binary();
        auto res_binary = new Vector!real([0.0, 0.0, 1.0, 1.0]);
        res_binary -= nn_binary.compute(vec);
        assert(res_binary.norm!"L2" <= 0.000001);
    }

    writeln("E");
    // Neural Network: Diamond structure with identity function only => implement f(x) = 2*x !
    {
        auto nn = new NeuralNetwork!real(6);
        nn.identity(0, "top")
          .identity(0, "bottom", null, ["input"])
          .identity(0, "output", null, ["bottom", "top"]);

        auto vec = new Vector!real(6, 1.0);
        auto res = nn.compute(vec);
        vec += vec;
        res -= vec;
        assert(res.norm!"L2" <= 0.000001);
    }

    // Bad naming of layers
    assertThrown((new NeuralNetwork!real(5).linear(0, false, 1.0, "blue").linear(7, true, 2.0, "blue")));

    writeln("Done.");
}
