module source.Layers;

/+  The layers of the Neural Networks.

    Basically, each layer can be seen as a function,
    which take a vector and return another vecotr of
    a possibly different length.
    Those functions have parameters (matrix, bias, ..)
    which are specific to each kind of layer (linear, recurrent, ..).
    All the logic of how these layers are assembled is
    in the NeuralNet object.

    If they are trained with an evolutionary algorithm
    (Which is the primary goal), we will need to have
    some function to handle *mutation* and *crossover*.
    This should be enough for many of the optimization
    we will implement (PSA, GA, RS, ES, and other obscure
    acronym...).

    The gradient of the layer will be difficult to
    compute due to the will to play with heavily recurrent
    networks (which is not the common cas because of
    the use of gradient-based optimization).
    However, it would be very interesting to know the gradient
    of the NeuralNet and could be investigated in this project.


 +/

abstract class Layer(S,T)
{
    static if (T.stringof.startsWith("Complex"))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    /// Name of the layer.
    string name;

    /// Type of the layer. 
    string typeId;

    S size_in;
    S size_out;

    void set_name(string _name)
    {
        if (_name !is null)
            name = _name;
    }

    abstract Vector!(S,T) apply();
}

/+ This layer implement a simple linear matrix transformation
   applied to the vector followed by adding a bias vector
   (which can be turner off). 
 +/
class LinearLayer(S,T) : Layer!(S,T)
{
    MatrixAbstract!(S,T) W;
    Vector!(S,T) bias;
    bool keep_bias;

    this(ref MatrixAbstract!(S,T) _W, bool _keep_bias = true)
    {
        W = _W;
        keep_bias = _keep_bias;
    }

    /// Random initialization of the matrix and vector.
    void init(in S _size_in, in S _size_out)
    {
        if (keep_bias) {
            bias = new Vector!(S,T)(size_out);
            bias.v[] = 0;
        }
    }

    /// Apply the function implemented by the layer to the vector.
    override
    auto apply(Vector!(S,T) vec)
    {
        auto tmp_vec = new Vector!(S,T)(size_in);
        tmp *= W;
        if (keep_bias)
            tmp += bias;
        return tmp;
    }
}
unittest {

}

/+ This layer can implement any function that take as input a
   Vector!(S,T) and return another Vector!(S,T).
   WARNING : Right now, the function will change the vector
   it is given in the case of a element-wise tranformation.
 +/
class FunctionLayer(S,T) : Layer!(S,T)
{
    Vector!(S,T) delegate(Vector!(S,T) v) func;
    Vector!(S,T) parameters;
    bool is_learnable = false;

    this(string easyfunc, in S size_in=0)
    {
        switch (easyfunc)
        {
            case "relu":
                if (!T.stringof.startsWith("Complex")) {
                    this( (T val) => val if val > 0 else 0);
                    break;
                }
                // else with use modRelu by default.
            case "modRelu":
                if (!T.stringof.startsWith("Complex"))
                    throw new Exception("the 'modRelu' function can only be used with complex number.");

                is_learnable = true;
                parameters = new Vector!(S,T)(size_in, 1.0);
                func = delegate(Vector!(S,T) v) {
                        auto tmp = v[0];
                        foreach(i; 0 .. v.length) {
                            tmp = v[i].abs + parameters[i];
                            v[i] = tmp*v[i]/v[i].abs if tmp > 0 else 0;
                        }
                };
                break;
            default:
                try {
                    // This should handle most of the case : tanh, cos, sin, sqrt, atanh, expi 
                    func = delegate(Vector!(S,T) v) {
                        foreach(i; 0 .. v.length)
                            mixin("v[i] = "~easyfunc~"(v[i]);");
                    };
                }
                catch (Exception e) {
                    assert(0, easyfunc ~ " is not a known function. Implement it !");
                }
        }
    }

    // The function to apply to the vector. Can be anything.
    this(Vector!(S,T) delegate(Vector!(S,T)) _func)
    {

    }

    // Create an element-wise function that apply a provided
    // function to a vector.
    auto
    this(T delegate(T) _func)
    {
        func =
        delegate(Vector!(S,T) v) {
            foreach(i; 0 .. v.length)
                v[i] = _func(v[i]);
        };
    }

    override
    auto apply(Vector!(S,T) v)
    {
        return func(v);
    }
}
unittest {
    
}

/+
class RecurrentLayer : Layer
+/
