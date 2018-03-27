module source.Layers;

import std.complex;
import std.functional: toDelegate;
import std.string : startsWith;

import source.Matrix;
import source.Parameters;

version(unittest)
{
    import std.stdio : writeln, write;
}

/+  The layers of the Neural Networks.

    Basically, each layer can be seen as a function, which take a vector and
    return another vecotr of a possibly different length. Those functions have
    parameters (matrix, bias, ..) which are specific to each kind of layer
    (linear, recurrent, ..). All the logic of how these layers are assembled is
    in the NeuralNet object.

    If they are trained with an evolutionary algorithm (Which is the primary
    goal), we will need to have some function to handle *mutation* and
    *crossover*. This should be enough for many of the optimization we will
    implement (PSA, GA, RS, ES, and other obscure acronym...).

    The gradient of the layer will be difficult to compute due to the will to
    play with heavily recurrent networks (which is not the common cas because
    of the use of gradient-based optimization). However, it would be very
    interesting to know the gradient of the NeuralNet and could be investigated
    in this project.

    In practice, each layers must implement two methods:
        - apply
        - compute
    Both apply the function implemented by the layer 


    TODO:
        - Shared parameter = convnet
        - share_parameter in NeuralNet between layer

        - REFACTOR: idea
            .The "layer" object should hold an array of parameters and a delegate
            of the following form: {Vector delegate(Vector, Parameter)}
            .Matrix/function layer should be easy to implement in this context
            .It should provide every one with a "general" enough object to create
             convnet (shared parameters), 
 +/

abstract class Layer(S,T)
{
    static if (T.stringof.startsWith("Complex"))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    /// Name of the layer.
    string name;

    /// Sizes
    S size_in;
    S size_out;

    /// Parameters, Layer-specific
    Parameters[] params;

    /// function applied to the vector.
    Vector!(S,T) delegate(in Vector!(S,T), in Parameter[]) pure func;

    /// Used by the optimizer to know if it must optimize the layer.
    bool isLearnable = false;

    /// Used
    void set_name(string _name)
    {
        if (_name !is null)
            name = _name;
    }

    abstract Vector!(S,T) compute(in Vector!(S,T));
}

/+ This layer implement a simple linear matrix transformation
   applied to the vector followed by adding a bias vector
   (which can be turner off). 
 +/
class MatrixLayer(S,T) : Layer!(S,T)
{

}
unittest {
    write("Unittest MatrixLayers Abstract ... ");

    write("Done.\n");
}

/+ This layer can implement any function that take as input a
   Vector!(S,T) and return another Vector!(S,T).
 +/
class FunctionLayer(S,T) : Layer!(S,T)
{
    /+ This implements common functions most will want to use
       like:
            -SoftMax/SoftPlus
            -relu/modRelu
    TODO:
        softmax
        softplus
        add parameter in delegate
     +/
    this(string easyfunc, in S size_in=0)
    {
        switch (easyfunc)
        {
            case "relu":
                if (!T.stringof.startsWith("Complex")) {
                    this(
                        delegate(T val) pure {
                            if (val > 0)
                                return val;
                            return 0;
                        }
                    );
                    break;
                }
                // else with use modRelu by default.
            case "modRelu":
                if (!T.stringof.startsWith("Complex"))
                    throw new Exception("The 'modRelu' function can only
                                         be used with complex number.");

                if (size_in == 0)
                    throw new Exception("'size_in' must be greater than zero
                                         when using 'modRelu'.");
                is_learnable = true;
                parameters = new Vector!(S,T)(size_in, 1.0);
                
                this(
                    delegate(in Vector!(S,T) v) pure {
                        auto tmp = v[0];
                        auto absv = v[0].abs;
                        auto res = v.dup;
                        foreach(i; 0 .. v.length) {
                            absv = v[i].abs;
                            tmp = absv + parameters[i];
                            if (tmp > 0) {
                                res[i] = tmp*v[i]/absv;
                            }
                            else {
                                res[i] = complex(0);
                            }
                        }
                    }
                );
                break;
            case "softmax":
                this(
                    delegate(in Vector!(S,T)) pure {
                    Tc s = 0;
                    }
                );
                break;
            default:
                try {
                    // This should handle most of the case : tanh, cos, sin, ...
                    this(
                        delegate(in Vector!(S,T) v) pure {
                            foreach(i; 0 .. v.length)
                            mixin("v[i] = "~easyfunc~"(v[i]);");
                        }
                    );
                }
                catch (Exception e) {
                    assert(0, easyfunc ~ ": Unknown function. Implement it !");
                }
        }
    }

    // The function to apply to the vector. Can be anything. DELEGATE
    this(Vector!(S,T) delegate(in Vector!(S,T)) pure _func)
    {
        func = delegate(in Vector!(S,T) _v, in Parameters[] _p=null) pure {
            if (_p !is null)
                throw new Exception("Parameters not allowed for 'func'.");
            return _func(_v);
        };
    }

    // The function to apply to the vector. Can be anything. FUNCTION
    this(Vector!(S,T) function(in Vector!(S,T)) pure _func)
    {
        this(toDelegate(_func));
    }

    // Create an element-wise function that apply a provided
    // function to a vector. DELEGATE
    this(T delegate(T) pure _func)
    {
        func = delegate(in Vector!(S,T) _v, in Parameters[] _p=null) pure {
            if (_p !is null)
                throw new Exception("Parameters not allowed for 'func'.");
            auto res = _v.dup;
            foreach(i; 0 .. v.length)
                res[i] = _func(v[i]);
            return res;
        };
    }

    // Create an element-wise function that apply a provided
    // function to a vector. FUNCTION
    this(T delegate(T) pure _func)
    {
        this(toDelegate(_func));
    }

    override
    Vector!(S,T) compute(in Vector!(S,T) v, in Parameters _p)
    {
        return func(res);
    }
}
unittest {
    write("Unittest FunctionLayers Abstract ... ");

    uint len = 1024;
    auto v = new Vector!(uint, Complex!real)(len, 1.0);
    auto f = new FunctionLayers!(uint, Complex!real)(len);

    write("Done.\n");
}

/+
class RecurrentLayer : Layer
+/
