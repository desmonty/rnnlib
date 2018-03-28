module source.Layer;

import std.complex;
import std.conv: to;
import std.functional: toDelegate;
import std.string : startsWith;

import source.Matrix;
import source.Parameter;

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

    /// Parameter, Layer-specific
    Parameter[] params = null;

    /// function applied to the vector.
    Vector!(S,T) delegate(in Vector!(S,T), in Parameter[]) func;

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
    write("Unittest MatrixLayers ... ");

    write("Done.\n");
}

/+ This layer can implement any function that take as input a
   Vector!(S,T) and return another Vector!(S,T).
 +/
class FunctionalLayer(S,T) : Layer!(S,T)
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
                static if (!T.stringof.startsWith("Complex")) {
                    this(
                        delegate(T val) {
                            if (val > 0)
                                return val;
                            return 0;
                        }
                    );
                    break;
                }
                // else with use modRelu by default.
            case "modRelu":
                static if (T.stringof.startsWith("Complex")) {
                    if (size_in == 0)
                        throw new Exception("'size_in' must be greater than zero
                                             when using 'modRelu'.");
                    isLearnable = true;
                    params = new Parameter[1];
                    params[0] = new Vector!(S,Tc)(size_in, 1.0);
                    
                    func = 
                        delegate(in Vector!(S,T) _v, in Parameter[] _p) {
                            auto absv = _v[0].abs;
                            auto tmp = absv;
                            auto res = _v.dup;
                            foreach(i; 0 .. _v.length) {
                                absv = _v[i].abs;
                                tmp = absv + (cast(Vector!(S,_v.Tc)) _p[0])[i];
                                if (tmp > 0) {
                                    res[i] = tmp*_v[i]/absv;
                                }
                                else {
                                    res[i] = complex(cast(_v.Tc) 0);
                                }
                            }
                            return res;
                        };
                }
                else
                    throw new Exception("The 'modRelu' function can only
                                         be used with complex number.");
                break;
            case "softmax":
                static if (!T.stringof.startsWith("Complex"))
                    this(
                        delegate(in Vector!(S,T) _v) {
                            T s = 0;
                            auto res = _v.dup;
                            foreach(i; 0 .. _v.length) {
                                res[i] = exp(v[i]);
                                s += res[i];
                            }
                            foreach(i; 0 .. _v.length)
                                res[i] /= s;
                            return res;
                        }
                    );
                else
                    throw new Exception("You will have to define your own
                                         complex-valued softmax.");
                break;
            default:
                assert(0, easyfunc ~ ": Unknown keyword. You can use one of the"
                                   ~ " following:\n"
                                   ~ " - 'relu' (for real-valued vectors)\n"
                                   ~ " - 'modRelu' (for complex-valued vectors)\n"
                                   ~ " - 'softmax' (for real-valued vectors)\n");
        }
    }

    // The function to apply to the vector. Can be anything. DELEGATE
    this(Vector!(S,T) delegate(in Vector!(S,T)) _func)
    {
        func = delegate(in Vector!(S,T) _v, in Parameter[] _p=null) {
            if (_p !is null)
                throw new Exception("Parameter not allowed for 'func'.");
            return _func(_v);
        };
    }

    // The function to apply to the vector. Can be anything. FUNCTION
    this(Vector!(S,T) function(in Vector!(S,T)) _func)
    {
        this(toDelegate(_func));
    }

    // Create an element-wise function that apply a provided
    // function to a vector. DELEGATE
    this(T delegate(T) _func)
    {
        func = delegate(in Vector!(S,T) _v, in Parameter[] _p=null) {
            if (_p !is null)
                throw new Exception("Parameter not allowed for 'func'.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = _func(_v[i]);
            return res;
        };
    }

    // Create an element-wise function that apply a provided
    // function to a vector. FUNCTION
    this(T function(T) _func)
    {
        this(toDelegate(_func));
    }

    this(in S size)
    {
        this(delegate(in Vector!(S,T) _v) {
                if (_v.length != size)
                    throw new Exception("Size mismatch in FunctionalLayer:\n"
                                       ~"Size of the FunctionalLayer: "
                                       ~to!string(size)~"\n"
                                       ~"Size of the Vector: "
                                       ~to!string(_v.length)~"\n");
                auto res = _v.dup;
                return res;
            }
        );
    }

    override
    Vector!(S,T) compute(in Vector!(S,T) _v)
    {
        return func(_v, params);
    }
}
unittest {
    write("Unittest FunctionalLayer ... ");

    uint len = 1024;
    alias Vec = Vector!(uint, Complex!real);
    alias Fl = FunctionalLayer!(uint, Complex!real);

    Vec blue(Vec _v) pure {
        auto res = _v.dup;
        res *= complex(4);
        return res;
    }

    auto v = new Vec(len, 1.0);
    auto f1 = new Fl(len);
    auto f2 = new Fl(&std.complex.cos);
    auto f3 = new Fl(&blue);

    auto v1 = f1.compute(v);
    auto v2 = f2.compute(v);
    auto v3 = f3.compute(v);
    writeln(v1.v);
    writeln(v2.v);
    writeln(v3.v);

    v1 -= v;
    v3 /= complex(4);
    v3 -= v;

    assert(v1.norm!"L2" <= 0.001);
    assert(v3.norm!"L2" <= 0.001);

    write("Done.\n");
}

/+
class RecurrentLayer : Layer
+/
