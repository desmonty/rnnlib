module source.Layer;

import std.algorithm;
import std.complex;
import std.conv: to;
import std.exception: assertThrown, enforce;
import std.functional: toDelegate;
import std.math;
import std.range;
import std.string : startsWith;

import source.Matrix;
import source.Parameter;

version(unittest)
{
    import std.stdio : writeln, write;
    import core.exception;
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

    In practice, each layers must implement one method:
        - compute
    which apply the layer to the vector and return the result.


    TODO:
        - convnet
        - share_parameter in NeuralNet between layer

 +/

abstract class Layer(T)
{
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    /// Name of the layer.
    string name;

    /// Sizes
    size_t size_in;
    size_t size_out;

    /// Parameter, Layer-specific
    Parameter[] params = null;

    /// Used
    @nogc @safe pure 
    void set_name(string _name)
    {
        if (_name !is null)
            name = _name;
    }

    abstract Vector!T compute(Vector!T);
}

/+ This layer implement a simple linear matrix transformation
   applied to the vector.
 +/
class MatrixLayer(T) : Layer!T
{
    @safe pure
    this(MatrixAbstract!T _M, in size_t _size_bias=0)
    {
        size_t num_param = 1;
        if (_size_bias)
            num_param = 2;

        params = new Parameter[num_param];
        params[0] = _M;

        if (_size_bias){
            auto vec = new Vector!T(_size_bias);
            vec.v[] = to!T(0);
            params[1] = vec;
        }
    }
    
    override
    Vector!T compute(Vector!T _v)
    {
        auto res = cast(MatrixAbstract!T) params[0] * _v;
        if (params.length > 1)
            res += cast(Vector!T) params[1];
        return res;
    }
}
unittest {
    write("Unittest: Layer: Matrix ... ");

    auto v = new Vector!real(4, 1.0);

    auto r = new ReflectionMatrix!real(2, 1.0);
    auto d = new DiagonalMatrix!real(2, 2.0);
    auto m = new BlockMatrix!real(4, 4, 2, [r, d], true);

    auto l = new MatrixLayer!real(m);

    auto w = v.dup;

    auto r1 = l.compute(v);
    auto r2 = m * v;

    r2 -= r1;
    v -= w;

    assert(r2.norm!"L2" <= 0.00001);
    assert(v.norm!"L2" <= 0.00001);

    write("Done.\n");
}

/+ This layer implement a simple linear matrix transformation
   applied to the vector.
 +/
class BiasLayer(T) : Layer!T
{
    @safe
    this(const size_t size_in, const Tc randomBound)
    {
        params = new Parameter[1];
        params[0] = new Vector!T(size_in, randomBound);
    }
    
    override
    Vector!T compute(Vector!T _v)
    {
        auto res = _v.dup;
        res += to!(Vector!T)(params[0]);
        return res;
    }

}
unittest {
    write("                 Bias ... ");

    auto v = new Vector!(Complex!real)(4, 0.5);

    auto b1 = new BiasLayer!(Complex!real)(4, 0);
    auto b2 = new BiasLayer!(Complex!real)(4, 1.0);

    auto b = to!(Vector!(Complex!real))(b2.params[0]);

    auto w = v.dup;
    w += b;
    auto k = b1.compute(v);
    auto l = b2.compute(v);

    l -= w;
    k -= v;

    assert(l.norm!"L2" <= 0.0001);
    assert(k.norm!"L2" <= 0.0001);

    write("Done.\n");
}

/+ This layer can implement any function that take as input a
   Vector!T and return another Vector!T.
 +/
class FunctionalLayer(T) : Layer!T
{
    /++This implements common functions that are not implemented already.
     + It includes the following:
     +      -SoftMax
     +      -relu
     +      -modRelu
     +/

    private {
        /// function applied to the vector.
        Vector!T delegate(Vector!T, Parameter[]) func;
    }


    this(string easyfunc, in size_t size_in=0)
    {
        switch (easyfunc)
        {
            case "relu":
                static if (!is(Complex!T : T)) {
                    func =
                        delegate(Vector!T _v, Parameter[] _p) {
                            auto res = _v.dup;
                            foreach(i; 0 .. _v.length)
                                if (res[i] < 0) res[i] = 0;
                            return res;
                        };
                    break;
                }
                // else with use modRelu by default.
            case "modRelu":
                static if (is(Complex!T : T)) {
                    enforce(size_in != 0, "'size_in' must be greater than zero
                                            when using 'modRelu'.");
                    params = new Parameter[1];
                    params[0] = new Vector!Tc(size_in, 1.0);
                    
                    func =
                        delegate(Vector!T _v, Parameter[] _p) {
                            auto absv = _v[0].abs;
                            auto tmp = absv;
                            auto res = _v.dup;
                            foreach(i; 0 .. _v.length) {
                                absv = _v[i].abs;
                                tmp = absv + (cast(Vector!Tc) _p[0])[i];
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
                static if (!is(Complex!T : T))
                    func =
                        delegate(Vector!T _v, Parameter[] _p) {
                            T s = 0;
                            auto res = _v.dup;
                            foreach(i; 0 .. _v.length) {
                                res.v[i] = exp(_v[i]);
                                s += res[i];
                            }
                            foreach(i; 0 .. _v.length)
                                res.v[i] /= s;
                            return res;
                        };
                else
                    throw new Exception("You will have to define your own
                                         complex-valued softmax.");
                break;
            default:
                throw new Exception(easyfunc
                                   ~ ": Unknown keyword. You can use one of the"
                                   ~ " following:\n"
                                   ~ " - 'relu' (for real-valued vectors)\n"
                                   ~ " - 'modRelu' (for complex-valued vectors)\n"
                                   ~ " - 'softmax' (for real-valued vectors)\n");
        }
    }

    // The function to apply to the vector. Can be anything. DELEGATE
    @safe pure
    this(Vector!T delegate(Vector!T) _func)
    {
        func = delegate(Vector!T _v, Parameter[] _p) {
            return _func(_v);
        };
    }

    // The function to apply to the vector. Can be anything. FUNCTION
    pure
    this(Vector!T function(Vector!T) _func)
    {
        this(toDelegate(_func));
    }

    // Create an element-wise function that apply a provided
    // function to a vector. DELEGATE
    @safe pure
    this(T delegate(T) _func)
    {
        func = delegate(Vector!T _v, Parameter[] _p) {
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = _func(_v[i]);
            return res;
        };
    }

    // Create an element-wise function that apply a provided
    // function to a vector. FUNCTION
    pure
    this(T function(T) _func)
    {
        this(toDelegate(_func));
    }

    @safe pure
    this(in size_t size)
    {
        this(delegate(Vector!T _v) {
                enforce(_v.length == size, "Size mismatch in FunctionalLayer:\n"
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
    Vector!T compute(Vector!T _v)
    {
        return func(_v, params);
    }
}
unittest {
    write("                 Functional ... ");

    alias Vec = Vector!(Complex!double);
    alias Fl = FunctionalLayer!(Complex!double);

    Vec blue(Vec _v) pure {
        auto res = _v.dup;
        res.v[] *= complex(4.0);
        return res;
    }

    auto ff = function(Vec _v) pure {
        auto res = _v.dup;
        res.v[] *= complex(4.0);
        return res;
    };

    uint len = 4;
    double pi = 3.1415923565;

    auto v = new Vec([complex(0.0), complex(1.0), complex(pi), complex(-1.0)]);
    auto e = new Vec([complex(0.0)]);

    // Length initialization.
    auto f1 = new Fl(len);
    auto v1 = f1.compute(v);
    v1 -= v;
    assert(v1.norm!"L2" <= 0.001);

    // Test how to use template function.
    // Must compile;
    auto f2 = new Fl(&cos!double);
    auto v2 = f2.compute(v);

    // Parameter-less function & delegate initialization.
    auto f3 = new Fl(ff);
    auto v3 = f3.compute(v);
    auto f4 = new Fl(&blue);
    auto v4 = f4.compute(v);

    v3 -= v4;
    assert(v3.norm!"L2" <= 0.001);

    v4.v[] /= complex(4.0);
    v4 -= v;
    assert(v4.norm!"L2" <= 0.001);


    // modRelu function.
    auto w = v.dup;
    w[2] = complex(0.5);
    auto f5 = new Fl("modRelu", 4);
    (cast(Vector!double) f5.params[0])[0] = -0.9;
    (cast(Vector!double) f5.params[0])[1] = -0.9;
    (cast(Vector!double) f5.params[0])[2] = -0.9;
    (cast(Vector!double) f5.params[0])[3] = -0.9;
    auto v5 = f5.compute(w);
    assert(abs(v5.sum) < 0.0001);

    // vector 'e' doesn't have the right length.
    assertThrown(f1.compute(e));
    // modRelu must be given the vector's size to create parameters.
    assertThrown(new Fl("modRelu"));
    // relu takes only real-valued vectors.
    assertThrown(new FunctionalLayer!(Complex!double)("relu"));
    // softmax takes only real-valued vectors.
    assertThrown(new FunctionalLayer!(Complex!double)("softmax"));
    // modRelu takes only complex-valued vectors.
    assertThrown(new FunctionalLayer!double("modRelu"));
    // Incorrect function name.
    assertThrown(new FunctionalLayer!real("this is incorrect."));

    auto vr = new Vector!double([0.0, 1.0, pi, -1.0]);

    // relu function.
    auto f6 = new FunctionalLayer!double("relu");
    auto vr6 = f6.compute(vr);
    assert(abs(vr6.sum - 1.0 - pi) <= 0.01);

    // softmax function.
    auto f7 = new FunctionalLayer!double("softmax");
    auto vr7 = f7.compute(vr);
    assert(abs(vr7.sum - 1.0) <= 0.001);

    // set the name of a layer.
    f1.set_name("f1");
    assert(f1.name == "f1");

    auto vv = new Vector!double(100, 0.2);

    write(" Done.\n");
}
