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
import source.Utils;

version(unittest)
{
    import std.stdio : writeln, write;
    import core.exception;
}



/++ The layers of the Neural Networks.
 +
 +  Basically, each layer can be seen as a function, which take a vector and
 +  return another vector of a possibly different length. Those functions have
 +  parameters (matrix, bias, ..) which are specific to each kind of layer
 +  (linear, recurrent, ..). All the logic of how these layers are assembled is
 +  in the NeuralNet object.
 +
 +  If they are trained with an evolutionary algorithm (Which is the primary
 +  goal), we will need to have some function to handle *mutation* and
 +  *crossover*. This should be enough for many of the optimization we will
 +  implement (PSA, GA, RS, ES, and other obscure acronym...).
 +
 +  The gradient of the layer will be difficult to compute due to the will to
 +  play with heavily recurrent networks (which is not the common cas because
 +  of the use of gradient-based optimization). However, it would be very
 +  interesting to know the gradient of the NeuralNet and could be investigated
 +  in this project.
 +
 +  In practice, each layers must implement one method:
 +      - compute
 +  which apply the layer to the vector and return the result.
 +
 +
 +  TODO:
 +      - convnet
 +      - share_parameter in NeuralNet between layer
 +
 +/
abstract class Layer(T)
{
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    /// Number of parameter in the layer.
    size_t size;

    /// Parameter, Layer-specific
    Parameter[] params = null;

    abstract Vector!T compute(Vector!T);
    abstract void takeOwnership(ref T[], ref size_t);
}

/+ This layer implement a simple linear matrix transformation
 + applied to the vector.
 +
 + M (type): Type of the matrix to be used in the layer.
 + T (type): Type of the matrix.
 +
 + WE ASSUME NOBODY WE'LL USE ANYTHING ELSE THAN A MATRIX FOR "M".
 + TODO: use some kind of limitation to make sure M is a known matrix
 + (std.variant, enum, set ...).
 +
 +/
class MatrixLayer(Mtype : M!T, alias M, T) : Layer!T
{
    // If M is BlockMatrix, T will be of the form N!U with N: matrixType and U: arithmetic type
    static if (__traits(compiles, BlockMatrix!T)) {
        mixin("alias TypeValue = "~T.stringof.split("(")[1][0 .. ($-1)]~";"); 
        static if (is(TypeValue: Complex!TypeValue))
            mixin("alias Tc = "~(TypeValue.stringof[8 .. $])~";");
    }
    else
        alias TypeValue = T;

    pragma(msg, Mtype.stringof);
    pragma(msg, M.stringof);
    pragma(msg, T.stringof);
    pragma(msg, TypeValue.stringof);
    pragma(msg, Tc.stringof);

    this()
    {
        // TODO: Add check to see if Mtype is a Matrix*!T
    }

    this(in Mtype _M)
    {
        this();
        static if (__traits(compiles, BlockMatrix!T))
            size = size_blocks * size_blocks * num_blocks;
        else
            size = _M.rows * _M.cols;
        // We keep a duplicate of the matrix.
        params = new Parameter[1];
        params[0] = _M.dup;
    }

    this(in Mtype _M, in Vector!TypeValue _v)
    {
        if (_v){
            this();
            enforce(_M.cols == _v.length, "Matrix / Bias dimensions mismatch.");

            static if (Mtype.stringof[0 .. 5] == "Block")
                size = size_blocks * size_blocks * num_blocks + _v.length;
            else
                size = _M.rows * _M.cols + _v.length;

            params = new Parameter[2];
            params[0] = _M.dup;
            params[1] = _v.dup;
        }
        else
            this(_M);
    }

    this(in size_t[2] _dim,
         in bool _bias=false,
         in Tc _random_init=0)
    {
        // Block matrix cannot be created automatically.
        static if (__traits(compiles, BlockMatrix!T))
            throw new Exception("Block matrix need to be created by the user.");
        else {
            // Create a bias vector if needed.
            Vector!TypeValue v = null;
            if (_bias)
                v = new Vector!T(_dim[0], _random_init);

            // Rectangular Random Matrices
            static if (__traits(compiles, new Mtype(_dim[0], _dim[1], _random_init))) {
                Mtype m = new Mtype(_dim[0], _dim[1], _random_init);
            }
            // Random Matrices
            else static if (__traits(compiles, new Mtype(_dim[0], _random_init))) {
                Mtype m = new Mtype(_dim[0], _random_init);
            }
            // Others
            else static if (__traits(compiles, new Mtype(_dim[0]))) {
                Mtype m = new Mtype(_dim[0]);
            }
            else static assert(0, "Automatic creation of the matrix failed.");

            this(m, v);
        }
    }

    this(in size_t _dim,
         in bool _bias=false,
         in Tc _random_init=0)
    {
        this([_dim, _dim], _bias, _random_init);
    }

    override
    @safe @nogc pure
    void takeOwnership(ref TypeValue[] _owner, ref size_t _index)
    {
        if (params) {
            auto tmp_params = cast(Mtype) this.params[0];
            takeOwnership_util_matrix!(Mtype, TypeValue)(_owner, tmp_params, _index);
            if (params.length > 1)
                takeOwnership_util!(TypeValue)(_owner, (cast(Vector!T) params[1]).v, _index);
        }
    }
    
    override
    Vector!TypeValue compute(Vector!TypeValue _v)
    {
        // Multiply the vector by the matrix first.
        auto res = (cast(Mtype) params[0]) * _v;

        // Add the bias vector if needed.
        if (params.length > 1)
            res += cast(Vector!TypeValue) params[1];
        return res;
    }
}
unittest {
    write("Unittest: Layer: Matrix ... ");

    auto v = new Vector!(Complex!real)(4, 1.0);

    auto r = new ReflectionMatrix!(Complex!real)(2, 1.0);
    auto d = new DiagonalMatrix!(Complex!real)(2, 2.0);
    auto gm = new Matrix!(Complex!real)(2, 2.0);
    auto b = new BlockMatrix!(Matrix!(Complex!real))(4, 4, 2, [gm, gm], true);
    auto f = new FourierMatrix!(Complex!real)(4);

    // Fourier
    {
        auto mf = new MatrixLayer!(FourierMatrix!(Complex!real))(4, true, 1.0);
        auto p = cast(Vector!(Complex!real)) mf.params[1];

        auto res1 = mf.compute(v);
        auto res2 = f * v;
        res2 += p;
        res2 -= res1;

        assert(p.norm!"L2" > 0.001);
        assert(res2.norm!"L2" <= 0.001);
    }
    // Matrix
    {
        auto vec = new Vector!double(4, 1.0);
        auto mm = new MatrixLayer!(Matrix!double)([2, 4], false, 10.0);
        auto m = cast(Matrix!double) mm.params[0]; 

        auto res1 = mm.compute(vec);
        auto res2 = m * vec;
        res2 -= res1;

        assert(res2.norm!"L2" <= 0.001);
        assert(res2.length == 2);
    }
    // Block
    {
        auto p = new Vector!(Complex!real)(4, 3.1415926535);
        auto mb = new MatrixLayer!(BlockMatrix!(Matrix!(Complex!real)))(b, p);
        auto m = cast(BlockMatrix!(Matrix!(Complex!real))) mb.params[0];

        auto res1 = mb.compute(v);
        auto res2 = m * v;
        res2 += p;
        res2 -= res1;

        assert(res2.norm!"L2" <= 0.0001);
    }
    // Permutation
    {
        auto p = new PermutationMatrix!float(16);
        auto w = new Vector!float(16, 1010101.0);

        auto mp = new MatrixLayer!(PermutationMatrix!float)(p);

        auto res1 = mp.compute(w);
        auto res2 = p * w;
    
        res1 -= res2;

        assert(res1.norm!"L2" <= 0.0001);

        mp = new MatrixLayer!(PermutationMatrix!float)(17);
        w = new Vector!float(17, 0.1);

        res1 = mp.compute(w);

        assert(abs(w.norm!"L2" - res1.norm!"L2") <= 0.001);
    }
    // Reflection
    {
        auto mr = new MatrixLayer!(ReflectionMatrix!(Complex!real))(4, false, 0.001);
        auto rm = new ReflectionMatrix!(Complex!real)(4, 5.0);

        mr.params[0] = rm.dup;
        
        auto res1 = mr.compute(v);
        auto res2 = rm * res1; // res2 should return to v.

        res2 -= v;

        assert(res2.norm!"L2" <= 0.0001);
    }
    // Unitary
    {
        auto mu = new MatrixLayer!(UnitaryMatrix!(Complex!real))(8, false, 1.0);
        auto w = new Vector!(Complex!real)(8, 1.0);

        auto u = cast(UnitaryMatrix!(Complex!real)) mu.params[0];

        auto res1 = mu.compute(w);
        assert(abs(res1.norm!"L2" - w.norm!"L2") <= 0.0001);
        
        w *= u;
        w -= res1;
        assert(w.norm!"L2" <= 0.0001);  
    }
    // Diagonal
    {
        auto md = new MatrixLayer!(DiagonalMatrix!(Complex!real))(2);
        md.params[0] = d;

        auto w = new Vector!(Complex!real)(2, 100000000.0);

        auto res = md.compute(w);
        w *= d;

        res -= w;

        assert(res.norm!"L2" <= 0.01);
    }

    write("Done.\n");
}


/+ This layer implement a simple linear matrix transformation
 + applied to the vector.
 + 
 + This might not really be useful..
 +/
class BiasLayer(T) : Layer!T
{
    @safe
    this(const size_t size_in, const Tc randomBound)
    {
        size = size_in;
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


    override
    @safe @nogc pure
    void takeOwnership(ref T[] _owner, ref size_t _index)
    {
        if (params)
            takeOwnership_util!(T)(_owner, (cast(Vector!T) params[0]).v, _index);
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

/++ This layer can implement any function that take as input a
 +  Vector!T and return another Vector!T.
 +
 +  It aims to be as general as possible. You can basically give everything
 +  as a delegate in the constructor.
 +
 +  For convinence, some function are already implemented and more are to go:
 +      -SoftMax
 +      -relu
 +      -modRelu
 + 
 +
 +  TODO: Add more functions ! The more, the merrier.
 +/
class FunctionalLayer(T) : Layer!T
{
    private {
        /// function applied to the vector.
        Vector!T delegate(Vector!T, Parameter[]) func;
    }

    this(string easyfunc, in size_t size_in=0)
    {
        switch (easyfunc)
        {
            case "relu":
                // function that compute "max(x, 0)" for every x in the vector.
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
                // Complex equivalent of the "relu" function.
                // It does, however, require a trainable bias vector.
                static if (is(Complex!T : T)) {
                    enforce(size_in != 0, "'size_in' must be greater than zero
                                            when using 'modRelu'.");
                    params = new Parameter[1];
                    params[0] = new Vector!Tc(size_in, 1.0);
                    size = size_in;
                    
                    func =
                        delegate(Vector!T _v, Parameter[] _p) {

                            static if (is(Complex!T : T)) {
                                import std.complex: abs;
                            }

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
                // Basic softmax function, can be used to obtain a probability distribution. 
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

    // The function to apply to the vector. Can be anything. DELEGATE
    @safe pure
    this(Vector!T delegate(Vector!T, Parameter[]) _func)
    {
        func = delegate(Vector!T _v, Parameter[] _p) {
            return _func(_v, _p);
        };
    }

    // The function to apply to the vector. Can be anything. Function
    pure
    this(Vector!T function(Vector!T, Parameter[]) _func)
    {
        this(toDelegate(_func));
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
    assert(std.complex.abs(v5.sum) < 0.0001);

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

    auto vv = new Vector!double(100, 0.2);

    write(" Done.\n");
}
