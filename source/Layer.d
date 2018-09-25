module source.Layer;

import std.algorithm;
import std.complex;
import std.conv: to;
import std.exception: enforce;
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
    import std.exception: assertThrown;
}

/// Activation Functions Handled by the Functional layer.
enum string[12] keywords_function = ["relu", "softmax", "binary",
                                     "logistic", "identity", "tanh",
                                     "arctan", "softsign", "softplus",
                                     "sin", "gaussian",""];

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
    abstract void apply(in Vector!T, ref Vector!T);
    abstract @safe @nogc pure void takeOwnership(ref T[], ref size_t);
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
if (!__traits(compiles, BlockMatrix!T))
{
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

    this(in Mtype _M, in Vector!T _v)
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
            Vector!T v = null;
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
    void takeOwnership(ref T[] _owner, ref size_t _index)
    {
        if (params) {
            auto tmp_params = cast(Mtype) this.params[0];
            takeOwnership_util_matrix!(Mtype, T)(_owner, tmp_params, _index);
            if (params.length > 1)
                takeOwnership_util!(T)(_owner, (cast(Vector!T) params[1]).v, _index);
        }
    }
    
    override
    Vector!T compute(Vector!T _v)
    {
        // Multiply the vector by the matrix first.
        auto res = (cast(Mtype) params[0]) * _v;

        // Add the bias vector if needed.
        if (params.length > 1)
            res += cast(Vector!T) params[1];
        return res;
    }
    
    override
    void apply(in Vector!T _v, ref Vector!T b)
    {
        static if (!is(Mtype: UnitaryMatrix!T)) {
            // Multiply the vector by the matrix first.
            (cast(Mtype) params[0]).multiply(_v, b);

            // Add the bias vector if needed.
            if (params.length > 1)
                b.v[] += (cast(Vector!T) params[1]).v[];
        }
    }
}
unittest {
    write("Unittest: Layer: Matrix ... ");

    { // Matrices
        auto v = new Vector!(Complex!real)(4, 1.0);

        auto r = new ReflectionMatrix!(Complex!real)(2, 1.0);
        auto d = new DiagonalMatrix!(Complex!real)(2, 2.0);
        auto gm = new Matrix!(Complex!real)(2, 2.0);
        auto b = new BlockMatrix!(Matrix!(Complex!real))(4, 4, 2, [gm, gm], true);
        auto f = new FourierMatrix!(Complex!real)(4);
        
        { // Fourier
            auto mf = new MatrixLayer!(FourierMatrix!(Complex!real))(4, true, 1.0);
            auto p = cast(Vector!(Complex!real)) mf.params[1];
            auto buf = new Vector!(Complex!real)(4);

            auto res1 = mf.compute(v);
            auto res2 = f * v;
            mf.apply(v, buf);
            res2 += p;
            res2 -= res1;
            buf -= res1;

            assert(p.norm!"L2" > 0.001);
            assert(res2.norm!"L2" <= 0.001);
            assert(buf.norm!"L2" <= 0.000001);
        }
        { // Matrix
            auto mm = new MatrixLayer!(Matrix!double)([2, 4], false, 10.0);
            auto m = cast(Matrix!double) mm.params[0];
            auto vec = new Vector!double(4, 1.0); 
            auto buf = new Vector!double(2); 

            auto res1 = mm.compute(vec);
            auto res2 = m * vec;
            mm.apply(vec, buf);
            res2 -= res1;
            buf -= res1;

            assert(res2.norm!"L2" <= 0.00001);
            assert(buf.norm!"L2" <= 0.00001);
            assert(res2.length == 2);
        }
        { // Permutation
            auto p = new PermutationMatrix!float(16);
            auto w = new Vector!float(16, 1010101.0);
            auto buf = new Vector!float(16, 0.1);

            auto mp = new MatrixLayer!(PermutationMatrix!float)(p);
            auto res1 = mp.compute(w);
            mp.apply(w, buf);
            auto res2 = p * w;
        
            res1 -= res2;
            buf -= res2;


            assert(res1.norm!"L2" <= 0.0001);
            assert(buf.norm!"L2" <= 0.0001);

            mp = new MatrixLayer!(PermutationMatrix!float)(17);
            w = new Vector!float(17, 0.1);

            res1 = mp.compute(w);

            assert(abs(w.norm!"L2" - res1.norm!"L2") <= 0.001);
        }
        { // Reflection
            auto mr = new MatrixLayer!(ReflectionMatrix!(Complex!real))(4, false, 0.001);
            auto rm = new ReflectionMatrix!(Complex!real)(4, 5.0);
            auto buf = new Vector!(Complex!real)(4);

            mr.params[0] = rm.dup;
            
            auto res1 = mr.compute(v);
            auto res2 = rm * res1; // res2 should return to v.
            mr.apply(v, buf);
            auto tmp = buf.dup;
            mr.apply(tmp, buf);

            res2 -= v;
            buf -= v;

            assert(res2.norm!"L2" <= 0.0001);
            assert(buf.norm!"L2" <= 0.0001);
        }
        { // Unitary
            auto mu = new MatrixLayer!(UnitaryMatrix!(Complex!real))(8, false, 1.0);
            auto w = new Vector!(Complex!real)(8, 1.0);

            auto u = cast(UnitaryMatrix!(Complex!real)) mu.params[0];

            auto res1 = mu.compute(w);
            assert(abs(res1.norm!"L2" - w.norm!"L2") <= 0.0001);
            
            w *= u;
            w -= res1;

            assert(w.norm!"L2" <= 0.0001);
        }
        { // Diagonal
            auto md = new MatrixLayer!(DiagonalMatrix!(Complex!real))(2);
            md.params[0] = d;

            auto w = new Vector!(Complex!real)(2, 100000000.0);
            auto buf = new Vector!(Complex!real)(2, 100000000.0);
            
            auto res = md.compute(w);
            md.apply(w, buf);
            w *= d;

            res -= w;
            buf -= w;

            assert(res.norm!"L2" <= 0.01);
            assert(buf.norm!"L2" <= 0.01);
        }
    }
    { // takeOwnership
        auto v = new Vector!real(4, 1.0);
        auto v2 = new Vector!real(2, 1.0);
        auto bias = new Vector!real(4, 1.0);
        auto bias2 = new Vector!real(2, 1.0);

        auto r = new ReflectionMatrix!real(4, 1.0);
        auto r1 = new ReflectionMatrix!real([1.0, 1.0, 1.0, 1.0]);
        auto layer_r = new MatrixLayer!(ReflectionMatrix!real)(r, bias);

        auto d = new DiagonalMatrix!real(4, 2.0);
        auto layer_d = new MatrixLayer!(DiagonalMatrix!real)(d, bias);

        auto u = new UnitaryMatrix!real(4, 1.0);
        auto u1 = new UnitaryMatrix!real(4); u1.params[] = 1.0; u1.perm = u.perm;
        auto layer_u = new MatrixLayer!(UnitaryMatrix!real)(u, bias);

        auto m = new Matrix!real(2, 2.0);
        auto m1 = new Matrix!real(2); m1.params[] = 1.0;
        auto layer_m = new MatrixLayer!(Matrix!real)(m, bias2);

        auto array_bouïlla = new real[10000];
        size_t indexouille = 0;

        layer_r.takeOwnership(array_bouïlla, indexouille);
        layer_d.takeOwnership(array_bouïlla, indexouille);
        layer_u.takeOwnership(array_bouïlla, indexouille);
        layer_m.takeOwnership(array_bouïlla, indexouille);

        array_bouïlla[] = 1.0;

        auto res_1 = layer_r.compute(v);
        res_1.v[] -= 1.0;
        res_1 = r1 * res_1;
        res_1 -= v;
        assert(res_1.norm!"L2" <= 0.00001);

        auto res_2 = layer_d.compute(v);
        res_2 -= v;
        res_2.v[] -= 1.0;
        assert(res_2.norm!"L2" <= 0.00001);

        auto res_3 = layer_u.compute(v);
        res_3 -= u1 * v;
        res_3.v[] -= 1.0;
        assert(res_3.norm!"L2" <= 0.00001);

        auto res_4 = layer_m.compute(v2);
        res_4 -= m1 * v2;
        res_4.v[] -= 1.0;
        assert(res_4.norm!"L2" <= 0.00001);

    }
    write("Done.\n");
}

class MatrixLayer(Mtype : BlockMatrix!(M!T), alias M, T) : Layer!T
{
    this()
    {
        // TODO: Add check to see if Mtype is a Matrix*!T
    }

    this(in Mtype _M)
    {
        this();
        size = _M.size_blocks * _M.size_blocks * _M.num_blocks;
        // We keep a duplicate of the matrix.
        params = new Parameter[1];
        params[0] = _M.dup;
    }

    this(in Mtype _M, in Vector!T _v)
    {
        this();
        enforce(_M.cols == _v.length, "Matrix / Bias dimensions mismatch.");

        size = _M.size_blocks * _M.size_blocks * _M.num_blocks + _v.length;
        
        params = new Parameter[2];
        params[0] = _M.dup;
        params[1] = _v.dup;
    }

    override
    @safe @nogc pure
    void takeOwnership(ref T[] _owner, ref size_t _index)
    {
        if (params) {
            auto tmp_params = cast(Mtype) this.params[0];
            takeOwnership_util_matrix!(Mtype, T)(_owner, tmp_params, _index);
            if (params.length > 1)
                takeOwnership_util!(T)(_owner, (cast(Vector!T) params[1]).v, _index);
        }
    }
    
    override
    Vector!T compute(Vector!T _v)
    {
        // Multiply the vector by the matrix first.
        auto res = (cast(Mtype) params[0]) * _v;

        // Add the bias vector if needed.
        if (params.length > 1)
            res += cast(Vector!T) params[1];
        return res;
    }

    override
    void apply(in Vector!T _v, ref Vector!T b)
    {
        // Multiply the vector by the matrix first.
        auto res = (cast(Mtype) params[0]) * _v;

        // Add the bias vector if needed.
        if (params.length > 1)
            res += cast(Vector!T) params[1];
        b.v[] = res.v[];
    }
}
unittest {
    write("Unittest: Layer: BlockMatrix ... ");

    { // Matrice
        auto v = new Vector!(Complex!real)(4, 1.0);

        auto r = new ReflectionMatrix!(Complex!real)(2, 1.0);
        auto d = new DiagonalMatrix!(Complex!real)(2, 2.0);
        auto gm = new Matrix!(Complex!real)(2, 2.0);
        auto b = new BlockMatrix!(Matrix!(Complex!real))(4, 4, 2, [gm, gm], true);
        auto f = new FourierMatrix!(Complex!real)(4);

        auto p = new Vector!(Complex!real)(4, 3.1415926535);
        auto mbp = new MatrixLayer!(BlockMatrix!(Matrix!(Complex!real)))(b, p);
        auto mb = new MatrixLayer!(BlockMatrix!(Matrix!(Complex!real)))(b);
        auto m = cast(BlockMatrix!(Matrix!(Complex!real))) (mb.params[0]);

        auto res1p = mbp.compute(v);
        auto res1 = mb.compute(v);
        auto res2p = m * v;
        auto res2 = res2p.dup;
        res2p += p;
        res2p -= res1p;
        res2 -= res1;

        assert(res2p.norm!"L2" <= 0.0001);
        assert(res2.norm!"L2" <= 0.0001);
    }

    { // takeOwnership
        auto v = new Vector!real(4, 1.0);
        auto bias = new Vector!real(4, 1.0);

        auto gm = new Matrix!real(2, 2.0);
        auto gm1 = new Matrix!real(2); gm1.params[] = 1.0;

        auto b = new BlockMatrix!(Matrix!real)(4, 4, 2, [gm, gm], false);
        auto b1 = new BlockMatrix!(Matrix!real)(4, 4, 2, [gm1, gm1], false);

        auto bml = new MatrixLayer!(BlockMatrix!(Matrix!real))(b, bias);

        auto array_bouïlla = new real[10000];
        size_t indexouille = 0;

        bml.takeOwnership(array_bouïlla, indexouille);

        array_bouïlla[] = 1.0;

        auto res1 = bml.compute(v);
        auto res2 = b1 * v;
        res2.v[] += 1.0;

        res1 -= res2;
        assert(res1.norm!"L2" <= 0.00001);
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
    void apply(in Vector!T _v, ref Vector!T b)
    {
        b.v[] = _v.v[];
        b.v[] += (cast(Vector!T) params[0]).v[];
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

    {
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


        size_t indexx;
        Complex!real ten = complex(10.0);
        auto arr = new Complex!real[4];

        b2.takeOwnership(arr, indexx);
        foreach(i; 0 .. 4)
            arr[i] = ten;

        auto res = b2.compute(v);
        res.v[] -= ten;
        res -= v;

        assert(res.norm!"L2" <= 0.00001);
    }

    {
        auto bias_layer = new BiasLayer!real(17, 6.67);
        auto buf = new Vector!real(17);
        auto vec = new Vector!real(17, 7.99);

        auto res = bias_layer.compute(vec);
        bias_layer.apply(vec, buf);

        buf -= res;

        assert(buf.norm!"L2" <= 0.0001);
    }

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
 +/
class FunctionalLayer(T, string strfunc="", TypeParameter...) : Layer!T
{
    protected {
        enum bool isKeyword = strfunc.isOneOf(keywords_function);
    }

    this() {
    }

    this(in size_t[] size_parameters, in Tc[] randomBound_parameters)
    {
        static if (!isKeyword) {
            params = new Parameter[TypeParameter.length];
            static foreach(i; 0 .. TypeParameter.length)
                mixin("params["~to!string(i)~"] = new "~TypeParameter[i].stringof~
                      "(size_parameters[i], randomBound_parameters[i]);");
            foreach(tmp_size; size_parameters)
                this.size += tmp_size;
        }
    }

    override
    @safe @nogc pure
    void takeOwnership(ref T[] _owner, ref size_t _index)
    {
        static if(!isKeyword) {
            static foreach(i; 0 .. TypeParameter.length) {
                static if (is(TypeParameter[i] : Vector!T))
                    takeOwnership_util!(T)(_owner, (cast(Vector!T) params[i]).v, _index);
                else
                    takeOwnership_util_matrix!(TypeParameter[i], T)
                                              (_owner, cast(TypeParameter[i]) params[i], _index);
            }
        }
    }

    override
    Vector!T compute(Vector!T _v)
    {
        static if (strfunc == "relu") {
            // function that compute "max(x, 0)" for every x in the vector.
            static if (is(Complex!T : T))
                static assert(0, "Relu function cannot be use on Complex-valued vectors.");

            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                if (res[i] < 0) res[i] = 0;
            return res;
        }
        else static if(strfunc == "softmax") {
            // Basic softmax function, can be used to obtain a probability distribution. 
            static if (is(Complex!T : T))
                static assert(0, "Softmax function cannot be used on Complex-valued vectors.");

            T s = 0;
            auto res = _v.dup;
            foreach(i; 0 .. _v.length) {
                res.v[i] = exp(_v[i]);
                s += res[i];
            }
            foreach(i; 0 .. _v.length)
                res.v[i] /= s;
            return res;
        }
        else static if(strfunc == "binary") {
            static if (is(Complex!T : T))
                static assert(0, "'binary' function undefined for Complex-valued vectors.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length){
                if (res[i] < 0) 
                    res[i] = 0.0;
                else
                    res[i] = 1.0;
            }
            return res;
        }
        else static if(strfunc == "logistic") {
            static if (is(Complex!T : T))
                static assert(0, "'logistic' function undefined for Complex-valued vectors.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = 1.0 / (1.0 + exp(-res[i]));
            return res;
        }
        else static if(strfunc == "identity") {
            auto res = _v.dup;
            return res;
        }
        else static if(strfunc == "tanh") {
            static if (is(Complex!T : T))
                static assert(0, "'tanh' function undefined for Complex-valued vectors.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = tanh(res[i]);
            return res;
        }
        else static if(strfunc == "arctan") {
            static if (is(Complex!T : T))
                static assert(0, "'arctan' function undefined for Complex-valued vectors.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = atan(res[i]);
            return res;
        }
        else static if(strfunc == "softsign") {
            static if (is(Complex!T : T))
                static assert(0, "'softsign' function undefined for Complex-valued vectors.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = res[i] / (1.0 + abs(res[i]));
            return res;
        }
        else static if(strfunc == "softplus") {
            static if (is(Complex!T : T))
                static assert(0, "'softplus' function undefined for Complex-valued vectors.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = log(1 + exp(res[i]));
            return res;
        }
        else static if(strfunc == "sin") {
            static if (is(Complex!T : T))
                static assert(0, "'sin' function undefined for Complex-valued vectors.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = sin(res[i]);
            return res;
        }
        else static if(strfunc == "gaussian") {
            static if (is(Complex!T : T))
                static assert(0, "'gaussian' function undefined for Complex-valued vectors.");
            auto res = _v.dup;
            foreach(i; 0 .. _v.length)
                res[i] = exp(-pow(res[i], 2));
            return res;
        }
        else static if(strfunc == "") return _v;
        else {
            auto b = new Vector!T(1); // This is here to make it compile when we use a buffer
            static foreach(i; 0 .. TypeParameter.length)
                mixin("auto p"~to!string(i)~" = cast("~TypeParameter[i].stringof~") params["~to!string(i)~"];");
            mixin(strfunc);
            return b;
        }
    }
    
    override
    void apply(in Vector!T _v, ref Vector!T b)
    {
        static if (strfunc == "relu") {
            // function that compute "max(x, 0)" for every x in the vector.
            static if (is(Complex!T : T))
                static assert(0, "Relu function cannot be use on Complex-valued vectors.");

            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length)
                if (b[i] < 0) b[i] = 0;
        }
        else static if(strfunc == "softmax") {
            // Basic softmax function, can be used to obtain a probability distribution. 
            static if (is(Complex!T : T))
                static assert(0, "Softmax function cannot be used on Complex-valued vectors.");

            T s = 0;
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length) {
                b.v[i] = exp(_v[i]);
                s += b[i];
            }
            foreach(i; 0 .. _v.length)
                b.v[i] /= s;
        }
        else static if(strfunc == "binary") {
            static if (is(Complex!T : T))
                static assert(0, "'binary' function undefined for Complex-valued vectors.");
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length){
                if (b[i] < 0) 
                    b[i] = 0.0;
                else
                    b[i] = 1.0;
            }
        }
        else static if(strfunc == "logistic") {
            static if (is(Complex!T : T))
                static assert(0, "'logistic' function undefined for Complex-valued vectors.");
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length)
                b[i] = 1.0 / (1.0 + exp(-b[i]));
        }
        else static if(strfunc == "identity") {
            b.v[] = _v.v[];
        }
        else static if(strfunc == "tanh") {
            static if (is(Complex!T : T))
                static assert(0, "'tanh' function undefined for Complex-valued vectors.");
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length)
                b[i] = tanh(b[i]);
        }
        else static if(strfunc == "arctan") {
            static if (is(Complex!T : T))
                static assert(0, "'arctan' function undefined for Complex-valued vectors.");
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length)
                b[i] = atan(b[i]);
        }
        else static if(strfunc == "softsign") {
            static if (is(Complex!T : T))
                static assert(0, "'softsign' function undefined for Complex-valued vectors.");
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length)
                b[i] = b[i] / (1.0 + abs(b[i]));
        }
        else static if(strfunc == "softplus") {
            static if (is(Complex!T : T))
                static assert(0, "'softplus' function undefined for Complex-valued vectors.");
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length)
                b[i] = log(1 + exp(b[i]));
        }
        else static if(strfunc == "sin") {
            static if (is(Complex!T : T))
                static assert(0, "'sin' function undefined for Complex-valued vectors.");
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length)
                b[i] = sin(b[i]);
        }
        else static if(strfunc == "gaussian") {
            static if (is(Complex!T : T))
                static assert(0, "'gaussian' function undefined for Complex-valued vectors.");
            b.v[] = _v.v[];
            foreach(i; 0 .. _v.length)
                b[i] = exp(-pow(b[i], 2));
        }
        else static if(strfunc == "") b.v[] = _v.v[];
        else {
            static foreach(i; 0 .. TypeParameter.length)
                mixin("auto p"~to!string(i)~" = cast("~TypeParameter[i].stringof~") params["~to!string(i)~"];");
            mixin(strfunc);
        }
    }
}
unittest {
    write("                 Functional ... ");

    {
        alias Vec = Vector!(Complex!double);

        enum string blue = "
            b.v[] = _v.v[];
            b.v[] *= p0.v[];";

        enum string ff = "
            b.v[] = _v.v[];
            b.v[] *= complex(4.0);";

        uint len = 4;
        double pi = 3.1415923565;

        auto v = new Vec([complex(0.0), complex(1.0), complex(pi), complex(-1.0)]);
        auto e = new Vec([complex(0.0)]);

        // Empty functional is identity.
        auto f1 = new FunctionalLayer!(Complex!double)();
        auto v1 = f1.compute(v);
        v1 -= v;
        assert(v1.norm!"L2" <= 0.001);

        // More complex functional without learnable parameter.
        auto f3 = new FunctionalLayer!(Complex!double, ff)();
        auto v3 = v.dup;
        f3.apply(v, v3);
        auto v3_bis = v.dup;
        v3_bis.v[] *= complex(4.0);
        v3 -= v3_bis;
        assert(v3.norm!"L2" <= 0.00001);

        auto f4 = new FunctionalLayer!(Complex!double, blue, Vec)([len], [1.0]);
        auto v4 = v.dup;
        f4.apply(v, v4);
        auto v4_bis = v.dup();
        auto f4_p1 = cast(Vec) f4.params[0];
        foreach(i; 0 .. v4.length) {
            v4[i] -= v4_bis[i] * f4_p1[i];
        }
        assert(v4.norm!"L2" <= 0.00001);

        auto vr = new Vector!double([0.0, 1.0, pi, -1.0]);

        // relu function.
        auto f6 = new FunctionalLayer!(double, "relu");
        auto vr6 = vr.dup;
        f6.apply(vr, vr6);
        assert(abs(vr6.sum - 1.0 - pi) <= 0.01);

        // softmax function.
        auto f7 = new FunctionalLayer!(double, "softmax");
        auto vr7 = f7.compute(vr);
        assert(abs(vr7.sum - 1.0) <= 0.001);

        auto vv = new Vector!double(100, 0.2);
    }

    { // takeOwnership
        enum string strfunc = "b.v[] = _v.v[]; b += p0;";
        auto f = new FunctionalLayer!(double, strfunc, Vector!double)([4UL], [10.0]);
        auto v = new Vector!double([1.0, 2.0, 3.0, 4.0]);

        auto a = new double[4];
        auto s = new Vector!double([5.0, 5.0, 5.0, 5.0]);
        size_t i = 0;

        f.takeOwnership(a, i);
        a[0] = 4.0;
        a[1] = 3.0;
        a[2] = 2.0;
        a[3] = 1.0;

        auto r = v.dup;
        f.apply(v, r);
        r -= s;
        assert(r.norm!"L2" <= 0.000001);
    }

    {
        auto f_relu = new FunctionalLayer!(real, "relu")();
        auto vec = new Vector!real([-1.0, -0.5, 0.5, 1.0]);
        auto b = new Vector!real(4);
        
        auto nn_relu = new FunctionalLayer!(real, "relu")();
        auto res_relu = new Vector!real([0.0, 0.0, 0.5, 1.0]);
        nn_relu.apply(vec, b);
        b -= nn_relu.compute(vec);
        res_relu -= nn_relu.compute(vec);
        assert(res_relu.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_logistic = new FunctionalLayer!(real, "logistic")();
        auto res_logistic = new Vector!real([1.0/(1.0 + exp( 1.0)), 1.0/(1.0 + exp( 0.5)),
                                             1.0/(1.0 + exp(-0.5)), 1.0/(1.0 + exp(-1.0))]);
        nn_logistic.apply(vec, b);
        b -= nn_logistic.compute(vec);
        res_logistic -= nn_logistic.compute(vec);
        assert(res_logistic.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_gaussian = new FunctionalLayer!(real, "gaussian")();
        auto res_gaussian = new Vector!real([exp(-1.0), exp(-0.25), exp(-0.25), exp(-1.0)]);
        nn_gaussian.apply(vec, b);
        b -= nn_gaussian.compute(vec);
        res_gaussian -= nn_gaussian.compute(vec);
        assert(res_gaussian.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_identity = new FunctionalLayer!(real, "identity")();
        auto res_identity = new Vector!real([-1.0, -0.5, 0.5, 1.0]);
        nn_identity.apply(vec, b);
        b -= nn_identity.compute(vec);
        res_identity -= nn_identity.compute(vec);
        assert(res_identity.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_tanh = new FunctionalLayer!(real, "tanh")();
        auto res_tanh = new Vector!real([tanh(-1.0), tanh(-0.5), tanh(0.5), tanh(1.0)]);
        nn_tanh.apply(vec, b);
        b -= nn_tanh.compute(vec);
        res_tanh -= nn_tanh.compute(vec);
        assert(res_tanh.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_arctan = new FunctionalLayer!(real, "arctan")();
        auto res_arctan = new Vector!real([atan(-1.0), atan(-0.5), atan(0.5), atan(1.0)]);
        nn_arctan.apply(vec, b);
        b -= nn_arctan.compute(vec);
        res_arctan -= nn_arctan.compute(vec);
        assert(res_arctan.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_softsign = new FunctionalLayer!(real, "softsign")();
        auto res_softsign = new Vector!real([-0.5, -1.0/3.0, 1.0/3.0, 0.5]);
        nn_softsign.apply(vec, b);
        b -= nn_softsign.compute(vec);
        res_softsign -= nn_softsign.compute(vec);
        assert(res_softsign.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_softplus = new FunctionalLayer!(real, "softplus")();
        auto res_softplus = new Vector!real([log(1+exp(-1.0)), log(1+exp(-0.5)), log(1 + exp(0.5)), log(1 + exp(1.0))]);
        nn_softplus.apply(vec, b);
        b -= nn_softplus.compute(vec);
        res_softplus -= nn_softplus.compute(vec);
        assert(res_softplus.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_sin = new FunctionalLayer!(real, "sin")();
        auto res_sin = new Vector!real([sin(-1.0), sin(-0.5), sin(0.5), sin(1.0)]);
        nn_sin.apply(vec, b);
        b -= nn_sin.compute(vec);
        res_sin -= nn_sin.compute(vec);
        assert(res_sin.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_binary = new FunctionalLayer!(real, "binary")();
        auto res_binary = new Vector!real([0.0, 0.0, 1.0, 1.0]);
        nn_binary.apply(vec, b);
        b -= nn_binary.compute(vec);
        res_binary -= nn_binary.compute(vec);
        assert(res_binary.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_softmax = new FunctionalLayer!(real, "softmax")();
        auto res_softmax = new Vector!real([0.06887305, 0.11355246, 0.3086676, 0.508906]);
        nn_softmax.apply(vec, b);
        b -= nn_softmax.compute(vec);
        res_softmax -= nn_softmax.compute(vec);
        assert(res_softmax.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);

        auto nn_nothing = new FunctionalLayer!(real, "")();
        auto res_nothing = vec.dup;
        nn_nothing.apply(vec, b);
        b -= nn_nothing.compute(vec);
        res_nothing -= nn_nothing.compute(vec);
        assert(res_nothing.norm!"L2" <= 0.000001);
        assert(b.norm!"L2" <= 0.000001);


        enum string strfunc = "b.v[]=_v.v[]; foreach(i; 0 .. b.length) {b.v[i] += p0.v[i];}";
        auto nn_weird = new FunctionalLayer!(real, strfunc, Vector!real)([17UL], [1.0]);
        auto ver = new Vector!real(17, 1.0);
        auto buf = new Vector!real(17);

        auto bias = (cast(Vector!real) nn_weird.params[0]).dup;
        
        nn_weird.apply(ver, buf);
        ver += bias;
        buf -= ver;

        assert(buf.norm!"L2" <= 0.000001);
    }

    write(" Done.\n");
}
