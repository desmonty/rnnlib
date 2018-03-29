module source.Matrix;

import std.algorithm;
import std.complex;
import std.exception: assertThrown, enforce;
import std.math;
import std.numeric : Fft, dotProduct;
import std.random;
import std.range : iota, array;
import std.string;

import source.Parameter;

version(unittest)
{
    import std.stdio : writeln, write;
    import source.Parameter;
    import std.datetime;
    import core.exception;
}


auto dot(R1, R2)(in R1[] lhs, in R2[] rhs)
{
    R1 s = lhs[0] * rhs[0];
    foreach(i; 1 .. lhs.length)
        s += lhs[i] * rhs[i];
    return s;
}
unittest
{
    assert(dot([1.0, 2.0, 3.0], [-6.0, -5.0, -1.0]) ==
           dot([-6.0, -5.0, -1.0], [1.0, 2.0, 3.0]));

    assert(abs(dot([complex(1.0, 8.5), complex(6.4, 3.58), complex(10.8, 7.65)],
               [-6.0, -5.0, -1.0])
           -
               dot([complex(6.4, 3.58), complex(10.8, 7.65), complex(1.0, 8.5)],
               [-5.0, -1.0, -6.0])) < 0.001);
}

bool isComplexType(T)()
{
    return T.stringof.startsWith("Complex");
}
unittest
{
    assert(isComplexType!(Complex!float));
    assert(isComplexType!(Complex!double));
    assert(isComplexType!(Complex!real));
    assert(!isComplexType!float);
    assert(!isComplexType!double);
    assert(!isComplexType!real);
    assert(!isComplexType!uint);
    assert(!isComplexType!int);
    assert(!isComplexType!long);
    assert(!isComplexType!size_t);
    assert(!isComplexType!short);
}

class MatrixAbstract(S, T) : Parameter {
    S rows, cols;
    string typeId;

    static if (isComplexType!T)
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    const
    Vector!(S,T) opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        return new Vector!(S,T)(this * v.v);
    }

/+
    const @property
    MatrixAbstract dup()
    {
        if (typeId.stringof.startsWith("BlockMatrix"))
            return new BlockMatrix!(S,T)(cast(BlockMatrix!(S,T)) this);
        else if (typeId.stringof.startsWith("UnitaryMatrix"))
            return new UnitaryMatrix!(S,T)(cast(UnitaryMatrix!(S,T)) this);
        else if (typeId.stringof.startsWith("DiagonalMatrix"))
            return new DiagonalMatrix!(S,T)(cast(DiagonalMatrix!(S,T)) this);
        else if (typeId.stringof.startsWith("ReflectionMatrix"))
            return new ReflectionMatrix!(S,T)(cast(ReflectionMatrix!(S,T)) this);
        else if (typeId.stringof.startsWith("PermutationMatrix"))
            return new PermutationMatrix!(S,T)(cast(PermutationMatrix!(S,T)) this);
        else if (typeId.stringof.startsWith("FourierMatrix"))
            return new FourierMatrix!(S,T)(cast(FourierMatrix!(S,T)) this);
        else if (typeId.stringof.startsWith("Matrix"))
            return new Matrix!(S,T)(cast(Matrix!(S,T)) this);
        else if (typeId.stringof.startsWith("PermutationMatrix"))
            return new PermutationMatrix!(S,T)(cast(PermutationMatrix!(S,T)) this);
    }

    BlockMatrix!(S,T) opCast()
    { return cast(BlockMatrix!(S,T)) this; }
    UnitaryMatrix!(S,T) opCast()
    { return cast(UnitaryMatrix!(S,T)) this; }
    DiagonalMatrix!(S,T) opCast()
    { return cast(DiagonalMatrix!(S,T)) this; }
    ReflectionMatrix!(S,T) opCast()
    { return cast(ReflectionMatrix!(S,T)) this; }
    PermutationMatrix!(S,T) opCast()
    { return cast(PermutationMatrix!(S,T)) this; }
    FourierMatrix!(S,T) opCast()
    { return cast(FourierMatrix!(S,T)) this; }
    Matrix!(S,T) opCast()
    { return cast(Matrix!(S,T)) this; }
    PermutationMatrix!(S,T) opCast()
    { return cast(PermutationMatrix!(S,T)) this; }+/

    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        // TODO: Refactor. This is ugly but one can't simply use mixin here.
        auto tmptypeId = split(typeId, "!")[0];
        switch (tmptypeId)
        {
            case "BlockMatrix":
                return cast(BlockMatrix!(S,T)) this * v;
            case "UnitaryMatrix":
                static if (isComplexType!T) {
                    return cast(UnitaryMatrix!(S,T)) this * v;
                }
                else assert(0, "Unitary matrices must be of complex type.");
            case "DiagonalMatrix":
                return cast(DiagonalMatrix!(S,T)) this * v;
            case "ReflectionMatrix":
                return cast(ReflectionMatrix!(S,T)) this * v;
            case "PermutationMatrix":
                return cast(PermutationMatrix!(S,T)) this * v;
            case "FourierMatrix":
                static if (isComplexType!T) {
                    return cast(FourierMatrix!(S,T)) this * v;
                }
                else assert(0, "Fourier matrices must be of complex type.");
            case "Matrix":
                return cast(Matrix!(S,T)) this * v;
            default:
                assert(0, tmptypeId~" is not in the 'switch'
                                      clause of MatrixAbstract");
        }
    }

    const
    Vector!(S,T) opBinaryRight(string op)(in Vector!(S,T) v)
    if (op=="/")
    {
        auto res = new Vector!(S,T)(v.v / this);
        return res;
    }

    const
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        // TODO: Refactor.
        auto tmptypeId = split(typeId, "!")[0];
        switch (tmptypeId)
        {
            case "BlockMatrix":
                return v / cast(BlockMatrix!(S,T)) this;
            case "UnitaryMatrix":
                static if (isComplexType!T) {
                    return v / cast(UnitaryMatrix!(S,T)) this;
                }
                else assert(0, "Unitary matrices must be of complex type.");
            case "DiagonalMatrix":
                return v / cast(DiagonalMatrix!(S,T)) this;
            case "ReflectionMatrix":
                return v / cast(ReflectionMatrix!(S,T)) this;
            case "PermutationMatrix":
                return v / cast(PermutationMatrix!(S,T)) this;
            case "FourierMatrix":
                static if (isComplexType!T) {
                    return v / cast(FourierMatrix!(S,T)) this;
                }
                else assert(0, "Fourier matrices must be of complex type.");
            case "Matrix":
                assert(0, "Division by a general matrix
                           is not yet implemented.");
            default:
                assert(0, tmptypeId~" is not in the 'switch'
                                      clause of MatrixAbstract");
        }
    }
}
unittest
{
    write("Unittest Matrix Abstract ... ");


    class ErrorMatrix(S,T) : MatrixAbstract!(S,T) {
        this() {
            typeId = "ErrorMatrix";
            rows = 0;
            cols = 0;
        }
    }

    auto len = 1024;
    auto m1 = new PermutationMatrix!(ulong, Complex!real)(len/4, 1.0);
    auto m2 = new DiagonalMatrix!(ulong, Complex!real)(len/4, 1.0);
    auto m3 = new ReflectionMatrix!(ulong, Complex!real)(len/4, 1.0);
    auto m4 = new FourierMatrix!(ulong, Complex!real)(len/4);
    auto bm = new BlockMatrix!(ulong, Complex!real)(len, len/4, [m1,m2,m3,m4],
                                                    false);
    auto um = new UnitaryMatrix!(ulong, Complex!real)(len, 3.14159265351313);
    auto mm = new Matrix!(ulong, Complex!real)(len, 5.6);
    auto em = new ErrorMatrix!(ulong, Complex!real)();

    auto list_mat = [m1, m2, m3, m4, bm, um, mm, em];
    auto m1_hyde = list_mat[0];
    auto m2_hyde = list_mat[1];
    auto m3_hyde = list_mat[2];
    auto m4_hyde = list_mat[3];
    auto bm_hyde = list_mat[4];
    auto um_hyde = list_mat[5];
    auto mm_hyde = list_mat[6];
    auto em_hyde = list_mat[7];


    auto v = new Vector!(ulong, Complex!real)(len);
    auto w = new Vector!(ulong, Complex!real)(len/4);
    foreach(i; 0 .. len)
        v[i] = complex(cast(real)(i*2 - len/2), cast(real)(len/3 - i/3.0));


    auto v2 = bm_hyde * v;
    auto v3 = v2 / bm_hyde;

    v3 -= v; assert(v3.norm!"L2" < 0.01);
    v2 -= v; assert(v2.norm!"L2" > 1.0);

    v2 = um_hyde * v;
    v3 = v2 / um_hyde;

    v3 -= v; assert(v3.norm!"L2" < 0.01);
    v2 -= v; assert(v2.norm!"L2" > 1.0);


    v2 = mm_hyde * v;
    v3 = mm * v;

    v3 -= v2; assert(v3.norm!"L2" < 0.01);
    v2 -= v; assert(v2.norm!"L2" > 1.0);

    bool error = false;
    try {
        auto ver = em_hyde * v;
    }
    catch (AssertError e) {
        error = true;
    }
    assert(error);
    error = false;
    try {
        auto ver = v / em_hyde;
    }
    catch (AssertError e) {
        error = true;
    }
    assert(error);
    error = false;
    try {
        auto ver = v / mm_hyde;
    }
    catch (AssertError e) {
        error = true;
    }
    assert(error);

    write("Done.\n");
}

class BlockMatrix(S,T) : MatrixAbstract!(S,T) {
    MatrixAbstract!(S,T)[] blocks;
    PermutationMatrix!(S,T) P, Q;

    S size_blocks;
    S num_blocks;
    S size_out;
    S size_in;

    // The two permutations are needed here so that 
    // every nodes can be connected to any others.
    // It is a way to "shuffle" the block matrix
    // and so to have a sparse matrix with the properties
    // of the block matrix (e.g. unitary).

    /+
        We present here the three different kinds of block matrix we
        can have. In the following, each '0' represent a zero square
        matrix of size 'size_blocks' and each number [1-9] represent
        a different square matrix of the same size.

        1/ Square block matrix:

            1 0 0 0 0
            0 2 0 0 0
            0 0 3 0 0
            0 0 0 4 0
            0 0 0 0 5

        2/ Rectangular block matrix with more columns:

            1 0 0 4 0
            0 2 0 0 5
            0 0 3 0 0

        2/ Rectangular block matrix with more rows:

            1 0 0
            0 2 0
            0 0 3
            4 0 0
            0 5 0
     +/
    pure
    this(){}

    pure
    this(in S size_in, in S size_out, in S size_blocks)
    {
        typeId = "BlockMatrix";
        enforce(size_out%size_blocks == 0,
                "'size_out' must be a multiple of 'size_blocks'.");
        enforce(size_in%size_blocks == 0,
                "'size_in' must be a multiple of 'size_blocks'.");
        rows = size_out;
        cols = size_in;

        this.size_blocks = size_blocks;
        S maxsize = size_out; if (maxsize<size_in) maxsize=size_in;
        this.num_blocks = maxsize/size_blocks;
        this.size_out = size_out;
        this.size_in = size_in;
    }

    this(in S size_in, in S size_out, in S size_blocks,
         MatrixAbstract!(S,T)[] blocks, bool randperm=false)
    {
        this(size_in, size_out, size_blocks);
        this.blocks = blocks;

        if (randperm) {
            P = new PermutationMatrix!(S,T)(size_in, 1.0);
            Q = new PermutationMatrix!(S,T)(size_out, 1.0);
        }
        else {
            P = new PermutationMatrix!(S,T)(size_in.iota.array);
            Q = new PermutationMatrix!(S,T)(size_out.iota.array);
        }
    }

    this(in S size_in, in S size_blocks,
         MatrixAbstract!(S,T)[] blocks, bool randperm=false)
    {
        this(size_in, size_in, size_blocks, blocks, randperm);
    }
    /+
    this(in BlockMatrix M)
    {
        this.size_blocks = M.size_blocks;
        this.num_blocks = M.num_blocks;
        this.size_out = M.size_out;
        this.size_in = M.size_in;

        this.P = M.P.dup;
        this.Q = M.Q.dup;
        auto tmp_blocks = new MatrixAbstract!(S,T)[M.num_blocks];
        foreach(i; 0 .. M.num_blocks)
            tmp_blocks[i] = M.blocks[i].dup;
        this.blocks = tmp_blocks;
    }+/

    const
    auto opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        auto res = this * v.v;
        return new Vector!(S,T)(res);
    }

    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        T[] vec = this.P * v;

        S blocks_in = size_in / size_blocks;
        S blocks_out = size_out / size_blocks;

        T[] res = new T[size_out];
        T[] s;
        S index;

        foreach(S b; 0 .. blocks_out) {
            // We take the first block matrix and multiply it with
            // the corresponding part of the vector.
            s = blocks[b] * vec[(b*size_blocks) .. ((b+1)*size_blocks)];
            // We then increment the index in case the block matrix
            // is rectangular with more columns than rows.
            index = b + blocks_out;
            while(index < blocks_in) {
                s[] += (blocks[index] *
                       vec[(index*size_blocks) .. ((index+1)*size_blocks)])[];
                index += blocks_out;
            }

            res[(b*size_blocks) .. ((b+1)*size_blocks)] = s;
        }

        res = this.Q * res;
        return res;
    }

    const
    auto opBinaryRight(string op)(in Vector!(S,T) v)
    if (op=="/")
    {
        return new Vector!(S,T)(v.v / this);
    }

    const
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        enforce(size_out == size_in, "Warning: Inverse of rectangular
                                      block matrix is not implemented");
        T[] vec = v / this.Q;

        S blocks_in = size_in / size_blocks;
        S blocks_out = size_out / size_blocks;

        T[] res = new T[size_out];

        foreach(S b; 0 .. blocks_out) {
            // We take the first block matrix and multiply it with
            // the corresponding part of the vector.
            res[(b*size_blocks) .. ((b+1)*size_blocks)] =
                vec[(b*size_blocks) .. ((b+1)*size_blocks)] / blocks[b];
        }

        res = res / this.P;
        return res;
    }
}
unittest
{
    write("Unittest Block Matrix ... ");

    auto len = 1024;
    auto m0 = new BlockMatrix!(uint, float)();
    auto m1 = new PermutationMatrix!(ulong, Complex!float)(len/4, 1.0);
    auto m2 = new DiagonalMatrix!(ulong, Complex!float)(len/4, 1.0);
    auto m3 = new ReflectionMatrix!(ulong, Complex!float)(len/4, 1.0);
    auto m4 = new FourierMatrix!(ulong, Complex!float)(len/4);
    auto bm = new BlockMatrix!(ulong, Complex!float)(len, len/4, [m1,m2,m3,m4],
                                                     false);

    auto v = new Vector!(ulong, Complex!float)(len);
    foreach(i; 0 .. len)
        v[i] = complex(cast(float)(i*2 - len/2), cast(float)(len/3 - i/3.0));

    auto mem = v.dup;


    auto v2 = bm * v;
    auto v3 = v2 / bm;
    v3 -= v;
    assert(v3.norm!"L2" < 1.0);
    v2 -= v;
    assert(v2.norm!"L2" > 1.0);

    assertThrown(new BlockMatrix!(uint, real)(4u, 4u, 3u));
    assertThrown(new BlockMatrix!(uint, real)(4u, 3u, 3u));

    write("Done.\n");
}

class UnitaryMatrix(S, T) : MatrixAbstract!(S,T)
{
    /+
        This matrix is defined in 
            Unitary Evolution Recurrent Neural Networks,
            Arjovsky, Shah, Bengio
            (http://proceedings.mlr.press/v48/arjovsky16.pdf)

        Its aim is to allow one to learn a unitary matrix that
        depends on number of parameters that is linear relatively
        to the size of matrix.

        The resulting function is:
            D_3 * R_2 * invF * D_2 * P * R_1 * F * D_1

        where D_i are diagonal unitary complex matrices,
              R_i are reflection unitary complex matrices,
              F and invF are respectivelly the fourier
                and inverse fourier transform,
              P is a permutation matrix.
     +/

    static assert(isComplexType!T,
               "UnitaryMatrix must be complex-valued.");

    PermutationMatrix!(S,T) perm;
    FourierMatrix!(S,T) fourier;
    Tc[] params;


    /+ The params vector include in the following order:
       + 3 diagonal unitary complex matrices.
       + 2 reflection unitary complex matrices.

       This will provide us with a simple way to change these
       parameters.
     +/

    pure
    this() {}

    this(in S size)
    {   
        this();
        assert(std.math.isPowerOf2(size), "Size of Unitary Matrix
                                           must be a power of 2.");
        typeId = "UnitaryMatrix";

        rows = size;
        cols = size;

        perm = new PermutationMatrix!(S,T)(size ,1.0);
        fourier = new FourierMatrix!(S,T)(size);
        params = new Tc[7*size];
    }

    this(in S size, in Tc randomBound)
    {
        this(size);

        foreach(i;0 .. params.length)
            params[i] = uniform(-randomBound, randomBound, rnd);
    }
    
    this(in UnitaryMatrix M)
    {
        auto res = new UnitaryMatrix!(S,T)();
        res.perm = M.perm.dup;
        res.fourier = M.fourier.dup;
        res.fourier = M.fourier.dup;
        res.params = M.params.dup;
    }

    const @property
    auto dup()
    {
        return new UnitaryMatrix(this);
    }


    /// Apply the "num"th diagonal matrix on the given vector.
    const pure
    void applyDiagonal(ref T[] v, in S num)
    {
        // we use expi to convert each value to a complex number
        // the value is the angle of the complex number with radius 1.
        S start_index = num*rows;
        foreach(i; 0 .. rows)
            v[i] *= cast(T) std.complex.expi(params[start_index + i]);
    }    /// Apply the "num"th diagonal matrix on the given vector.
    
    const pure
    void applyDiagonalInv(ref T[] v, in S num)
    {
        // we use expi to convert each value to a complex number
        // the value is the angle of the complex number with radius 1.
        S start_index = num*rows;
        foreach(i; 0 .. rows)
            v[i] /= cast(T) std.complex.expi(params[start_index + i]);
    }

    /// Apply the "num"th reflection matrix on the given vector.
    const pure
    auto applyReflection(ref T[] v, in S num)
    {
        // The '+3' is because the diagonal matrices are first
        // in the params array.
        // The '*2' is because each reflection matrix need
        // 2 * rows parameters to be defined.
        S start_index = (2*num + 3)*rows;
        S start_indexPlRows = start_index + rows;
        auto a = params[start_index .. start_indexPlRows];
        auto b = params[start_indexPlRows .. start_indexPlRows + rows];
        T tmp_c = dot(v, a) - complex(0, 1) * dot(v, b);
        T[] tmp_vec = new T[rows];
        tmp_c *= 2 / (dot(a, a) + dot(b, b));

        foreach(i; 0 .. rows)
            tmp_vec[i] = complex(a[i], b[i]) *  tmp_c;

        v[] -= tmp_vec[];
    }

    const
    Vector!(S,T) opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        return new Vector!(S,T)(this * v.v);
    }

    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        T[] res = v.dup;

        applyDiagonal(res, 0);
        
        res = fourier * res;

        applyReflection(res, 0);
        res = perm * res;
        applyDiagonal(res, 1);
        
        res = res / fourier;
        
        applyReflection(res, 1);
        applyDiagonal(res, 2);

        return res;
    }

    const
    Vector!(S,T) opBinaryRight(string op)(in Vector!(S,T) v)
    if (op=="/")
    {
        return new Vector!(S,T)(v.v / this);
    }

    const
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        T[] res = v.dup;

        applyDiagonalInv(res, 2);
        applyReflection(res, 1);
        
        res = fourier * res;

        applyDiagonalInv(res, 1);
        res = res / perm;
        applyReflection(res, 0);
        
        res = res / fourier;
        
        applyDiagonalInv(res, 0);

        return res;
    }
}
unittest
{
    write("Unittest Unitary Matrix ... ");
    {
        auto m0 = new UnitaryMatrix!(uint, Complex!float)();
        auto m = new UnitaryMatrix!(uint, Complex!double)(1024, 9.0);
        auto n= m.dup;
        auto v = new Vector!(uint, Complex!double)(1024, 1.2);

        auto k = v.dup;
        auto l = k.dup;

        auto r = m * v;
        v = r / m;

        auto s = m * l;
        l = s / m;

        k -= v;
        l -= v;

        assert(k.norm!"L2" < 0.00001);
        assert(l.norm!"L2" < 0.00001);

        bool error =  false;
        try {
            auto err = new UnitaryMatrix!(uint, float)(); 
        }
        catch (AssertError e) {
            error = true;
        }
        assert(error);
    }

    write("Done.\n");
}

class FourierMatrix(S,T) : MatrixAbstract!(S,T) {
    static assert(isComplexType!T,
               "FourierMatrix must be complex-valued.");

    Fft objFFT;

    this(S size)
    {
        typeId = "FourierMatrix";
        rows = size;
        cols = size;
        objFFT = new Fft(size);
    };
    
    this(in FourierMatrix M)
    {
        this(M.rows);
    }

    const @property
    auto dup()
    {
        return new FourierMatrix(this);
    }

    const
    Vector!(S,T) opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        return new Vector!(S,T)(this * v.v);
    }

    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        return objFFT.fft!(Tc)(v);
    }


    const
    Vector!(S,T) opBinaryRight(string op)(in Vector!(S,T) v)
    if (op=="/")
    {
        return new Vector!(S,T)(v.v / this);
    }

    const
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        return objFFT.inverseFft!(Tc)(v);
    }
}
unittest
{
    write("Unittest Fourrier Matrix ... ");
    {
        alias Fourier = FourierMatrix!(uint, Complex!double);
        auto f = new Fourier(1024);
        auto g =  f.dup;
        auto v = new Complex!double[2048];
        auto vc = v[15 .. 1039];

        assert(vc.length == 1024);

        auto rnd_tmp = Random(cast(uint) ((Clock.currTime()
                         - SysTime(unixTimeToStdTime(0))).total!"msecs"));

        foreach(i;0 .. 1024)
            vc[i] = complex(uniform(-1.0, 1.0, rnd_tmp),
                            uniform(-1.0, 1.0, rnd_tmp));

        auto r1 = f * (vc / f);
        auto r2 = (g * vc) / g;

        foreach(i;0 .. 1024){
            assert(std.complex.abs(r1[i] - vc[i]) <= 0.0001);
            assert(std.complex.abs(r2[i] - vc[i]) <= 0.0001);
            assert(std.complex.abs(r1[i] - r1[i]) <= 0.0001);
        }

        bool error =  false;
        try {
            auto err = new FourierMatrix!(uint, float)(); 
        }
        catch (AssertError e) {
            error = true;
        }
        assert(error);
    }

    write("Done.\n");
}

class DiagonalMatrix(S,T) : MatrixAbstract!(S,T) {
    T[] mat;
    
    /// Constructor
    pure
    this(in S size)
    {
        typeId = "DiagonalMatrix";
        rows = size; cols = size;
        mat = new T[size];
    }

    /// Simple constructor with random initialization
    this(in S size, in Tc randomBound)
    {
        this(size);
        static if (isComplexType!T) {
            foreach(i;0 .. size)
                mat[i] = complex(uniform(-randomBound, randomBound, rnd),
                                 uniform(-randomBound, randomBound, rnd));
        }
        else {
            foreach(i;0 .. size)
                mat[i] = uniform(-randomBound, randomBound, rnd);
        }
    }

    /// Copy-constructor
    pure
    this(in DiagonalMatrix M)
    {
        this(M.rows);
        foreach(i;0 .. rows)
            this.mat[i] = M.mat[i];
    }

    /// Constructor from list
    pure
    this(in T[] valarr)
    {
        this(cast(S) valarr.length);
        mat = valarr.dup;
    }


    const @property pure
    auto dup()
    {
        return new DiagonalMatrix(this);
    }

    @property const pure
    S length()
    {
        return cast(S) rows;
    }

    /// Assign Value to indices.
    pure
    void opIndexAssign(T value, in S i, in S j)
    {if (i == j) mat[i] = value;}
    /// Assign Value to index.
    pure
    void opIndexAssign(T value, in S i)
    {mat[i] = value;}

    /// Return value by indices.
    pure
    ref T opIndex(in S i, in S j)
    {return mat[i];}
    /// Return value by index.
    pure
    ref T opIndex(in S i)
    {return mat[i];}

    /// Operation +-*/ between Diagonal Matrix.
    const pure
    auto opBinary(string op)(in DiagonalMatrix other)
    if (op=="+" || op=="-" || op=="*" || op=="/")
    {
        auto res = new DiagonalMatrix(this);
        foreach(i;0 .. rows)
            mixin("res.mat[i] " ~ op ~ "= other.mat[i];");
        return res;
    }

    /// Operation-Assign +-*/ between Diagonal Matrix.
    pure
    void opOpAssign(string op)(in DiagonalMatrix other)
    if (op=="+" || op=="-" || op=="*" || op=="/")
    {
        foreach(i;0 .. rows)
            mixin("mat[i] " ~ op ~ "= other.mat[i];");
    }

    ///  Vector multiplication.
    const pure
    Vector!(S,T) opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        auto vres = v.dup;
        foreach(i; 0 .. v.length)
            vres[i] = mat[i] * vres[i];
        return vres;
    }

    const pure
    auto opBinary(string op)(in T[] other)
    if (op=="*")
    {
        auto res = new T[other.length];
        foreach(i;0 .. rows)
            res[i] = mat[i] * other[i];
        return res;
    }

    const pure
    Vector!(S,T) opBinaryRight(string op)(in Vector!(S,T) v)
    if (op=="/")
    {
        return new Vector!(S,T)(v.v / this);
    }

    const pure
    auto opBinaryRight(string op)(in T[] other)
    if (op=="/")
    {
        auto res = new T[other.length];
        foreach(i;0 .. rows)
            res[i] = other[i] / mat[i];
        return res;
    }

    /// Operation +-*/ on Matrix.
    const pure
    Matrix!(S,T) opBinary(string op)(in Matrix!(S,T) M)
    {
        static if (op=="+" || op=="-") {
            auto res = M.dup;
            foreach(i;0 .. M.rows)
                mixin("res[i,i] " ~ op ~ "= M[i,i];");
            return res;
        }
        else static if (op=="*" || op=="/"){
            // TODO : Implement
            auto res = M.dup;
            foreach(i;0 .. M.rows)
                mixin("res[i,i] " ~ op ~ "= M[i,i];");
            return res;
        }
        else static assert(0, "Binary operation '"~op~"' is not implemented.");
    }

    const pure
    Matrix!(S,T) opBinary(string op)(in Matrix!(S,T) M)
    if (op=="*" || op=="/")
    {
        auto res = M.dup;
        foreach(i;0 .. M.rows)
            mixin("res[i,i] " ~ op ~ "= M[i,i];");
        return res;
    }

    @property const
    T sum()
    {return mat.sum;}
}
unittest
{
    write("Unittest DiagonalMatrix ... ");
    {
        alias Diag = DiagonalMatrix!(uint, double);
        auto m1 = new Diag(4);
        auto m2 = new Diag([0, 1, 2, 3]);
        m1[0]=-0;m1[1]=-1;m1[2]=-2;m1[3]=-3;

        auto m3 = m1.dup;
        auto m4 = new Diag(m2);

        assert(m1[2] == -2);
        assert(m4.sum == m2.sum);
        assert(m3.sum == m1.sum);

        m3 += m2;
        assert(m3.sum == 0);

        m3 -= m4;
        assert(m3[3] == m1[3]);

        auto v = [10.0, 3.0, 5.0, 1.0, 8.0, 4.0, 6.0];
        auto w = m2 * v[0 .. 4];
        assert(w[0] == 0);
        assert(w[1] == 3);
    }
    {
        alias Diag = DiagonalMatrix!(uint, Complex!double);
        auto m1 = new Diag(4, 3.0);
        auto m2 = new Diag([complex(0), complex(1),
                            complex(2), complex(3)]);
        m1[0]=complex(-0);m1[1]=complex(-1);
        m1[2]=complex(-2);m1[3]=complex(-3);

        auto m3 = m1.dup;
        auto m4 = new Diag(m2);

        assert(m1[2] == complex(-2));
        assert(m4.sum == m2.sum);
        assert(m3.sum == m1.sum);

        m3 += m2;
        assert(m3.sum.abs < 0.0001);

        m3 -= m4;
        assert(m3[3] == m1[3]);
        assert(m3[1, 1] == m1[1]);
        assert(m1.length == 4);

        m1[1, 2] = complex(10000.0);
        assert(m3[1, 1] == m1[1]);
    }

    write("Done.\n");
}

class ReflectionMatrix(S,T) : MatrixAbstract!(S,T) {
    Vector!(S,T) vec;
    real invSqNormVec2 = 1.0;
    
    /+
     + We define a reflection matrix to be of the form:
     +  I - 2vv*/||v||^2
     + where I is the identity matrix
     + v is a complex vector
     + v* denotes the conjugate transpose of v
     + ||v||^2 is the euclidean norm of v
     +/

    /// Constructor
    pure
    this(in S size)
    {
        typeId = "ReflectionMatrix";
        rows = size; cols = size;
        vec = new Vector!(S,T)(size);
    }

    /// Simple constructor with random initialization
    this(in S size, in Tc randomBound)
    {
        this(size);

        static if (isComplexType!T) {
            foreach(S i;0 .. size)
                vec[i] = complex(uniform(-randomBound, randomBound, rnd),
                                 uniform(-randomBound, randomBound, rnd));
        }
        else {
            foreach(S i;0 .. size)
               vec[i] = uniform(-randomBound, randomBound, rnd);
        }
        compute_invSqNormVec2();
    }


    /// Copy-constructor
    this(in ReflectionMatrix dupl)
    {
        rows = dupl.vec.length; cols = dupl.vec.length;
        vec = dupl.vec.dup;
        invSqNormVec2 = dupl.invSqNormVec2;
    }

    /// Constructor from list.
    pure
    this(in T[] valarr)
    {
        rows = cast(S) valarr.length;
        cols = cast(S) valarr.length;
        vec = new Vector!(S,T)(valarr);
        compute_invSqNormVec2();
    }
    /// Constructor from Vector.
    pure
    this(in Vector!(S,T) valarr)
    {
        this(valarr.v);
    }


    const @property pure
    auto dup()
    {
        return new ReflectionMatrix(this);
    }

    /// Compute the norm (n) of the reflection vector and store -2n^-2
    pure
    void compute_invSqNormVec2()
    {
        invSqNormVec2 = -2*pow(vec.norm!"L2",-2);
    }

    @property const pure
    S length()
    {
        return cast(S) rows;
    }

    /+ Vector multiplication.
     + As we only store the vector that define the reflection
     + We can comme up with a linear-time matrix-vector multiplication.
     +/
    const pure
    Vector!(S,T) opBinary(string op)(in Vector!(S, T) v)
    if (op=="*")
    {
        return this * v.v;
    }

    const pure
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        T[] vres = v.dup;
        T s = vec.conjdot(v, vec);
        T[] tmp = vec.v.dup;
        s *= invSqNormVec2;
        foreach(i; 0 .. cols)
            tmp[i] = tmp[i] * s;
        vres[] += tmp[];
        return vres;
    }

    const pure
    Vector!(S,T) opBinaryRight(string op)(in Vector!(S, T) v)
    if (op=="/")
    {
        return v.v / this;
    }

    const pure
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        // This is not a bug !
        // The inverse of a reflection is the very same reflection.
        T[] vres = v.dup;
        T s = vec.conjdot(v, vec);
        T[] tmp = vec.v.dup;
        s *= invSqNormVec2;
        foreach(i; 0 .. cols)
            tmp[i] = tmp[i] * s;
        vres[] += tmp[];
        return vres;
    }

    const pure
    Matrix!(S,T) toMatrix()
    {
        auto res = new Matrix!(S,T)(rows, cols);
        T s;
        foreach(S i; 0 .. rows) {
            s = vec[i]*invSqNormVec2;
            foreach(S j; 0 .. cols){
                static if (isComplexType!T)
                    res[i,j] = s*vec[j].conj;
                else
                    res[i,j] = s*vec[j];
            }
            static if (isComplexType!T)
                res[i,i] = res[i,i] + cast(T) complex(1);
            else
                res[i,i] = res[i,i] + cast(T) 1;
        }
        return res;
    }
}
unittest
{
    write("Unittest Reflection ... ");
    {
        // Verification of the multiplication algorithm.
        alias Reflection = ReflectionMatrix!(uint, Complex!real);
        ReflectionMatrix!(uint, Complex!real) r1 = new Reflection(100, 1.0);
        ReflectionMatrix!(uint, Complex!real) r1bis = r1.dup;
        Matrix!(uint, Complex!real) m1 = r1.toMatrix();

        r1bis.vec -= r1.vec;
        assert(r1bis.vec.norm!"L2" <= 0.0001);

        auto v = new Vector!(uint, Complex!real)(100, 1.0);
        auto u = new Vector!(uint, Complex!real)(v);

        v.opOpAssign!"*"(r1);
        u.opOpAssign!"*"(m1);
        v -= u;
        assert(v.norm!"L2" < 0.0001);
    }
    {

        auto vref = new Vector!(uint, real)(100, 1.0);

        auto r1 = new ReflectionMatrix!(uint, real)(vref);
        auto r1bis = new ReflectionMatrix!(uint, real)(vref.v);
        Matrix!(uint, real) m1 = r1.toMatrix();

        r1bis.vec -= r1.vec;
        assert(r1bis.vec.norm!"L2" <= 0.0001);

        auto v = new Vector!(uint, real)(100, 1.0);
        auto u = new Vector!(uint, real)(v);

        v.opOpAssign!"*"(r1);
        u.opOpAssign!"*"(m1);
        v -= u;
        assert(v.norm!"L2" < 0.0001);

        assert(r1.length == 100);
        
    }
    writeln("Done.");
}


class PermutationMatrix(S,T) : MatrixAbstract!(S,T) {
    S[] perm;

    /// Constructor
    pure
    this(in S size)
    {
        typeId = "PermutationMatrix";
        rows = size; cols = size;
        perm = new S[size];
    }

    /// Simple constructor with random initialization
    /// Here the randomBound is not used and a simple
    /// random permutation is returned.
    this(in S size, in float randomBound)
    {
        this(size.iota.array);
        randomShuffle(perm);
    }


    /// Copy-constructor
    pure
    this(in PermutationMatrix dupl)
    {
        this(dupl.cols);
        perm = dupl.perm.dup;
    }

    /// Constructor from list (trusted to be a permutation)
    pure
    this(in S[] valarr)
    {
        this(cast(S) valarr.length);
        perm = valarr.dup;
    }


    const @property pure
    auto dup()
    {
        return new PermutationMatrix(this);
    }

    @property @nogc
    const pure
    S length()
    {
        return cast(S) rows;
    }

    const
    pure @safe  @nogc
    auto permute(size_t i)
    {
        return perm[i];
    }


    ///  Vector multiplication.
    const pure
    Vector!(S,T) opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        auto vres = new Vector!(S,T)(v.length);
        foreach(i; 0 .. v.length)
            vres[i] = v[perm[i]];
        return vres;
    }

    const pure
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        auto vres = new T[v.length];
        foreach(i; 0 .. v.length)
            vres[i] = v[perm[i]];
        return vres;
    }

    const pure
    Vector!(S,T) opBinaryRight(string op)(in Vector!(S,T) v)
    if (op=="/")
    {
        return new Vector!(S,T)(v.v / this);
    }

    const pure
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        auto vres = new T[v.length];
        foreach(i; 0 .. v.length)
            vres[perm[i]] = v[i];
        return vres;
    }
}
unittest
{
    write("Unittest Permutation ... ");
    
    alias Perm = PermutationMatrix!(uint, float);
    
    auto p = new Perm(1_000, 3);
    auto o = p.dup;

    assert(p.perm[4] < p.perm.length);
    p.perm[] -= o.perm[];
    assert(p.perm.sum == 0);

    assert(p.length == 1_000);
    
    write("Done.\n");
}

class Matrix(S,T) : MatrixAbstract!(S,T) {
    T[] mat;

    /// Simple constructor
    pure
    this(in S rows, in S cols)
    {
        typeId = "Matrix";
        mat = new T[rows*cols];
        this.rows = rows;
        this.cols = cols;
    }
    
    pure
    this(in S rows)
    {
        this(rows, rows);
    }

    this(in S rows, in Tc randomBound)
    {
        this(rows, rows, randomBound);
    }

    /// Simple constructor with random initialization
    this(in S rows, in S cols, in Tc randomBound)
    {
        this(rows, cols);
        static if (isComplexType!T) {
            foreach(i;0 .. mat.length)
                mat[i] = complex(uniform(-randomBound, randomBound, rnd),
                                 uniform(-randomBound, randomBound, rnd));
        }
        else {
            foreach(i;0 .. rows*cols)
                mat[i] = uniform(-randomBound, randomBound, rnd);
        }
    }

    /// Copy-constructor
    pure
    this(in Matrix dupl)
    {
        this(dupl.rows, dupl.cols);
        foreach(i;0 .. rows*cols) {
            mat[i] = dupl.mat[i];
        }
    }

    const @property pure
    auto dup()
    {
        return new Matrix(this);
    }

    @property const pure
    S length()
    {
        return cast(S) rows;
    }

    pure
    void opIndexAssign(T value, in S i, in S j)
    {mat[i*cols + j] = value;}

    /// Return value by index.
    pure const
    T opIndex(in S i, in S j)
    {return mat[i*cols + j];}
   
    /// Simple math operation without memory allocation.
    pure
    void opOpAssign(string op)(in Matrix other)
    {
             static if (op == "+") { mat[] += other.mat[]; }
        else static if (op == "-") { mat[] -= other.mat[]; }
        else static if (op == "*") { this  = this * other; }
        else static assert(0, "Operator "~op~" not implemented.");
    }

    const pure
    auto opBinary(string op)(in Matrix other)
    if (op == "+" || op == "-")
    {
        auto res = new Matrix(this);
        foreach(i;0 .. mat.length)
            mixin("res.mat[i] " ~ op ~ "= other.mat[i];");
        return res;
    }

    const pure
    auto opBinary(string op)(in Matrix other)
    if (op == "*")
    {
        enforce(cols == other.rows,
            "This is not a correct matrix multiplication");
        auto res = new Matrix(rows, other.cols);
        foreach(i;0 .. rows) {
            foreach(j; 0 .. other.cols) {
                T s;
                foreach(k; 0 .. cols){
                    s += mat[i*cols + k] * other.mat[k*other.cols + j];
                }
                res[i, j] = s;
            }
        }
        return res;
    }

    const pure
    auto opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        return new Vector!(S,T)(this * v.v);
    }

    const pure
    auto opBinary(string op)(in T[] v)
    if (op=="*")
    {
        auto res = new T[v.length];
        foreach(S i; 0 .. rows){
            T s = mat[i*cols]*v[0];
            foreach(S j; 1 .. cols)
                s += mat[i*cols + j]*v[j];
            res[i] = s;
        }
        return res;
    }
}
unittest
{
    auto m1 = new Matrix!(uint, Complex!float)(10, 30, 5.0f);
    auto m2 = m1.dup;
    auto m3 = new Matrix!(uint, Complex!float)(30, 5, 10.0f);
    auto m4 = new Matrix!(size_t, real)(100, 1.0);
    auto m5 = new Matrix!(size_t, real)(100);
    m5.mat = m4.mat.dup;

    m2 -= m1;
    assert(m2.mat.sum.abs < 0.1);

    m5 -= m4;
    assert(m5.mat.sum.abs < 0.1);

    m2 += m1;
    auto m6 = m1 * m3;
    assert(m6.length == m1.length);

    bool isFailed = false;
    try
    {
        auto m7 = m3 * m1;
    }
    catch (Exception e)
    {
        isFailed = true;
    }
    assert(isFailed);
}

