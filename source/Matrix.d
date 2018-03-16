module source.Matrix;

import std.algorithm;
import std.complex;
import std.exception : enforce;
import std.math;
import std.numeric : Fft, dotProduct;
import std.random;
import std.range : iota, array;
import std.string;

import source.Parameters;



class MatrixAbstract(S, T) : Parameter {
    private S rows, cols;
    protected string typeId;

    static if (T.stringof.startsWith("Complex"))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

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
        // TODO: Refactor. This is ugly but one can't simply use mixin here.
        auto tmptypeId = split(typeId, "!")[0];
        switch (tmptypeId)
        {
            case "BlockMatrix":
                return cast(BlockMatrix!(S,T)) this * v;
            case "UnitaryMatrix":
                static if (T.stringof.startsWith("Complex")) {
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
                static if (T.stringof.startsWith("Complex")) {
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
                static if (T.stringof.startsWith("Complex")) {
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
                static if (T.stringof.startsWith("Complex")) {
                    return v / cast(FourierMatrix!(S,T)) this;
                }
                else assert(0, "Fourier matrices must be of complex type.");
            case "Matrix":
                assert(0, "Division by a general matrix is not yet implemented.");
            default:
                assert(0, tmptypeId~" is not in the 'switch'
                                      clause of MatrixAbstract");
        }
    }
}
unittest
{
    import std.stdio: write;
    write("Unittest Matrix Abstract ... ");

    auto len = 1024;
    auto m1 = new PermutationMatrix!(ulong, Complex!real)(len/4, 1.0);
    auto m2 = new DiagonalMatrix!(ulong, Complex!real)(len/4, 1.0);
    auto m3 = new ReflectionMatrix!(ulong, Complex!real)(len/4, 1.0);
    auto m4 = new FourierMatrix!(ulong, Complex!real)(len/4);
    auto bm = new BlockMatrix!(ulong, Complex!real)(len, len/4, [m1,m2,m3,m4], false);
    
    auto list_mat = [m1, m2, m3, m4, bm];
    auto m1_hiden = list_mat[0];
    auto m2_hiden = list_mat[1];
    auto m3_hiden = list_mat[2];
    auto m4_hiden = list_mat[3];
    auto bm_hiden = list_mat[4];

    auto v = new Vector!(ulong, Complex!real)(len);
    auto w = new Vector!(ulong, Complex!real)(len/4);
    foreach(i; 0 .. len)
        v[i] = complex(cast(real)(i*2 - len/2), cast(real)(len/3 - i/3.0));


    auto v2 = bm_hiden * v;
    auto v3 = v2 / bm_hiden;

    v3 -= v;
    assert(v3.norm!"L2" < 1.0);
    v2 -= v;
    assert(v2.norm!"L2" > 1.0);

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
    
    this(in S size, in S size_blocks,
         MatrixAbstract!(S,T)[] blocks, bool randperm=false)
    {
        this(size, size, size_blocks, blocks, randperm);
    }

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

        import std.stdio: writeln;
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
    import std.stdio: write;
    write("Unittest Block Matrix ... ");

    auto len = 1024;
    auto m1 = new PermutationMatrix!(ulong, Complex!float)(len/4, 1.0);
    auto m2 = new DiagonalMatrix!(ulong, Complex!float)(len/4, 1.0);
    auto m3 = new ReflectionMatrix!(ulong, Complex!float)(len/4, 1.0);
    auto m4 = new FourierMatrix!(ulong, Complex!float)(len/4);
    auto bm = new BlockMatrix!(ulong, Complex!float)(len, len/4, [m1,m2,m3,m4], false);

    auto v = new Vector!(ulong, Complex!float)(len);
    foreach(i; 0 .. len)
        v[i] = complex(cast(float)(i*2 - len/2), cast(float)(len/3 - i/3.0));

    auto mem= v.dup;


    auto v2 = bm * v;
    auto v3 = v2 / bm;
    v3 -= v;
    assert(v3.norm!"L2" < 1.0);
    v2 -= v;
    assert(v2.norm!"L2" > 1.0);

    write("Done.\n");
}

class UnitaryMatrix(S, T) : MatrixAbstract!(S,T) {
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


    PermutationMatrix!(S,T) perm;
    FourierMatrix!(S,T) fourier;
    Vector!(S,Tc) params;

    /+ The params vector include in the following order:
       + 3 diagonal unitary complex matrices.
       + 2 reflection unitary complex matrices.

       This will provide us with a simple way to change these
       parameters.
     +/

    this(){}

    this(in S size)
    {
        typeId = "UnitaryMatrix";

        rows = size;
        cols = size;

        perm = new PermutationMatrix!(S,T)(size, 1.0);
        fourier = new FourierMatrix!(S,T)(size);
        params = new Vector!(S,Tc)(7*size);
    }

    this(in S size, in Tc randomBound)
    {
        this(size);

        foreach(i;0 .. params.length)
            params[i] = uniform(-randomBound, randomBound, rnd);
    }


    /// Apply the "num"th diagonal matrix on the given vector.
    const
    void applyDiagonal(ref T[] v, in int num)
    {
        // we use expi to convert each value to a complex number
        // the value is the angle of the complex number with radius 1.
        S start_index = num*rows;
        foreach(i; 0 .. rows)
            v[i] *= cast(T) std.complex.expi(params[start_index + i]);
    }    /// Apply the "num"th diagonal matrix on the given vector.
    
    const
    void applyDiagonalInv(ref T[] v, in int num)
    {
        // we use expi to convert each value to a complex number
        // the value is the angle of the complex number with radius 1.
        S start_index = num*rows;
        foreach(i; 0 .. rows)
            v[i] /= cast(T) std.complex.expi(params[start_index + i]);
    }

    /// Apply the "num"th reflection matrix on the given vector.
    const
    void applyReflection(ref T[] v, in int num)
    {
        // The '+3' is because the diagonal matrices are first
        // in the params array.
        // The '*2' is because each reflection matrix need
        // 2 * rows parameters to be defined.
        S start_index = (2*num + 3)*rows;
        T[] tmp_a = dotProduct(v, params[start_index .. start_index + rows]);
        T[] tmp_b = dotProduct(v, params[start_index + rows .. start_index + 2*rows]);
        T[] tmp = new T[rows];
        T a, b;
        Tc invsq = -2 / (dotProduct(params[start_index .. start_index + rows],
                                    params[start_index .. start_index + rows])
                      + (dotProduct(params[start_index + rows .. start_index + 2*rows],
                                    params[start_index + rows .. start_index + 2*rows])));

        foreach(i; 0 .. rows){
            a = params[start_index + i]];
            b = params[start_index + i + rows]];
            tmp[i] = complex(a*tmp_a + b*tmp_b, b*tmp_a - a*tmp_b) * invsq;
        }

        v[] -= tmp[];
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
}

class FourierMatrix(S,T) : MatrixAbstract!(S,T) {
    Fft objFFT;
    
    this(S size)
    {
        typeId = "FourierMatrix";
        rows = size;
        cols = size;
        objFFT = new Fft(size);
    };


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
    import std.stdio : write;
    import std.datetime;
    write("Unittest Fourrier Matrix ... ");
    {
        alias Fourier = FourierMatrix!(uint, Complex!double);
        auto f = new Fourier(1024);
        auto v = new Complex!double[2048];
        auto vc = v[15 .. 1039];

        assert(vc.length == 1024);

        auto rnd_tmp = Random(cast(uint) ((Clock.currTime()
                         - SysTime(unixTimeToStdTime(0))).total!"msecs"));

        foreach(i;0 .. 1024)
            vc[i] = complex(uniform(-1.0, 1.0, rnd_tmp),
                            uniform(-1.0, 1.0, rnd_tmp));

        auto r1 = f * (vc / f);
        auto r2 = (f * vc) / f;

        foreach(i;0 .. 1024){
            assert(std.complex.abs(r1[i] - vc[i]) <= 0.0001);
            assert(std.complex.abs(r2[i] - vc[i]) <= 0.0001);
            assert(std.complex.abs(r1[i] - r1[i]) <= 0.0001);
        }
    }

    write("Done.\n");
}

class DiagonalMatrix(S,T) : MatrixAbstract!(S,T) {
    T[] mat;
    
    /// Constructor
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
        static if (T.stringof.startsWith("Complex")) {
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
    this(in DiagonalMatrix dupl)
    {
        this(dupl.cols);
        mat = dupl.mat.dup;
    }

    /// Constructor from list
    this(in T[] valarr)
    {
        this(cast(S) valarr.length);
        mat = valarr.dup;
    }

    @property const
    S length()
    {
        return cast(S) rows;
    }

    /// Assign Value to indices.
    void opIndexAssign(T value, in S i, in S j)
    {if (i == j) mat[i] = value;}
    /// Assign Value to index.
    void opIndexAssign(T value, in S i)
    {mat[i] = value;}

    /// Return value by indices.
    ref T opIndex(in S i, in S j)
    {return mat[i];}
    /// Return value by index.
    ref T opIndex(in S i)
    {return mat[i];}

    /// Return a duplicate.
    @property const
    auto dup()
    {
        auto res = new DiagonalMatrix(rows);
        foreach(i;0 .. rows)
            res.mat[i] = mat[i];

        return res;
    }

    /// Operation +-*/ between Diagonal Matrix.
    const
    auto opBinary(string op)(in DiagonalMatrix other)
    if (op=="+" || op=="-" || op=="*" || op=="/")
    {
        auto res = new DiagonalMatrix(this);
        foreach(i;0 .. rows)
            mixin("res.mat[i] " ~ op ~ "= other.mat[i];");
        return res;
    }

    /// Operation-Assign +-*/ between Diagonal Matrix.
    void opOpAssign(string op)(in DiagonalMatrix other)
    if (op=="+" || op=="-" || op=="*" || op=="/")
    {
        foreach(i;0 .. rows)
            mixin("mat[i] " ~ op ~ "= other.mat[i];");
    }

    ///  Vector multiplication.
    const
    Vector!(S,T) opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        auto vres = v.dup;
        foreach(i; 0 .. v.length)
            vres[i] = mat[i] * vres[i];
        return vres;
    }

    const
    auto opBinary(string op)(in T[] other)
    if (op=="*")
    {
        auto res = new T[other.length];
        foreach(i;0 .. rows)
            res[i] = mat[i] * other[i];
        return res;
    }

    const
    Vector!(S,T) opBinaryRight(string op)(in Vector!(S,T) v)
    if (op=="/")
    {
        return new Vector!(S,T)(v.v / this);
    }

    const
    auto opBinaryRight(string op)(in T[] other)
    if (op=="/")
    {
        auto res = new T[other.length];
        foreach(i;0 .. rows)
            res[i] = other[i] / mat[i];
        return res;
    }

    /// Operation +-*/ on Matrix.
    const
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

    const
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
    import std.stdio : write;
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
     +	I - 2vv*/||v||^2
     + where I is the identity matrix
     + v is a complex vector
     + v* denotes the conjugate transpose of v
     + ||v||^2 is the euclidean norm of v
     +/

    /// Constructor
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

        static if (T.stringof.startsWith("Complex")) {
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
    this(in T[] valarr)
    {
        rows = cast(S) valarr.length;
        cols = cast(S) valarr.length;
        vec = new Vector!(S,T)(valarr);
        compute_invSqNormVec2();
    }
    /// Constructor from Vector.
    this(in Vector!(S,T) valarr)
    {
        this(valarr.v);
    }

    /// Compute the norm (n) of the reflection vector and store -2n^-2
    void compute_invSqNormVec2()
    {
        invSqNormVec2 = -2*pow(vec.norm!"L2",-2);
    }

    /// Return a duplicate.
    @property const
    auto dup()
    {
    	return new ReflectionMatrix(this);
    }

    @property const
    S length()
    {
        return cast(S) rows;
    }

    /+ Vector multiplication.
     + As we only store the vector that define the reflection
     + We can comme up with a linear-time matrix-vector multiplication.
     +/
    const
    Vector!(S,T) opBinary(string op)(in Vector!(S, T) v)
    if (op=="*")
    {
        return this * v.v;
    }

    const
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

    const
    Vector!(S,T) opBinaryRight(string op)(in Vector!(S, T) v)
    if (op=="/")
    {
        return v.v / this;
    }

    const
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

    const
    Matrix!(S,T) toMatrix()
    {
        auto res = new Matrix!(S,T)(rows, cols);
        T s;
        foreach(S i; 0 .. rows) {
            s = vec[i]*invSqNormVec2;
            foreach(S j; 0 .. cols){
                static if (T.stringof.startsWith("Complex"))
                    res[i,j] = s*vec[j].conj;
                else
                    res[i,j] = s*vec[j];
            }
            static if (T.stringof.startsWith("Complex"))
                res[i,i] = res[i,i] + cast(T) complex(1);
            else
                res[i,i] = res[i,i] + cast(T) 1;
        }
        return res;
    }
}
unittest
{
    import std.stdio : writeln, write;
    import source.Parameters;
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
    this(in PermutationMatrix dupl)
    {
        this(dupl.cols);
        perm = dupl.perm.dup;
    }

    /// Constructor from list (trusted to be a permutation)
    this(in S[] valarr)
    {
        this(cast(S) valarr.length);
        perm = valarr.dup;
    }

    /// Return a duplicate.
    @property
    const
    auto dup()
    {
        return new PermutationMatrix(this);
    }

    @property @nogc
    const
    S length()
    {
        return cast(S) rows;
    }

    const
    auto permute(size_t i)
    {
        return perm[i];
    }


    ///  Vector multiplication.
    const
    Vector!(S,T) opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        auto vres = new Vector!(S,T)(v.length);
        foreach(i; 0 .. v.length)
            vres[i] = v[perm[i]];
        return vres;
    }

    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        auto vres = new T[v.length];
        foreach(i; 0 .. v.length)
            vres[i] = v[perm[i]];
        return vres;
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
        auto vres = new T[v.length];
        foreach(i; 0 .. v.length)
            vres[perm[i]] = v[i];
        return vres;
    }
}
unittest
{
    import std.stdio : write;
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
    this(in S rows, in S cols)
    {
        typeId = "Matrix";
        mat = new T[rows*cols];
        this.rows = rows;
        this.cols = cols;
    }
    
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
        static if (T.stringof.startsWith("Complex")) {
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
    this(in Matrix dupl)
    {
        this(dupl.rows, dupl.cols);
        foreach(i;0 .. rows*cols) {
            mat[i] = dupl.mat[i];
        }
    }

    @property const
    S length()
    {
        return cast(S) rows;
    }

    void opIndexAssign(T value, in S i, in S j)
    {mat[i*cols + j] = value;}

    /// Return value by index.
    const T opIndex(in S i, in S j)
    {return mat[i*cols + j];}
   
    /// Simple math operation without memory allocation.
    void opOpAssign(string op)(in Matrix other)
    {
             static if (op == "+") { mat[] += other.mat[]; }
        else static if (op == "-") { mat[] -= other.mat[]; }
        else static if (op == "*") { this  = this * other; }
        else static assert(0, "Operator "~op~" not implemented.");
    }

    @property const
    auto dup()
    {
        return new Matrix!(S,T)(this);
    }

    const
    auto opBinary(string op)(in Matrix other)
    if (op == "+" || op == "-")
    {
        auto res = new Matrix(this);
        foreach(i;0 .. mat.length)
            mixin("res.mat[i] " ~ op ~ "= other.mat[i];");
        return res;
    }

    const
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

    const
    auto opBinary(string op)(in Vector!(S,T) v)
    if (op=="*")
    {
        auto res = this * v.v;
        return res;
    }

    const
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

