module source.Matrix;

import std.algorithm;
import std.complex;
import std.conv: to;
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

/++ A dot product function. We can make use of vectorization/fma 
 +  optimization when dealing with non-complex number.
 +
 + TODO:
 +      -implment optimization
 +      -make sure it is correctly used in the library. 
 +/
pure @safe 
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

/++ MatrixAbstract class.
 + 
 +  An abstract class of matrix to create array of heterogenous matrix
 +  and create general-purpose function. It is mostly a wrapper.
 +/
class MatrixAbstract(T) : Parameter {
    // TODO: make member private.
    size_t rows, cols;

    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    // We have to constructor to allow the existence of a pure and impure one
    // TODO: bool _b might not be useful because pureness is differentiate them.
    pure @safe
    this(){}
    
    @safe
    this(bool _b){
        super(true);
    }

    // Return a duplicate of the matrix. It should actually call the dup function
    // of its type (FourierMatrix, UnitaryMatrix, ...).
    const @property
    MatrixAbstract!T dup()
    {
        assert(0, "'"~typeId~"': function 'dup' not implemented");
    }

    // Simple Matrix-Vector Multiplication.
    const
    Vector!T opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        return new Vector!T(this * v.v);
    }

    // Wrapper Matrix-Vector Multiplication.
    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");
        // TODO: Refactor. This is ugly but one can't simply use mixin here.
        switch (typeId)
        {
            case "BlockMatrix":
                return cast(BlockMatrix!T) this * v;
            case "UnitaryMatrix":
                static if (is(Complex!T : T)) {
                    return cast(UnitaryMatrix!T) this * v;
                }
                else assert(0, "Unitary matrices must be of complex type.");
            case "DiagonalMatrix":
                return cast(DiagonalMatrix!T) this * v;
            case "ReflectionMatrix":
                return cast(ReflectionMatrix!T) this * v;
            case "PermutationMatrix":
                return cast(PermutationMatrix!T) this * v;
            case "FourierMatrix":
                static if (is(Complex!T : T)) {
                    return cast(FourierMatrix!T) this * v;
                }
                else assert(0, "Fourier matrices must be of complex type.");
            case "Matrix":
                return cast(Matrix!T) this * v;
            default:
                assert(0, "'"~typeId~"' is not in the 'switch' "~
                                      "clause of MatrixAbstract");
        }
    }

    // Simple inverse(Matrix)-Vector Multiplication.
    const
    Vector!T opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        auto res = new Vector!T(v.v / this);
        return res;
    }

    // Wrapper inverse(Matrix)-Vector Multiplication.
    const
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        enforce(v.length == cols, "Matrix-Vector division: dimensions mismatch.");
        // TODO: Refactor.
        switch (typeId)
        {
            case "BlockMatrix":
                return v / cast(BlockMatrix!T) this;
            case "UnitaryMatrix":
                static if (is(Complex!T : T)) {
                    return v / cast(UnitaryMatrix!T) this;
                }
                else assert(0, "Unitary matrices must be of complex type.");
            case "DiagonalMatrix":
                return v / cast(DiagonalMatrix!T) this;
            case "ReflectionMatrix":
                return v / cast(ReflectionMatrix!T) this;
            case "PermutationMatrix":
                return v / cast(PermutationMatrix!T) this;
            case "FourierMatrix":
                static if (is(Complex!T : T)) {
                    return v / cast(FourierMatrix!T) this;
                }
                else assert(0, "Fourier matrices must be of complex type.");
            case "Matrix":
                assert(0, "Division by a general matrix
                           is not yet implemented.");
            default:
                assert(0, "'"~typeId~"' is not in the 'switch' "~
                                      "clause of MatrixAbstract");
        }
    }
}
unittest
{
    write("Unittest: Matrix: Abstract ... ");

    // We create a badly constructed Matrix Class.
    // This will result in an error => You can't create your own Matrix.
    class ErrorMatrix(T) : MatrixAbstract!T {
        this(immutable size_t _r, immutable size_t _c) {
            typeId = "ErrorMatrix";
            rows = _r;
            cols = _c;
        }
    }

    auto len = 1024;
    auto m1 = new PermutationMatrix!(Complex!real)(len/4, 1.0);
    auto m2 = new DiagonalMatrix!(Complex!real)(len/4, 1.0);
    auto m3 = new ReflectionMatrix!(Complex!real)(len/4, 1.0);
    auto m4 = new FourierMatrix!(Complex!real)(len/4);
    auto bm = new BlockMatrix!(Complex!real)(len, len/4, [m1,m2,m3,m4],
                                                    false);
    auto um = new UnitaryMatrix!(Complex!real)(len, 3.14159265351313);
    auto mm = new Matrix!(Complex!real)(len, 5.6);
    auto em = new ErrorMatrix!(Complex!real)(len, len);

    // The type of list_mat will be "MatrixAbstract!T[]" as a
    // generalization of each matrices. Hence, every **_hyde matrices
    // will be of type "MatrixAbstract!T".
    auto list_mat = [m1, m2, m3, m4, bm, um, mm, em];
    auto m1_hyde = list_mat[0];
    auto m2_hyde = list_mat[1];
    auto m3_hyde = list_mat[2];
    auto m4_hyde = list_mat[3];
    auto bm_hyde = list_mat[4];
    auto um_hyde = list_mat[5];
    auto mm_hyde = list_mat[6];
    auto em_hyde = list_mat[7];


    // dup
    {
        auto dm1 = m1_hyde.dup;
        auto dm2 = m2_hyde.dup;
        auto dm3 = m3_hyde.dup;
        auto dm4 = m4_hyde.dup;
        auto dbm = bm_hyde.dup;
        auto dum = um_hyde.dup;
        auto dmm = mm_hyde.dup;

        auto v1 = new Vector!(Complex!real)(len/4, 1.0);
        auto v2 = new Vector!(Complex!real)(len, 1.0);

        Vector!(Complex!real) res1, res2;

        res1 = dm1*v1;
        res2 = m1*v1;
        res1 -= res2;
        assert(res1.norm!"L2" <= 0.0001);

        res1 = dm2*v1;
        res2 = m2*v1;
        res1 -= res2;
        assert(res1.norm!"L2" <= 0.0001);

        res1 = dm3*v1;
        res2 = m3*v1;
        res1 -= res2;
        assert(res1.norm!"L2" <= 0.0001);

        res1 = dm4*v1;
        res2 = m4*v1;
        res1 -= res2;
        assert(res1.norm!"L2" <= 0.0001);

        res1 = dbm*v2;
        res2 = bm*v2;
        res1 -= res2;
        assert(res1.norm!"L2" <= 0.0001);

        res1 = dum*v2;
        res2 = um*v2;
        res1 -= res2;
        assert(res1.norm!"L2" <= 0.0001);

        bool error = false;
        try {
            auto dem = em_hyde.dup;
        }
        catch (AssertError e) { error = true; }
        assert(error);
    }

    // We test the multiplication for all Matrix type.
    // This is to make sure, a MatrixAbstract!T will call the right
    // multiplication algorithm in function of its true type (Fourier, Permutation, ...)
    auto mr = new Matrix!real(len, 5.6);
    auto dr = new DiagonalMatrix!real(len, 5.6);

    auto list_mat2 = [mr, dr];
    auto mr_hyde = list_mat[0];

    auto v = new Vector!(Complex!real)(len);
    auto w = new Vector!(Complex!real)(len/4);
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

    // class ErrorMatrix is not handled by MatrixAbstract in multiplication.
    bool error = false;
    try {
        auto ver = em_hyde * v;
    }
    catch (AssertError e) { error = true; }
    assert(error);

    // Same as above, but with division.
    error = false;
    try {
        auto ver = v / em_hyde;
    }
    catch (AssertError e) { error = true; }
    assert(error);

    // Division by a Matrix is not implemented.
    error = false;
    try {
        auto ver = v / mm_hyde;
    }
    catch (AssertError e) { error = true; }
    assert(error);

    // Unitary Matrix must be of complex type.
    error = false;
    try {
    	auto mpp = new Matrix!float(4, 4, 0.1);
    	mpp.typeId = "UnitaryMatrix";
    	MatrixAbstract!float mmp = mpp;
    	auto vvv = new Vector!float(4, 0.1);

    	auto res = mmp * vvv;
    }
    catch (AssertError e) { error = true; }
    assert(error);

    // Fourier Matrix must be of complex type.
    error = false;
    try {
    	auto mpp = new Matrix!float(4, 4, 0.1);
    	mpp.typeId = "FourierMatrix";
    	MatrixAbstract!float mmp = mpp;
    	auto vvv = new Vector!float(4, 0.1);

    	auto res = mmp * vvv;
    }
    catch (AssertError e) { error = true; }
    assert(error);

    write("Done.\n");
}

/++ Block Matrix class.
 +
 +  Block matrix are used as a way to obtain "sparse" matrix (with a lot of zeros)
 +  while controlling certain measure of the matrix such as its spectrum.
 +
 +  For a general introduction, see: https://en.wikipedia.org/wiki/Block_matrix
 +
 +  Note: The blcok matrices you can create using this class can only have one of
 +        the three format shown below.
 + 
 +  We present here the three different kinds of block matrix we
 +  can have. In the following, each '0' represent a zero square
 +  matrix of size 'size_blocks' and each number [1-9] represent
 +  a different square matrix of the same size.
 +
 + 1/ Square block matrix:
 +
 +      1 0 0 0 0
 +      0 2 0 0 0
 +      0 0 3 0 0
 +      0 0 0 4 0
 +      0 0 0 0 5
 +
 +  2/ Rectangular block matrix with more columns:
 +
 +      1 0 0 4 0
 +      0 2 0 0 5
 +      0 0 3 0 0
 +
 +  2/ Rectangular block matrix with more rows:
 +
 +      1 0 0
 +      0 2 0
 +      0 0 3
 +      4 0 0
 +      0 5 0
 +/
class BlockMatrix(T) : MatrixAbstract!T {
	// List of matrices.
    MatrixAbstract!T[] blocks;
    // Permutation, can be "turned off" with randperm = false.
    PermutationMatrix!T P, Q;

    size_t size_blocks;
    size_t num_blocks;
    size_t size_out;
    size_t size_in;

    // The two permutations are needed here so that 
    // every nodes can be connected to any others.
    // It is a way to "shuffle" the block matrix
    // and so to have a sparse matrix with the properties
    // of the block matrix (e.g. unitary).

    pure @safe
    this(){typeId = "BlockMatrix";}

    pure @safe
    this(in size_t size_in, in size_t size_out, in size_t size_blocks)
    {
        typeId = "BlockMatrix";
        enforce(size_out%size_blocks == 0,
                "'size_out' must be a multiple of 'size_blocks'.");
        enforce(size_in%size_blocks == 0,
                "'size_in' must be a multiple of 'size_blocks'.");
        rows = size_out;
        cols = size_in;

        this.size_blocks = size_blocks;
        size_t maxsize = size_out; if (maxsize<size_in) maxsize=size_in;
        this.num_blocks = maxsize/size_blocks;
        this.size_out = size_out;
        this.size_in = size_in;
    }

    @safe
    this(in size_t size_in, in size_t size_out, in size_t size_blocks,
         MatrixAbstract!T[] blocks, bool randperm=false)
    {
        this(size_in, size_out, size_blocks);
        this.blocks = blocks;

        if (randperm) {
            P = new PermutationMatrix!T(size_in, 1.0);
            Q = new PermutationMatrix!T(size_out, 1.0);
        }
        else {
            P = new PermutationMatrix!T(size_in.iota.array);
            Q = new PermutationMatrix!T(size_out.iota.array);
        }
    }

    @safe
    this(in size_t size_in, in size_t size_blocks,
         MatrixAbstract!T[] blocks, bool randperm=false)
    {
        this(size_in, size_in, size_blocks, blocks, randperm);
    }

    override
    const @property
    BlockMatrix!T dup()
    {
        auto res = new BlockMatrix!T(size_in, size_out, size_blocks);
        res.P = P.dup;
        res.Q = Q.dup;
        res.blocks = new MatrixAbstract!T[res.num_blocks];

        foreach(i; 0 .. res.num_blocks)
            res.blocks[i] = blocks[i].dup;

        return res;
    }

    const
    auto opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        auto res = this * v.v;
        return new Vector!T(res);
    }

    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");
        T[] vec = this.P * v;

        size_t blocks_in = size_in / size_blocks;
        size_t blocks_out = size_out / size_blocks;

        T[] res = new T[size_out];
        T[] s;
        size_t index;

        foreach(size_t b; 0 .. blocks_out) {
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
    auto opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        return new Vector!T(v.v / this);
    }

    const
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        enforce(v.length == cols, "Matrix-Vector division: dimensions mismatch.");
        enforce(size_out == size_in, "Inverse of rectangular
                                      block matrix is not implemented");
        T[] vec = v / this.Q;

        size_t blocks_in = size_in / size_blocks;
        size_t blocks_out = size_out / size_blocks;

        T[] res = new T[size_out];

        foreach(size_t b; 0 .. blocks_out) {
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
    write("                  Block ... ");

    auto len = 1024;
    auto m0 = new BlockMatrix!float();
    auto m1 = new PermutationMatrix!(Complex!float)(len/4, 1.0);
    auto m2 = new DiagonalMatrix!(Complex!float)(len/4, 1.0);
    auto m3 = new ReflectionMatrix!(Complex!float)(len/4, 1.0);
    auto m4 = new FourierMatrix!(Complex!float)(len/4);
    auto bm = new BlockMatrix!(Complex!float)(len, len/4, [m1,m2,m3,m4],
                                                     false);

    auto v = new Vector!(Complex!float)(len);
    foreach(i; 0 .. len)
        v[i] = complex(cast(float)(i*2 - len/2), cast(float)(len/3 - i/3.0));

    auto mem = v.dup;


    auto v2 = bm * v;
    auto v3 = v2 / bm;
    v3 -= v;
    assert(v3.norm!"L2" < 1.0);
    v2 -= v;
    assert(v2.norm!"L2" > 1.0);

    assertThrown(new BlockMatrix!real(4u, 4u, 3u));
    assertThrown(new BlockMatrix!real(4u, 3u, 3u));

    write("Done.\n");
}

/++ Unitary Matrix class.
 +
 +  TODO: take non-complex number as input.
 +
 +      This matrix is defined in 
 +          Unitary Evolution Recurrent Neural Networks,
 +          Arjovsky, Shah, Bengio
 +          (http://proceedings.mlr.press/v48/arjovsky16.pdf)
 +
 +      Its aim is to allow one to learn a unitary matrix that
 +      depends on number of parameters that is linear relatively
 +      to the size of matrix.
 +
 +      The resulting function is:
 +          D_3 * R_2 * invF * D_2 * P * R_1 * F * D_1
 +
 +      where D_i are diagonal unitary complex matrices,
 +            R_i are reflection unitary complex matrices,
 +            F and invF are respectivelly the fourier
 +              and inverse fourier transform,
 +            P is a permutation matrix.
 +/
class UnitaryMatrix(T) : MatrixAbstract!T
if (is(Complex!T : T))
{
    static assert(is(Complex!T : T),
               "UnitaryMatrix must be complex-valued.");

    PermutationMatrix!T perm;
    FourierMatrix!T fourier;
    Tc[] params;


    /+ The params vector include in the following order:
       + 3 diagonal unitary complex matrices.
       + 2 reflection unitary complex matrices.

       This will provide us with a simple way to change these
       parameters.
     +/

    pure @safe
    this() {}

    this(in size_t size)
    {   
        this();
        assert(std.math.isPowerOf2(size), "Size of Unitary Matrix
                                           must be a power of 2.");
        typeId = "UnitaryMatrix";

        rows = size;
        cols = size;

        perm = new PermutationMatrix!T(size ,1.0);
        fourier = new FourierMatrix!T(size);
        params = new Tc[7*size];
    }

    this(in size_t size, in Tc randomBound)
    {
        this(size);

        if (randomBound <= 0)
            throw new Exception("'randomBound' must be > 0");
	    foreach(i;0 .. params.length)
	        params[i] = uniform(-randomBound, randomBound, rnd);
    }
    
    this(in UnitaryMatrix M)
    {
        typeId = "UnitaryMatrix";
        this();
        rows = M.rows;
        cols = M.cols;
        perm = M.perm.dup;
        fourier = M.fourier.dup;
        fourier = M.fourier.dup;
        params = M.params.dup;
    }

    override
    const @property 
    UnitaryMatrix!T dup()
    {
        return new UnitaryMatrix(this);
    }


    /// Apply the "num"th diagonal matrix on the given vector.
    const pure @safe
    void applyDiagonal(ref T[] v, in size_t num)
    {
        // we use expi to convert each value to a complex number
        // the value is the angle of the complex number with radius 1.
        size_t start_index = num*rows;
        foreach(i; 0 .. rows)
            v[i] *= cast(T) std.complex.expi(params[start_index + i]);
    }    /// Apply the "num"th diagonal matrix on the given vector.
    
    const pure @safe
    void applyDiagonalInv(ref T[] v, in size_t num)
    {
        // we use expi to convert each value to a complex number
        // the value is the angle of the complex number with radius 1.
        size_t start_index = num*rows;
        foreach(i; 0 .. rows)
            v[i] /= cast(T) std.complex.expi(params[start_index + i]);
    }

    /// Apply the "num"th reflection matrix on the given vector.
    const pure @safe
    auto applyReflection(ref T[] v, in size_t num)
    {
        // The '+3' is because the diagonal matrices are first
        // in the params array.
        // The '*2' is because each reflection matrix need
        // 2 * rows parameters to be defined.
        size_t start_index = (2*num + 3)*rows;
        size_t start_indexPlRows = start_index + rows;
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
    Vector!T opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        return new Vector!T(this * v.v);
    }

    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");

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
    Vector!T opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        return new Vector!T(v.v / this);
    }

    const
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        enforce(v.length == cols, "Matrix-Vector division: dimensions mismatch.");
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
    write("                  Unitary ... ");
    {
        auto m0 = new UnitaryMatrix!(Complex!float)();
        auto m = new UnitaryMatrix!(Complex!double)(1024, 9.0);
        auto n= m.dup;
        auto v = new Vector!(Complex!double)(1024, 1.2);

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

        assertThrown(new UnitaryMatrix!(Complex!float)(4, 0.0));
    }

    write("Done.\n");
}

/++ Fourier Matrix class.
 +
 +  Implement the fourier matrix.
 +
 +/
class FourierMatrix(T) : MatrixAbstract!T
if (is(Complex!T : T))
{
    Fft objFFT;
    
    this(size_t size)
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

    override
    const @property
    FourierMatrix!T dup()
    {
        return new FourierMatrix(this);
    }

    const
    Vector!T opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        return new Vector!T(this * v.v);
    }

    const
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");
        return objFFT.fft!Tc(v);
    }


    const
    Vector!T opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        enforce(v.length == cols, "Matrix-Vector divison: dimensions mismatch.");
        return new Vector!T(v.v / this);
    }

    const
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        return objFFT.inverseFft!Tc(v);
    }
}
unittest
{
    write("                  Fourrier ... ");
    {
        alias Fourier = FourierMatrix!(Complex!double);
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
    }

    write("Done.\n");
}


/++ Diagonal matrix class.
 +
 + TODO: There mmight be too much things, we could need to remove some methods.
 +
 +/
class DiagonalMatrix(T) : MatrixAbstract!T {
    T[] params;
    
    /// Constructor
    pure @safe
    this(in size_t size)
    {
        typeId = "DiagonalMatrix";
        rows = size; cols = size;
        params = new T[size];
    }

    /// Simple constructor with random initialization
    @safe
    this(in size_t size, in Tc randomBound)
    {
        this(size);

        if (randomBound < 0)
            throw new Exception("'randomBound' must be >= 0");
        if (randomBound.abs == 0) {
            foreach(i; 0 .. params.length)
                params[i] = to!(T)(0);
        }
        else {
            static if (is(Complex!T : T)) {
                foreach(i;0 .. size)
                    params[i] = complex(uniform(-randomBound, randomBound, rnd),
                                     uniform(-randomBound, randomBound, rnd));
            }
            else {
                foreach(i;0 .. size)
                    params[i] = uniform(-randomBound, randomBound, rnd);
            }
        }
    }

    /// Copy-constructor
    pure @safe
    this(in DiagonalMatrix M)
    {
        this(M.rows);
        foreach(i;0 .. rows)
            this.params[i] = M.params[i];
    }

    /// Constructor from list
    pure @safe
    this(in T[] valarr)
    {
        this(valarr.length);
        params = valarr.dup;
    }


    override
    const @property
    DiagonalMatrix!T dup()
    {
        return new DiagonalMatrix(this);
    }

    @property const pure @safe
    size_t length()
    {
        return rows;
    }

    /// Assign Value to indices.
    pure @safe
    void opIndexAssign(T value, in size_t i, in size_t j)
    {if (i == j) params[i] = value;}
    /// Assign Value to index.
    pure @safe
    void opIndexAssign(T value, in size_t i)
    {params[i] = value;}

    /// Return value by indices.
    pure @safe
    ref T opIndex(in size_t i, in size_t j)
    {return params[i];}
    /// Return value by index.
    pure @safe
    ref T opIndex(in size_t i)
    {return params[i];}

    /// Operation +-*/ between Diagonal Matrix.
    const pure @safe
    auto opBinary(string op)(in DiagonalMatrix other)
    if (op=="+" || op=="-" || op=="*" || op=="/")
    {
        auto res = new DiagonalMatrix(this);
        foreach(i;0 .. rows)
            mixin("res.params[i] " ~ op ~ "= other.params[i];");
        return res;
    }

    /// Operation-Assign +-*/ between Diagonal Matrix.
    pure @safe
    void opOpAssign(string op)(in DiagonalMatrix other)
    if (op=="+" || op=="-" || op=="*" || op=="/")
    {
        foreach(i;0 .. rows)
            mixin("params[i] " ~ op ~ "= other.params[i];");
    }

    ///  Vector multiplication.
    const pure @safe
    Vector!T opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        return new Vector!T(this * v.v);
    }

    const pure @safe
    auto opBinary(string op)(in T[] other)
    if (op=="*")
    {
        enforce(other.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");
        auto res = new T[other.length];
        foreach(i;0 .. rows)
            res[i] = params[i] * other[i];
        return res;
    }

    const pure @safe
    Vector!T opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        return new Vector!T(v.v / this);
    }

    const pure @safe
    auto opBinaryRight(string op)(in T[] other)
    if (op=="/")
    {
        enforce(other.length == cols, "Matrix-Vector division: dimensions mismatch.");
        auto res = new T[other.length];
        foreach(i;0 .. rows)
            res[i] = other[i] / params[i];
        return res;
    }

    /// Operation +-*/ on Matrix.
    const pure @safe
    Matrix!T opBinary(string op)(in Matrix!T M)
    {
        static if (op=="+" || op=="-") {
            auto res = M.dup;
            foreach(i;0 .. M.rows)
                mixin("res[i,i] " ~ op ~ "= M[i,i];");
            return res;
        }
        else static if (op=="*" || op=="/"){
            auto res = M.dup;
            foreach(i;0 .. M.rows)
                mixin("res[i,i] " ~ op ~ "= M[i,i];");
            return res;
        }
        else static assert(0, "Binary operation '"~op~"' is not implemented.");
    }

    const pure @safe
    Matrix!T opBinary(string op)(in Matrix!T M)
    if (op=="*" || op=="/")
    {
        auto res = M.dup;
        foreach(i;0 .. M.rows)
            mixin("res[i,i] " ~ op ~ "= M[i,i];");
        return res;
    }

    @property const @safe
    T sum()
    {
        return params.sum;
    }
}
unittest
{
    write("                  Diagonal ... ");
    {
        alias Diag = DiagonalMatrix!double;
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
        alias Diag = DiagonalMatrix!(Complex!double);
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

    assertThrown(new DiagonalMatrix!float(4, -6.0));
    auto d = new DiagonalMatrix!real(1000, 0.0);
    auto v = new Vector!real(1000, 10.0);

    v *= d;

    assert(v.norm!"L1" <= 0.001);

    write("Done.\n");
}

/++ Reflection matrix class.
 +
 +
 + We define a reflection matrix to be of the form:
 +  I - 2vv*/||v||^2
 + where I is the identity matrix
 + v is a complex vector
 + v* denotes the conjugate transpose of v
 + ||v||^2 is the euclidean norm of v
 +/
class ReflectionMatrix(T) : MatrixAbstract!T {
    Vector!T vec;
    
    /// Constructor
    pure @safe
    this(in size_t size)
    {
        typeId = "ReflectionMatrix";
        rows = size; cols = size;
        vec = new Vector!T(size);
    }

    /// Simple constructor with random initialization
    @safe
    this(in size_t size, in Tc randomBound)
    {
        this(size);

        if (randomBound <= 0)
            throw new Exception("'randomBound' must be > 0");

        static if (is(Complex!T : T)) {
            foreach(size_t i;0 .. size)
                vec[i] = complex(uniform(-randomBound, randomBound, rnd),
                                 uniform(-randomBound, randomBound, rnd));
        }
        else {
            foreach(size_t i;0 .. size)
               vec[i] = uniform(-randomBound, randomBound, rnd);
        }
    }


    /// Copy-constructor
    @safe
    this(in ReflectionMatrix dupl)
    {
        typeId = "ReflectionMatrix";
        rows = dupl.vec.length;
        cols = dupl.vec.length;
        vec = dupl.vec.dup;
    }

    /// Constructor from list.
    pure @safe
    this(in T[] valarr)
    {
        rows = valarr.length;
        cols = valarr.length;
        vec = new Vector!T(valarr);
    }
    /// Constructor from Vector.
    pure @safe
    this(in Vector!T valarr)
    {
        this(valarr.v);
    }


    override
    const @property
    ReflectionMatrix!T dup()
    {
        return new ReflectionMatrix(this);
    }

    @property const pure @safe
    size_t length()
    {
        return rows;
    }

    @property const pure @safe
    Tc invSqNormVec2()
    {
        return -2*pow(vec.norm!"L2",-2);
    }

    /+ Vector multiplication.
     + As we only store the vector that define the reflection
     + We can comme up with a linear-time matrix-vector multiplication.
     +/
    const pure @safe
    Vector!T opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        return new Vector!T(this * v.v);
    }

    const pure @safe
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");
        Tc invSqNormVec2 = -2*pow(vec.norm!"L2",-2);
        T[] vres = v.dup;
        T s = vec.conjdot(v, vec);
        T[] tmp = vec.v.dup;
        s *= invSqNormVec2;
        foreach(i; 0 .. cols)
            tmp[i] = tmp[i] * s;
        vres[] += tmp[];
        return vres;
    }

    const pure @safe
    Vector!T opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        return v.v / this;
    }

    const pure @safe
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        enforce(v.length == cols, "Matrix-Vector division: dimensions mismatch.");
        Tc invSqNormVec2 = -2*pow(vec.norm!"L2",-2);
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

    const pure @safe
    Matrix!T toMatrix()
    {
        auto res = new Matrix!T(rows, cols);
        Tc invSqNormVec2 = -2*pow(vec.norm!"L2",-2);
        T s;
        foreach(size_t i; 0 .. rows) {
            s = vec[i]*invSqNormVec2;
            foreach(size_t j; 0 .. cols){
                static if (is(Complex!T : T))
                    res[i,j] = s*vec[j].conj;
                else
                    res[i,j] = s*vec[j];
            }
            static if (is(Complex!T : T))
                res[i,i] = res[i,i] + cast(T) complex(1);
            else
                res[i,i] = res[i,i] + cast(T) 1;
        }
        return res;
    }
}
unittest
{
    write("                  Reflection ... ");
    {
        // Verification of the multiplication algorithm.
        alias Reflection = ReflectionMatrix!(Complex!real);
        ReflectionMatrix!(Complex!real) r1 = new Reflection(100, 1.0);
        ReflectionMatrix!(Complex!real) r1bis = r1.dup;
        Matrix!(Complex!real) m1 = r1.toMatrix();

        r1bis.vec -= r1.vec;
        assert(r1bis.vec.norm!"L2" <= 0.0001);

        auto v = new Vector!(Complex!real)(100, 1.0);
        auto u = new Vector!(Complex!real)(v);

        v.opOpAssign!"*"(r1);
        u.opOpAssign!"*"(m1);
        v -= u;
        assert(v.norm!"L2" < 0.0001);
    }
    {
        auto vref = new Vector!real(100, 1.0);

        auto r1 = new ReflectionMatrix!real(vref);
        auto r1bis = new ReflectionMatrix!real(vref.v);
        Matrix!real m1 = r1.toMatrix();

        r1bis.vec -= r1.vec;
        assert(r1bis.vec.norm!"L2" <= 0.0001);

        auto v = new Vector!real(100, 1.0);
        auto u = new Vector!real(v);

        v.opOpAssign!"*"(r1);
        u.opOpAssign!"*"(m1);
        v -= u;
        assert(v.norm!"L2" < 0.0001);

        assert(r1.length == 100);
        
    }

    assertThrown(new ReflectionMatrix!(Complex!real)(10, 0.0));

    writeln("Done.");
}

/++ Permutation matrix class.
 +/
class PermutationMatrix(T) : MatrixAbstract!T {
    size_t[] perm;

    /// Constructor
    pure @safe
    this(in size_t size)
    {
        typeId = "PermutationMatrix";
        rows = size; cols = size;
        perm = new size_t[size];
    }

    /// Simple constructor with random initialization
    /// Here the randomBound is not used and a simple
    /// random permutation is returned.
    @safe
    this(in size_t size, in float randomBound)
    {
        this(size.iota.array);
        randomShuffle(perm);
    }


    /// Copy-constructor
    pure @safe
    this(in PermutationMatrix dupl)
    {
        this(dupl.cols);
        perm = dupl.perm.dup;
    }

    /// Constructor from list (trusted to be a permutation)
    pure @safe
    this(in size_t[] valarr)
    {
        this(valarr.length);
        perm = valarr.dup;
    }


    override
    const @property
    PermutationMatrix!T dup()
    {
        return new PermutationMatrix(this);
    }

    @property @nogc
    const pure @safe
    size_t length()
    {
        return rows;
    }

    const
    pure @safe  @nogc
    auto permute(size_t i)
    {
        return perm[i];
    }


    ///  Vector multiplication.
    const pure @safe
    Vector!T opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        auto vres = new Vector!T(v.length);
        foreach(i; 0 .. v.length)
            vres[i] = v[perm[i]];
        return vres;
    }

    const pure @safe
    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");
        auto vres = new T[v.length];
        foreach(i; 0 .. v.length)
            vres[i] = v[perm[i]];
        return vres;
    }

    const pure @safe
    Vector!T opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        return new Vector!T(v.v / this);
    }

    const pure @safe
    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        enforce(v.length == cols, "Matrix-Vector division: dimensions mismatch.");
        auto vres = new T[v.length];
        foreach(i; 0 .. v.length)
            vres[perm[i]] = v[i];
        return vres;
    }
}
unittest
{
    write("                  Permutation ... ");
    
    alias Perm = PermutationMatrix!float;
    
    auto p = new Perm(1_000, 3);
    auto o = p.dup;

    assert(p.perm[4] < p.perm.length);
    p.perm[] -= o.perm[];
    assert(p.perm.sum == 0);

    assert(p.length == 1_000);
    
    write("Done.\n");
}

/++ General matrix class.
 +
 +  A class that implement a general matrix (with rows*cols parameter).
 +
 +  Note: Contrary to most of the others matrix, this class doesn't implement
 +  	  the division operator (multiplication by the inverse).
 +
 +/
class Matrix(T) : MatrixAbstract!T {
    T[] params;

    /// Simple constructor
    pure @safe
    this(in size_t rows, in size_t cols)
    {
        super();
        typeId = "Matrix";
        params = new T[rows*cols];
        this.rows = rows;
        this.cols = cols;
    }
    
    pure @safe
    this(in size_t rows)
    {
        this(rows, rows);
    }

    @safe
    this(in size_t rows, in Tc randomBound)
    {
        this(rows, rows, randomBound);
    }

    /// Simple constructor with random initialization
    @safe
    this(in size_t rows, in size_t cols, in Tc randomBound)
    {
        super(true);
        typeId = "Matrix";
        params = new T[rows*cols];
        this.rows = rows;
        this.cols = cols;

        if (randomBound < 0)
            throw new Exception("'randomBound' must be >= 0");
        if (randomBound.abs == 0) {
            foreach(i; 0 .. params.length)
                params[i] = to!(T)(0);
        }
        else {
            static if (is(Complex!T : T)) {
                foreach(i;0 .. params.length)
                    params[i] = complex(uniform(-randomBound, randomBound, rnd),
                                     uniform(-randomBound, randomBound, rnd));
            }
            else {
                foreach(i;0 .. rows*cols)
                    params[i] = uniform(-randomBound, randomBound, rnd);
            }
        }
    }

    /// Copy-constructor
    pure @safe
    this(in Matrix dupl)
    {
        this(dupl.rows, dupl.cols);
        foreach(i;0 .. rows*cols) {
            params[i] = dupl.params[i];
        }
    }

    override
    const @property
    Matrix!T dup()
    {
        return new Matrix(this);
    }

    @property const pure @safe
    size_t length()
    {
        return rows;
    }

    pure @safe
    void opIndexAssign(T value, in size_t i, in size_t j)
    {params[i*cols + j] = value;}

    /// Return value by index.
    pure const @safe
    T opIndex(in size_t i, in size_t j)
    {return params[i*cols + j];}
   
    /// Simple math operation without memory allocation.
    pure @safe
    void opOpAssign(string op)(in Matrix other)
    {
             static if (op == "+") { params[] += other.params[]; }
        else static if (op == "-") { params[] -= other.params[]; }
        else static if (op == "*") { this  = this * other; }
        else static assert(0, "Operator "~op~" not implemented.");
    }

    const pure @safe
    auto opBinary(string op)(in Matrix other)
    if (op == "+" || op == "-")
    {
        auto res = new Matrix(this);
        foreach(i;0 .. params.length)
            mixin("res.params[i] " ~ op ~ "= other.params[i];");
        return res;
    }

    const pure @safe
    auto opBinary(string op)(in Matrix other)
    if (op == "*")
    {
        enforce(cols == other.rows,"Matrix-Matrix multiplication: dimensions mismatch.");
        auto res = new Matrix(rows, other.cols);
        foreach(i;0 .. rows) {
            foreach(j; 0 .. other.cols) {
                T s;
                foreach(k; 0 .. cols){
                    s += params[i*cols + k] * other.params[k*other.cols + j];
                }
                res[i, j] = s;
            }
        }
        return res;
    }

    const pure @safe
    auto opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        return new Vector!T(this * v.v);
    }

    const pure @safe
    auto opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector division: dimensions mismatch.");
        auto res = new T[rows];
        foreach(size_t i; 0 .. rows){
            T s = params[i*cols]*v[0];
            foreach(size_t j; 1 .. cols)
                s += params[i*cols + j]*v[j];
            res[i] = s;
        }
        return res;
    }
}
unittest
{
    write("                  Matrix ... ");
    auto m1 = new Matrix!(Complex!float)(10, 30, 5.0f);
    auto m2 = m1.dup;
    auto m3 = new Matrix!(Complex!float)(30, 5, 10.0f);
    auto m4 = new Matrix!real(100, 1.0);
    auto m5 = new Matrix!real(100);
    m5.params = m4.params.dup;

    m2 -= m1;
    assert(m2.params.sum.abs < 0.1);

    m5 -= m4;
    assert(m5.params.sum.abs < 0.1);

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

    assertThrown(new Matrix!float(5, -0.5));

    auto m = new Matrix!float(7, 2, 0.0);

    auto v = new Vector!float(2, 1.0);

    v *= m;

    assert(v.length == 7);
    assert(v.norm!"L2" <= 0.0001);

    writeln("Done.");
}
