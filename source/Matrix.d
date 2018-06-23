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
    write("Unittest: Matrix: Dot Product ...");
    assert(dot([1.0, 2.0, 3.0], [-6.0, -5.0, -1.0]) ==
           dot([-6.0, -5.0, -1.0], [1.0, 2.0, 3.0]));

    assert(std.complex.abs(dot([complex(1.0, 8.5), complex(6.4, 3.58), complex(10.8, 7.65)],
               [-6.0, -5.0, -1.0])
           -
               dot([complex(6.4, 3.58), complex(10.8, 7.65), complex(1.0, 8.5)],
               [-5.0, -1.0, -6.0])) < 0.001);
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
class BlockMatrix(Mtype : M!T, alias M, T) : Parameter {
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    // List of matrices.
    Mtype[] blocks;

    // Permutation, can be "turned off" with randperm = false.
    PermutationMatrix!T P, Q;

    size_t rows, cols;

    size_t size_blocks;
    size_t num_blocks;
    size_t size_out;
    size_t size_in;

    // The two permutations are needed here so that 
    // every nodes can be connected to any others.
    // It is a way to "shuffle" the block matrix
    // and so to have a sparse matrix with the properties
    // of the block matrix (e.g. unitary).

    this(){}

    pure @safe
    this(in size_t size_in, in size_t size_out, in size_t size_blocks)
    {
        enforce(size_out%size_blocks == 0,
                "'size_out' must be a multiple of 'size_blocks'.");
        enforce(size_in%size_blocks == 0,
                "'size_in' must be a multiple of 'size_blocks'.");
        this.rows = size_out;
        this.cols = size_in;

        this.size_blocks = size_blocks;
        size_t maxsize = size_out; if (maxsize<size_in) maxsize=size_in;
        this.num_blocks = maxsize/size_blocks;
        this.size_out = size_out;
        this.size_in = size_in;
    }

    @safe
    this(in size_t size_in, in size_t size_out, in size_t size_blocks,
         Mtype[] blocks, bool randperm=false)
    {
        enforce(!randperm || (size_in == size_out), "randperm => (size_in == size_out)." );
        this(size_in, size_out, size_blocks);
        this.blocks = blocks;

        if (randperm) {
            P = new PermutationMatrix!T(size_in, 1.0);
            Q = new PermutationMatrix!T(size_out, 1.0);
        }
    }

    @safe
    this(in size_t size_in, in size_t size_blocks,
         Mtype[] blocks, bool randperm=false)
    {
        this(size_in, size_in, size_blocks, blocks, randperm);
    }

    const @property
    auto dup()
    {
        auto res = new BlockMatrix!(Mtype)(size_in, size_out, size_blocks);
        res.P = P.dup;
        res.Q = Q.dup;
        res.blocks = new Mtype[res.num_blocks];

        foreach(i; 0 .. res.num_blocks)
            res.blocks[i] = blocks[i].dup;

        return res;
    }

    auto opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        auto res = this * v.v;
        return new Vector!T(res);
    }

    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");
        T[] res;
        if (P) res = this.P * v;
        else res = new T[size_out];

        size_t blocks_in = size_in / size_blocks;
        size_t blocks_out = size_out / size_blocks;

        T[] s;
        size_t index;
        size_t b_tmp = 0; // used when size_in < size_out to cycle through the small vector.

        // We construct two different forloop here.
        // The goal of this dichotomy is that it allow us to save computation in both case.
        if (size_out > size_in)
            foreach(size_t b; 0 .. blocks_out) {
                // We take the first block matrix and multiply it with
                // the corresponding part of the vector.
                s = blocks[b] * v[(b_tmp*size_blocks) .. ((b_tmp+1)*size_blocks)];
                res[(b*size_blocks) .. ((b+1)*size_blocks)] = s;

                // This line is needed only in this case.
                if (++b_tmp == blocks_in) b_tmp = 0;
            }
        else
            foreach(size_t b; 0 .. blocks_out) {
                // We take the first block matrix and multiply it with
                // the corresponding part of the vector.
                s = blocks[b] * v[(b*size_blocks) .. ((b+1)*size_blocks)];

                // We then increment the index in case the block matrix
                // is rectangular with more columns than rows.
                // The while loop is only needed in this case.
                index = b + blocks_out;
                while(index < blocks_in) {
                    s[] += (blocks[index] *
                           v[(index*size_blocks) .. ((index+1)*size_blocks)])[];
                    index += blocks_out;
                }

                res[(b*size_blocks) .. ((b+1)*size_blocks)] = s;
            }


        if (Q) res = this.Q * res;
        return res;
    }

    auto opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        return new Vector!T(v.v / this);
    }

    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        enforce(v.length == cols, "Matrix-Vector division: dimensions mismatch.");
        enforce(size_out == size_in, "Inverse of rectangular BlockMatrix not implemented");
        T[] res;
        if (P) res = v / this.Q;
        else res = new T[size_in];
        
        size_t blocks_in = size_in / size_blocks;
        size_t blocks_out = size_out / size_blocks;

        foreach(size_t b; 0 .. blocks_out) {
            // We take the first block matrix and multiply it with
            // the corresponding part of the vector.
            res[(b*size_blocks) .. ((b+1)*size_blocks)] =
                v[(b*size_blocks) .. ((b+1)*size_blocks)] / blocks[b];
        }

        if (P) res = res / this.P;
        return res;
    }
}
unittest
{
    write("                  Block ... ");

    // create an empt BlockMatrix for the seek of coverage.
    auto mtmp_blue_unused = new BlockMatrix!(UnitaryMatrix!real)();

    auto len = 1024;
    
    // Matrix | in: len | out: len
    auto vec1 = new Vector!float(len, 1.0);
    auto m1_1 = new Matrix!float(len/4, 1.0);
    auto m1_2 = new Matrix!float(len/4, 1.0);
    auto m1_3 = new Matrix!float(len/4, 1.0);
    auto m1_4 = new Matrix!float(len/4, 1.0);
    auto b1 = new BlockMatrix!(Matrix!float)
                              (len, len/4, [m1_1, m1_2, m1_3, m1_4]);
    auto res1_1 = b1*vec1;
    auto res1_2 = vec1.dup();
    res1_2.v[0 .. len/4] = m1_1 * vec1.v[0 .. len/4];
    res1_2.v[len/4 .. len/2] = m1_2 * vec1.v[len/4 .. len/2];
    res1_2.v[len/2 .. (len/2 + len/4)] = m1_3 * vec1.v[len/2 .. (len/4 + len/2)];
    res1_2.v[(len/2 + len/4) .. $] = m1_4 * vec1.v[(len/4 + len/2) .. $];
    res1_1 -= res1_2;
    assert(res1_1.norm!"L2" <= 0.00001);
    
    // PermutationMatrix | in: len | out: len/2
    auto vec2 = new Vector!(Complex!float)(len, 1.0);
    auto m2_1 = new PermutationMatrix!(Complex!float)(len/4, 1.0);
    auto m2_2 = new PermutationMatrix!(Complex!float)(len/4, 1.0);
    auto m2_3 = new PermutationMatrix!(Complex!float)(len/4, 1.0);
    auto m2_4 = new PermutationMatrix!(Complex!float)(len/4, 1.0);
    auto b2 = new BlockMatrix!(PermutationMatrix!(Complex!float))
                              (len, len/2, len/4, [m2_1, m2_2, m2_3, m2_4]);
    auto res2_1 = b2*vec2;
    auto res2_2 = new Vector!(Complex!float)(len/2);
    res2_2.v[0 .. len/4] = m2_1 * vec2.v[0 .. len/4];
    res2_2.v[len/4 .. len/2] = m2_2 * vec2.v[len/4 .. len/2];
    res2_2.v[0 .. len/4][] += (m2_3 * vec2.v[len/2 .. (len/4 + len/2)])[];
    res2_2.v[len/4 .. len/2][] += (m2_4 * vec2.v[(len/4 + len/2) .. $])[];
    res2_1 -= res2_2;
    assert(res2_1.norm!"L2" <= 0.00001);
    
    // DiagonalMatrix | in: len/2 | out: len
    auto vec3 = new Vector!(Complex!real)(len/2, 1.0);
    auto m3_1 = new DiagonalMatrix!(Complex!real)(len/4, 1.0);
    auto m3_2 = new DiagonalMatrix!(Complex!real)(len/4, 1.0);
    auto m3_3 = new DiagonalMatrix!(Complex!real)(len/4, 1.0);
    auto m3_4 = new DiagonalMatrix!(Complex!real)(len/4, 1.0);
    auto b3 = new BlockMatrix!(DiagonalMatrix!(Complex!real))
                              (len/2, len, len/4, [m3_1, m3_2, m3_3, m3_4]);
    auto res3_1 = b3*vec3;
    auto res3_2 = new Vector!(Complex!real)(len);
    res3_2.v[0 .. len/4] = m3_1 * vec3.v[0 .. len/4];
    res3_2.v[len/4 .. len/2] = m3_2 * vec3.v[len/4 .. len/2];
    res3_2.v[len/2 .. (len/2 + len/4)] = m3_3 * vec3.v[0 .. len/4];
    res3_2.v[(len/2 + len/4) .. $] = m3_4 * vec3.v[len/4 .. len/2];
    res3_1 -= res3_2;
    assert(res3_1.norm!"L2" <= 0.00001);
    
    // ReflectionMatrix | in: len | out: len
    auto vec4 = new Vector!double(len, 1.0);
    auto m4_1 = new ReflectionMatrix!double(len/4, 1.0);
    auto m4_2 = new ReflectionMatrix!double(len/4, 1.0);
    auto m4_3 = new ReflectionMatrix!double(len/4, 1.0);
    auto m4_4 = new ReflectionMatrix!double(len/4, 1.0);
    auto b4 = new BlockMatrix!(ReflectionMatrix!double)
                              (len, len/4, [m4_1, m4_2, m4_3, m4_4]);
    auto res4_1 = b4*vec4;
    auto res4_2 = vec4.dup();
    res4_2.v[0 .. len/4] = m4_1 * vec4.v[0 .. len/4];
    res4_2.v[len/4 .. len/2] = m4_2 * vec4.v[len/4 .. len/2];
    res4_2.v[len/2 .. (len/2 + len/4)] = m4_3 * vec4.v[len/2 .. (len/4 + len/2)];
    res4_2.v[(len/2 + len/4) .. $] = m4_4 * vec4.v[(len/4 + len/2) .. $];
    res4_1 -= res4_2;
    assert(res4_1.norm!"L2" <= 0.00001);
    
    // FourierMatrix
    auto vec5 = new Vector!(Complex!double)(len, 1.0);
    auto m5_1 = new FourierMatrix!(Complex!double)(len/4);
    auto m5_2 = new FourierMatrix!(Complex!double)(len/4);
    auto m5_3 = new FourierMatrix!(Complex!double)(len/4);
    auto m5_4 = new FourierMatrix!(Complex!double)(len/4);
    auto b5 = new BlockMatrix!(FourierMatrix!(Complex!double))
                              (len, len/4, [m5_1, m5_2, m5_3, m5_4]);
    auto res5_1 = b5*vec5;
    auto res5_2 = vec5.dup();
    res5_2.v[0 .. len/4] = m5_1 * vec5.v[0 .. len/4];
    res5_2.v[len/4 .. len/2] = m5_2 * vec5.v[len/4 .. len/2];
    res5_2.v[len/2 .. (len/2 + len/4)] = m5_3 * vec5.v[len/2 .. (len/4 + len/2)];
    res5_2.v[(len/2 + len/4) .. $] = m5_4 * vec5.v[(len/4 + len/2) .. $];
    res5_1 -= res5_2;
    assert(res5_1.norm!"L2" <= 0.00001);


    assertThrown(new BlockMatrix!(DiagonalMatrix!real)(4u, 4u, 3u));
    assertThrown(new BlockMatrix!(DiagonalMatrix!real)(4u, 3u, 3u));

    write("Done.\n");
}


/++ Unitary Matrix class.
 +
 +  TODO: take non-complex number as input.
 +        handle permutation
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
class UnitaryMatrix(T) : Parameter
{
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    size_t rows, cols;

    PermutationMatrix!T perm;
    FourierMatrix!T fourier;
    Tc[] params;
    // used to save temporary values during the calculation to avoid mem alloc.
    
    static if (!is(Complex!T: T)) mixin("Tc[] tmp_vector;");
    else mixin("T[] tmp_vector;");

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

        rows = size;
        cols = size;

        perm = new PermutationMatrix!T(size ,1.0);
        fourier = new FourierMatrix!T(size);
        params = new Tc[7*size];
        
        static if (!is(Complex!T : T)) {
            tmp_vector = new Tc[4*size];
        }
        else {
            tmp_vector = new T[size];
        }
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
        this();
        rows = M.rows;
        cols = M.cols;
        perm = M.perm.dup;
        fourier = M.fourier.dup;
        fourier = M.fourier.dup;
        params = M.params.dup;
        tmp_vector = M.tmp_vector.dup;
    }

    const @property 
    UnitaryMatrix!T dup()
    {
        return new UnitaryMatrix(this);
    }

    /// Apply the "num"th diagonal matrix on the given vector.
    pure @safe
    void applyDiagonal(T)(ref T[] v, in size_t num)
    {
        static if (!is(Complex!T : T)) {
            // we use expi to convert each value to a complex number
            // the value is the angle of the complex number with radius 1.
            size_t start_index = num*rows;
            size_t end_index = num*rows + rows;

            // save cos & sin values
            fill(tmp_vector[0 .. $>>2], params[start_index .. end_index].map!(a => cos(a)));
            fill(tmp_vector[$>>2 .. $>>1], params[start_index .. end_index].map!(a => sin(a)));

            // do some math magic
            tmp_vector[$>>1 .. $ - ($>>2)] = v[0 .. $>>1] * tmp_vector[0 .. $>>2]
                                         - v[$>>1 .. $] * tmp_vector[$>>2 .. $>>1];

            tmp_vector[$ - ($>>2) .. $] = v[$>>1 .. $] * tmp_vector[0 .. $>>2]
                                      + v[0 .. $>>1] * tmp_vector[$>>2 .. $>>1];

            v[] = tmp_vector[$>>1 .. $];
        }
        else {
            size_t start_index = num*rows;
            foreach(i; 0 .. rows)
                v[i] *= cast(T) std.complex.expi(params[start_index + i]);
        }
    }

    pure @safe
    void applyDiagonalInv(ref T[] v, in size_t num)
    {
        static if (!is(Complex!T : T)) {
            // we use expi to convert each value to a complex number
            // the value is the angle of the complex number with radius 1.
            // SAME as applyDiagonal except we take -sin.
            size_t start_index = num*rows;
            size_t end_index = num*rows + rows;

            // save cos & sin values
            fill(tmp_vector[0 .. $>>2], params[start_index .. end_index].map!(a => cos(a)));
            fill(tmp_vector[$>>2 .. $>>1], params[start_index .. end_index].map!(a => -sin(a)));

            // do some math magic
            tmp_vector[$>>1 .. $ - ($>>2)] = v[0 .. $>>1] * tmp_vector[0 .. $>>2]
                                         - v[$>>1 .. $] * tmp_vector[$>>2 .. $>>1];

            tmp_vector[$ - ($>>2) .. $] = v[$>>1 .. $] * tmp_vector[0 .. $>>2]
                                      + v[0 .. $>>1] * tmp_vector[$>>2 .. $>>1];

            v[] = tmp_vector[$>>1 .. $];
        }
        else {
            // we use expi to convert each value to a complex number
            // the value is the angle of the complex number with radius 1.
            size_t start_index = num*rows;
            foreach(i; 0 .. rows)
                v[i] /= cast(T) std.complex.expi(params[start_index + i]);
        }
    }

    /// Apply the "num"th reflection matrix on the given vector.
    // TODO pure @safe
    auto applyReflection(ref T[] v, in size_t num)
    {
        size_t start_index = (2*num + 3)*rows;
        static if (!is(Complex!T : T)) {
            auto h = params[start_index .. start_index + 2*rows];

            auto t_a = dot(h[0 .. rows], v[0 .. rows])
                     + dot(h[rows .. rows << 1], v[rows .. rows << 1]);
            auto t_b = dot(h[0 .. rows], v[rows .. rows << 1])
                     - dot(h[rows .. rows << 1], v[0 .. rows]);

            tmp_vector[0 .. rows][]  = h[0 .. rows][];
            tmp_vector[0 .. rows][] *= t_a;
            tmp_vector[(rows<<1) .. ((rows<<1)+rows)][] = h[rows .. rows << 1][];
            tmp_vector[(rows<<1) .. ((rows<<1)+rows)][] *= t_b;
            tmp_vector[0 .. rows][] -= tmp_vector[(rows<<1) .. ((rows<<1)+rows)][];

            tmp_vector[rows .. (rows<<1)][] = h[0 .. rows][];
            tmp_vector[rows .. (rows<<1)][] *= t_b;
            tmp_vector[(rows<<1) .. ((rows<<1)+rows)][] = h[rows .. rows << 1][];
            tmp_vector[(rows<<1) .. ((rows<<1)+rows)][] *= t_a;
            tmp_vector[rows .. (rows<<1)][] += tmp_vector[(rows<<1) .. ((rows<<1)+rows)][];

            tmp_vector[0 .. (rows<<1)][] *= 2.0 / h.map!(a => a*a).sum;

            v[] -= tmp_vector[0 .. rows << 1];


        }
        else {
            // The '+3' is because the diagonal matrices are first
            // in the params array.
            // The '*2' is because each reflection matrix need
            // 2 * rows parameters to be defined.
            size_t start_indexPlRows = start_index + rows;
            auto a = params[start_index .. start_indexPlRows];
            auto b = params[start_indexPlRows .. start_indexPlRows + rows];
            T tmp_c = dot(v, a) - complex(0, 1) * dot(v, b);
            tmp_c *= 2 / (dot(a, a) + dot(b, b));

            foreach(i; 0 .. rows)
                tmp_vector[i] = complex(a[i], b[i]) *  tmp_c;

            v[] -= tmp_vector[];
        }
    }

    /// Apply permutation multiplication on array based on the type of the matrix.
    @safe pure
    T[] permMult(T[] tmp_v)
    {
        // Complex valued vector
        static if (is(Complex!T : T)) {
            return this.perm * tmp_v;
        }
        // Real valued vector
        else {
            tmp_vector[0 .. ($ >> 1)][] = tmp_v[];
            size_t tmp_len = tmp_v.length >> 1;
            foreach(i; 0 .. tmp_len){
                tmp_v[i] = tmp_vector[this.perm.perm[i]];
                tmp_v[i + tmp_len] = tmp_vector[this.perm.perm[i] + tmp_len];
            }
            return tmp_v;
        }
    }

    /// Apply permutation inverse on array based on the type of the matrix.
    @safe pure
    T[] permInv(T[] tmp_v)
    {
        // Complex valued vector
        static if (is(Complex!T : T)) {
            return tmp_v / this.perm;
        }
        // Real valued vector
        else {
            tmp_vector[0 .. ($ >> 1)][] = tmp_v[];
            size_t tmp_len = tmp_v.length >> 1;
            foreach(i; 0 .. tmp_len){
                tmp_v[this.perm.perm[i]] = tmp_vector[i];
                tmp_v[this.perm.perm[i] + tmp_len] = tmp_vector[i + tmp_len];
            }
            return tmp_v;
        }
    }

    Vector!T opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        return new Vector!T(this * v.v);
    }

    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        enforce(v.length == cols, "Matrix-Vector multiplication: dimensions mismatch.");

        static if (!is(Complex!T : T)) {
            T[] res = new T[v.length << 1];
            res[0 .. v.length] = v[];
            res[v.length .. $] = 0;
        }
        else {
            T[] res = v.dup;
        }

        this.applyDiagonal(res, 0);

        res = fourier * res;

        this.applyReflection(res, 0);
        res = this.permMult(res);
        this.applyDiagonal(res, 1);

        res = res / fourier;

        this.applyReflection(res, 1);
        this.applyDiagonal(res, 2);

        static if (!is(Complex!T: T))
            return res[0 .. ($ >> 1)][];
        else
            return res;
    }

    Vector!T opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        return new Vector!T(v.v / this);
    }

    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        enforce(v.length == cols, "Matrix-Vector division: dimensions mismatch.");
        
        static if (!is(Complex!T : T)) {
            T[] res = new T[v.length << 1];
            res[0 .. v.length] = v[];
            res[v.length .. $] = 0;
        }
        else {
            T[] res = v.dup;
        }

        this.applyDiagonalInv(res, 2);
        this.applyReflection(res, 1);
        
        res = fourier * res;

        this.applyDiagonalInv(res, 1);
        res = this.permInv(res);
        this.applyReflection(res, 0);
        
        res = res / fourier;
        
        this.applyDiagonalInv(res, 0);

        static if (!is(Complex!T: T))
            return res[0 .. ($ >> 1)][];
        else
            return res;
    }
}
unittest
{
    write("                  Unitary ... ");
    // Complex valued Matrix
    {
        auto m0 = new UnitaryMatrix!(Complex!float)();
        auto m = new UnitaryMatrix!(Complex!double)(1024, 9.0);
        auto n= m.dup;
        auto v = new Vector!(Complex!double)(1024, 1.2);
        auto memcop = v.dup;

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

    // There is a bug in the applyReflection method
    // Real valued Matrix
    {
        size_t len = 2;
        auto mc = new UnitaryMatrix!(Complex!real)(len, 1.0);
        auto mr = new UnitaryMatrix!real(len, 1.0);
    
        // Copy mc parameters into mr.
        mr.params = mc.params.dup;
        mr.perm.perm = mc.perm.perm.dup;

        auto vr = new Vector!real(len, 1.0);
        auto vc = new Vector!(Complex!real)(len);
        foreach(i; 0 .. len)
            vc[i] = complex(vr[i], 0);
        
        { // Multiplication
            auto res_c = mc * vc;
            auto res_r = mr * vr;

            // Take only real values.
            auto res_c_r = new Vector!real(len);
            foreach(i; 0 .. len)
                res_c_r[i] = res_c[i].re;

            res_c_r -= res_r;
            assert(res_c_r.norm!"L2" <= 0.00001);
        }

        { // Inverse
            auto res_c = vc / mc;
            auto res_r = vr / mr;

            // Take only real values.
            auto res_c_r = new Vector!real(len);
            foreach(i; 0 .. len)
                res_c_r[i] = res_c[i].re;

            res_c_r -= res_r;
            assert(res_c_r.norm!"L2" <= 0.00001);
        }
    }

    write("Done.\n");
}

/++ Fourier Matrix class.
 +
 +  Implement the fourier matrix.
 +
 +  TODO: Allow non complex vector as input => assume it is of the form [real values, imaginary values]
 +/
class FourierMatrix(T) : Parameter
{
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    size_t rows, cols;

    Fft objFFT;
    Complex!T[] tmp_vector;
    Complex!T[] tmp_result;
    T[] return_vector_real;
    
    this(size_t size)
    {
        rows = size;
        cols = size;
        objFFT = new Fft(size);

        tmp_vector = new Complex!T[size];
        tmp_result = new Complex!T[size];
        return_vector_real = new T[size << 1];
    };
    
    this(in FourierMatrix M)
    {
        this(M.rows);
    }

    const @property
    FourierMatrix!T dup()
    {
        return new FourierMatrix(this);
    }

    Vector!T opBinary(string op)(in Vector!T v)
    if (op=="*")
    {
        return new Vector!T(this * v.v);
    }

    T[] opBinary(string op)(in T[] v)
    if (op=="*")
    {
        static if (!is(Complex!T : T)) {
            enforce((v.length >> 1) == cols, "Matrix-Vector divison: dimensions mismatch.");
            foreach(i; 0 .. v.length >> 1)
            {
                tmp_vector[i] = complex(v[i], v[i+($>>1)]);
            }
            objFFT.fft(tmp_vector, tmp_result);
            foreach(i, c; tmp_result)
            {
                return_vector_real[i] = c.re;
                return_vector_real[i + ($ >> 1)] = c.im;
            }
            return return_vector_real;
        }
        else {
            enforce(v.length == cols, "Matrix-Vector divison: dimensions mismatch.");
            objFFT.fft(v, tmp_vector);
            tmp_result[] = tmp_vector[];
            return tmp_result;
        }
    }

    Vector!T opBinaryRight(string op)(in Vector!T v)
    if (op=="/")
    {
        return new Vector!T(v.v / this);
    }

    T[] opBinaryRight(string op)(in T[] v)
    if (op=="/")
    {
        static if (!is(Complex!T : T)) {
            enforce((v.length >> 1) == cols, "Matrix-Vector divison: dimensions mismatch.");
            foreach(i; 0 .. v.length >> 1)
            {
                tmp_vector[i] = complex(v[i], v[i+($>>1)]);
            }
            objFFT.inverseFft(tmp_vector, tmp_result);
            foreach(i, c; tmp_result)
            {
                return_vector_real[i] = c.re;
                return_vector_real[i + ($ >> 1)] = c.im;
            }
            return return_vector_real;
        }
        else {
            enforce(v.length == cols, "Matrix-Vector divison: dimensions mismatch.");
            objFFT.inverseFft(v, tmp_vector);
            tmp_result[] = tmp_vector[];
            return tmp_result;
        }
    }
}
unittest
{
    write("                  Fourrier ... ");

    size_t len = 4;
    // Complex
    {
        alias Fourier = FourierMatrix!(Complex!double);
        auto f = new Fourier(len);
        auto g =  f.dup;
        auto vc = new Complex!double[len];

        auto rnd_tmp = Random(cast(uint) ((Clock.currTime()
                         - SysTime(unixTimeToStdTime(0))).total!"msecs"));

        foreach(i;0 .. len)
            vc[i] = complex(uniform(-1.0, 1.0, rnd_tmp),
                            uniform(-1.0, 1.0, rnd_tmp));

        auto r1 = f * (vc / f);
        auto r2 = (g * vc) / g;
        auto r3 = (f * vc) / f;

        foreach(i;0 .. len){
            assert(std.complex.abs(r1[i] - vc[i]) <= 0.0001);
            assert(std.complex.abs(r2[i] - vc[i]) <= 0.0001);
            assert(std.complex.abs(r1[i] - r1[i]) <= 0.0001);
        }
    }

    // Real
    {
        alias Fourier = FourierMatrix!real;
        auto f = new Fourier(len);
        auto g = new FourierMatrix!(Complex!real)(len);

        auto v = new Vector!real(len << 1, 1.0);

        auto c = new Vector!(Complex!real)(len, 0);

        foreach(i; 0 .. len)
            c[i] = complex(v[i], v[i + len]);
        
        auto r1 = f*v;
        auto r2 = g*c;

        auto r = v.dup;
        foreach(i; 0 .. len)
        {
            r[i] = r2[i].re;
            r[i + len] = r2[i].im;
        }

        r -= r1;

        assert(r.norm!"L2" <= 0.0001);
    }

    write("Done.\n");
}

/++ Diagonal matrix class.
 +
 + TODO: There mmight be too much things, we could need to remove some methods.
 +
 +/
class DiagonalMatrix(T) : Parameter {
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    T[] params;

    size_t rows, cols;
    /// Constructor
    pure @safe
    this(in size_t size)
    {
        this.rows = size; this.cols = size;
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
        assert(std.complex.abs(m3.sum) < 0.0001);

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
class ReflectionMatrix(T) : Parameter {
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    Vector!T vec;
    
    size_t rows, cols;

    /// Constructor
    pure @safe
    this(in size_t size)
    {
        this.rows = size; this.cols = size;
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
        this.rows = dupl.vec.length;
        this.cols = dupl.vec.length;
        vec = dupl.vec.dup;
    }

    /// Constructor from list.
    pure @safe
    this(in T[] valarr)
    {
        this.rows = valarr.length;
        this.cols = valarr.length;
        vec = new Vector!T(valarr);
    }
    /// Constructor from Vector.
    pure @safe
    this(in Vector!T valarr)
    {
        this(valarr.v);
    }


    
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

    @property const pure @safe @nogc
    Tc invSqNormVec2() {
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
        T[] vres = v.dup;

        T s = vec.conjdot(v, vec);
        T[] tmp = vec.v.dup;
        s *= -2*pow(vec.norm!"L2",-2);
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
        // The inverse of a reflection is the very same reflection.
        T[] vres = v.dup;
        T s = vec.conjdot(v, vec);
        T[] tmp = vec.v.dup;
        s *= -2*pow(vec.norm!"L2",-2);
        foreach(i; 0 .. cols)
            tmp[i] = tmp[i] * s;
        vres[] += tmp[];
        return vres;
    }

    const pure @safe
    Matrix!T toMatrix()
    {
        auto res = new Matrix!T(rows, cols);
        T s;
        foreach(size_t i; 0 .. rows) {
            s = vec[i]*(-2*pow(vec.norm!"L2",-2));
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
class PermutationMatrix(T) : Parameter {
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    size_t[] perm;

    size_t rows, cols;
    /// Constructor
    pure @safe
    this(in size_t size)
    {
        this.rows = size; this.cols = size;
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
        return new Vector!T(this * v.v);
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
class Matrix(T) : Parameter {
    static if (is(Complex!T : T))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    T[] params;

    size_t rows, cols;
    /// Simple constructor
    pure @safe
    this(in size_t rows, in size_t cols)
    {
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
    assert(std.complex.abs(m2.params.sum) < 0.1);

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
