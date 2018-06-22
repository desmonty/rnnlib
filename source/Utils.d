module source.Utils;

import std.algorithm;
import std.complex: abs;
import std.exception: assertThrown, enforce;
import std.math: abs;
import std.range;

import source.Layer;
import source.Matrix;
import source.Parameter;

version(unittest) {
    import std.complex;
    import std.stdio: write, writeln;
}

/++ CellRange is used to get all the values inside a cell that
 +  will be reduced together.
 +  It allows us to let the user defines its own reducer in a very
 +  simple way.
 +/ 
class CellRange(T)
{
    private
    {
      size_t pos_x, pos_y,
             cur_pos, 
             cell_w, cell_h,
             shift, width;

      const Vector!T vec;

      bool is_empty = false;
    }

    @nogc @safe pure
    this(in size_t _pos_x, in size_t _pos_y,
         in size_t _cell_w, in size_t _cell_h,
         in size_t _width, ref const Vector!T _v)
    {
        cur_pos = _pos_y * _width + _pos_x;
        pos_x = 0;
        pos_y = 0;
        width = _width;
        cell_h = _cell_h - 1;
        cell_w = _cell_w - 1;
        vec = _v;

        shift = width - cell_w - 1;
    }

    @property @safe @nogc
    const pure
    T front() {
        return vec[cur_pos];
    }

    int opApply(scope int delegate(T) dg)
    {
        int result;

        size_t tmp = cur_pos;
        T tmp_val;
        outer: for (pos_y = 0; pos_y <= cell_h; ++pos_y) {
            for (pos_x = 0; pos_x <= cell_w; ++pos_x) {
                tmp_val = vec[tmp];
                result = dg(tmp_val);
                tmp += 1;

                if (result)
                    break outer;
            }
            tmp += shift;
        }
        return result;
    }

    int opApply(scope int delegate(size_t, T) dg)
    {
        int result;

        size_t tmp = cur_pos;
        T tmp_val;

        outer: for (pos_y = 0; pos_y <= cell_h; ++pos_y) {
            for (pos_x = 0; pos_x <= cell_w; ++pos_x) {
                tmp_val = vec[tmp];
                result = dg(tmp, tmp_val);
                tmp += 1;

                if (result)
                    break outer;
            }
            tmp += shift;
        }
        return result;
    }
}
unittest {
    
    auto _v = new Vector!float([1.5, 2.5, 3.5, 4.5]);

    auto cell_range = new CellRange!float(0, 0, 1, 2, 2, _v);


    float ii;
    size_t jj;
    foreach(i; cell_range) {
        ii = i;
        break;
    }
    assert(ii == 1.5);

    foreach(j, i; cell_range) {
        ii = i;
        jj = j;
        break;
    }
    assert(ii == 1.5 && jj == 0);

}

/++ FrameRange is used to enhanced clean up the following code.
 +  -By adding complexity ..?
 +  -Yes ! It is worth it because it will allow us to do a simple
 +    "foreach" and get both...
 +/
struct FrameRange
{
    private {
        const size_t[] arr;
        const size_t width;
    }

    @nogc @safe pure
    this(ref const size_t[] _arr, const size_t _width) {
        arr = _arr;
        width = _width;
    }

    int opApply(scope int delegate(size_t, size_t) dg)
    {
        int result;

        // if the cut_frame is empty, we do not cut the frame
        // and so we just return the first position with the width.
        if (!arr)
            return dg(width, 0);

        size_t frame_len = arr[0];

        result = dg(frame_len, 0);
        if (result)
            return result;

        immutable size_t len_1 = arr.length -1;
        for (size_t index = 0; index < len_1; ++index) {
            frame_len = arr[index+1] - arr[index];
            result = dg(frame_len, arr[index]);
            if (result)
                break;
        }

        frame_len = width - arr[len_1];
        result = dg(frame_len, arr[len_1]);

        return result;
    }
}


/++ Function that create a general pooling function
 +  See https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
 +  for description.
 +
 +  Template Args:
 +      T: type of the vector's elements.
 +/
@safe
auto createPoolingFunction(T)(in size_t height, in size_t width,
                              in size_t stride_height, in size_t stride_width,
                              in size_t frame_height, in size_t frame_width,
                              in size_t[] cut_height, in size_t[] cut_width,
                              T delegate(CellRange!T) reducer)
{
    /+  This function create a delegate.
        That delegate take as input a vector and apply
        a pooling of specified form to it.
        The first element of the vector is assumed to be the Top Left "pixel"
        and the rest are seen in a "Left-Right/Top-Left" fashion.

        Args:
            height (size_t): height of the picture (let's assume it's a pic). 
            width (size_t): width of the picture (let's assume it's a pic). 
            stride_height (size_t): number of pixel to move to take the next frame
            stride_width (size_t): number of pixel to move to take the next frame
            frame_height (size_t): height of the frame to look at
            frame_width (size_t): width of the frame to look at
            cut_height (size_t[]): List of indices to cut the frame (see below).
            cut_width (size_t[]): List of indices to cut the frame (see below).
            reducer (T delegate(CellRange)): Function that takes an CellRange
                and return a value. This is used to define how you want to
                reduce each cut of the frame (max pooling, average pooling, ...)

        Example:
            height = 10
            width = 10
            stride_height = 2
            stride_width = 2
            frame_height = 2
            frame_width = 5
            cut_height = [1]
            cut_width = [2,3]
            reducer = max

        you will have a delegate that take a vector of size 10*10 = 100
        and which return a vector of size:
            (cut_height.length + 1) * (cut_width.length + 1)
           *(1 + floor( (height - frame_height + 1)) / stride_height) )
           *(1 + floor( (width - frame_width + 1)) / stride_width) )
           = 120.

           The first frame will be in the top left as shown below.

           + - - - - -+- - - - - +
           | a A|b|c C|          |
           +----------|          |
           | d D|e|f F|          |
           +----------+          |
           |                     |
           |                     |
           |                     |
           |                     |
           |                     |
           |                     |
           |                     |
           |                     |
           + - - - - - - - - - - +

           This frame consist of 6 different part (the six letters) which
           will all be reduced to one value by the reducer.
           Hence, this frame will give 6 of the 120 values of the final vector,
           max(a, A), max(b), max(c, C), max(d, D), max(e) and max(f, F).
           The frame will then be moved 'stride_width' pixels to the right
           and the processus will be iterated until the frame cannot move
           to te right. At this point, we move the frame 'stride_height'
           to the bottom and replace it at the left and continue the pooling.
           We continue until the frame cannot move to the right nor the bottom.
           The values computed during this process are added iteratively
           in a vector in a "Left-Right/Top-Left" fashion.
     +/
    if (cut_width) {
        enforce(minElement(cut_width), "cut_width cannot contains '0'");
        enforce(maxElement(cut_width) < width, "max(cut_width) must be < width-1");
        enforce(isSorted(cut_width), "cut_width must be sorted");

        for (size_t tmp = 0; tmp < cut_width.length-1;)
            enforce(cut_width[tmp] < cut_width[++tmp],"cut_width cannot contains doublons");
    }
    if (cut_height) {
        enforce(minElement(cut_height), "cut_height cannot contains '0'");
        enforce(maxElement(cut_height) < height, "max(cut_height) must be < height-1");
        enforce(isSorted(cut_height), "cut_height must be sorted");

        for (size_t tmp = 0; tmp < cut_height.length-1;)
            enforce(cut_height[tmp] < cut_height[++tmp],"cut_height cannot contains doublons");
    }


    // TODO: test the magic formula
    size_t lenRetVec = (cut_height.length + 1)
                      *(cut_width.length  + 1)
                      *(1 + (height - frame_height) / stride_height)
                      *(1 + (width  - frame_width)  / stride_width);

    return delegate(in Vector! T _v) {
        auto res = new Vector! T(lenRetVec, 0.1);



        size_t tmp_fr_x, tmp_fr_y,
               cell_h, cell_w,
               tmp_y, tmp_x,
               index = 0;

        CellRange!T cell_range;
        auto f_range_width = FrameRange(cut_width, frame_width);
        auto f_range_height = FrameRange(cut_height, frame_height);

        // we move the starting point of the frame using the first two "for"
        for (tmp_y = 0; tmp_y <= height - frame_height; tmp_y += stride_height) {
            for (tmp_x = 0; tmp_x <= width - frame_width; tmp_x += stride_width) {
                // We iterate over the frame's cells using the last two "for"
                foreach(cell_h, tmp_fr_y; f_range_height) {
                    foreach(cell_w, tmp_fr_x; f_range_width) {
                        // we create the range that will iterte over the
                        // values in the cell.
                        cell_range = new CellRange!T(tmp_fr_x + tmp_x,
                                                     tmp_fr_y + tmp_y,
                                                     cell_w, cell_h,
                                                     width, _v);
                        // And finally we just reduce the range using the
                        // the user defined reducer.
                        res[index++] = reducer(cell_range);
                    }
                }
            }
        }

        return res;
    };
}
unittest {
    write("Unittest: Utils: createPoolingFunction ...");

    size_t[] av = [1, 3];
    size_t[] lenav = [1, 2, 2];
    size_t[] valav = [0, 1, 3];
    size_t width = 5;

    auto fr = FrameRange(av, width);

    size_t tmp_index = 0;
    foreach(a, b; fr){
        assert((a == lenav[tmp_index]) && (b == valav[tmp_index]));
        tmp_index += 1;
    }
    tmp_index = 0;
    foreach(a, b; fr){
        assert((a == lenav[tmp_index]) && (b == valav[tmp_index]));
        tmp_index += 1;
        break;
    }
    tmp_index = 0;
    foreach(a, b; fr){
        assert((a == lenav[tmp_index]) && (b == valav[tmp_index]));
        tmp_index += 1;
        if (tmp_index >1)
            break;
    }


    // max pooling, stride 2, square frame, cut
    auto tmp1 = createPoolingFunction!double(10, 10, 2, 2, 4, 4, [2], [2],
                                                   delegate(CellRange!double _range) {
                                                        double s = _range.front;
                                                        foreach(a,e; _range)
                                                            s = max(s, e);
                                                        return s;
                                                   });

    // max pooling, stride 2, square frame, no cut
    auto tmp2 = createPoolingFunction!double(10, 10, 2, 2, 4, 4, [], [],
                                                   delegate(CellRange!double _range) {
                                                        double s = _range.front;
                                                        foreach(a,e; _range)
                                                            s = max(s, e);
                                                        return s;
                                                   });

    // min pooling, rectangular stride, rectangular frame, no cut
    auto tmp3 = createPoolingFunction!double(10, 10, 3, 2, 2, 4, [], [],
                                                   delegate(CellRange!double _range) {
                                                        double s = _range.front;
                                                        foreach(a,e; _range)
                                                            s = min(s, e);
                                                        return s;
                                                   });

    // average pooling, no stride, sqared frame, no cut
    auto tmp4 = createPoolingFunction!double(10, 10, 2, 2, 2, 2, [], [],
                                                   delegate(CellRange!double _range) {
                                                        double s = 0;
                                                        size_t len_range = 0;
                                                        foreach(e; _range){
                                                            s += e;
                                                            ++len_range;
                                                        }
                                                        return s / len_range;
                                                   });

    // max pooling, stride 2, square frame, multi-cut
    auto tmp5 = createPoolingFunction!double(10, 10, 2, 2, 10, 10, [2,7], [2,7],
                                                   delegate(CellRange!double _range) {
                                                        double s = _range.front;
                                                        foreach(a,e; _range)
                                                            s = max(s, e);
                                                        return s;
                                                   });

    auto layer1 = new FunctionalLayer!double(tmp1);
    auto layer2 = new FunctionalLayer!double(tmp2);
    auto layer3 = new FunctionalLayer!double(tmp3);
    auto layer4 = new FunctionalLayer!double(tmp4);
    auto layer5 = new FunctionalLayer!double(tmp5);

    auto vec = new Vector!double([1.,0.,  0.,1.,  1.,0.,  0.,0.,  0.,5.,
                                  0.,0.,  0.,2.,  2.,3.,  4.,4.,  1.,2.,
                                         
                                  1.,6.,  0.,1.,  1.,8.,  0.,3.,  0.,0.,
                                  4.,0.,  7.,2.,  2.,3.,  9.,4.,  0.,0.,
                                         
                                  .4,.6,  0.,1.5,  1.9,0.1,  0.4,0.7,  0.2,5.5,
                                  .9,1.5,  0.,2.5,  2.0,3.5,  4.0,4.5,  1.1,2.9,
                                         
                             -12311.,6., -30.,3., 8.,0.,  9.,0., -9.,-0.,
                                 -0.,1.,  7.,2.,  2.,3.,  4.,4., -1.,0.,
                                         
                                  1.,1.,  2.,2., -9.,2.,  0.,4.,  5.,5.,
                                  1.,1.,  2.,2., -5.,3.,  4.,0., -1.,2.,
                                 ]);


    auto res = layer1.compute(vec);
    auto re2 = layer2.compute(vec);
    auto re3 = layer3.compute(vec);
    auto re4 = layer4.compute(vec);
    auto re5 = layer5.compute(vec);

    auto s = [ 1, 2, 6, 7, 2, 3, 7, 8,
               3, 4, 8, 9, 4, 5, 9, 0,
               6, 7, 1.5, 2.5, 7, 8, 2.5, 3.5,
               8, 9, 3.5, 4.5, 9, 0, 4.5, 5.5,
               1.5, 2.5, 6, 7, 2.5, 3.5, 7, 8,
               3.5, 4.5, 8, 9, 4.5, 5.5, 9, 0,
               6, 7, 1, 2, 7, 8, 2, 3,
               8, 9, 3, 4, 9, 0, 4, 5];

    double[] s2 = [7, 8, 9, 9,
                   7, 8, 9, 9,
                   7, 8, 9, 9,
                   7, 8, 9, 9];

    double[] s3 = [0, 0, 0, 0, 0, 0, 0.1, 0, -12311,
                  -30, 0, -9];

    double[] s4 = [0.25, 0.75, 1.5, 2, 2, 2.75,
                   2.5, 3.5, 4, 0, 0.85, 1,
                   1.875, 2.4, 2.425, -3076,
                   -4.5, 3.25, 4.25, -2.5, 1,
                   2, -2.25, 2, 2.75];

    double[] s5 = [1, 4, 5, 6, 9, 5.5, 1, 7, 5];

    s[] -= res.v[];
    assert(abs(s.sum) <= 0.001);

    s2[] -= re2.v[];
    assert(abs(s2.sum) <= 0.001);

    auto ss = s3.dup;
    s3[] -= re3.v[];
    assert(abs(s3.sum) <= 0.001);

    s4[] -= re4.v[];
    assert(abs(s4.sum) <= 0.001);

    s5[] -= re5.v[];
    assert(abs(s5.sum) <= 0.001);



    writeln(" Done.");
}


/++ A function that put the _ownee array inside the _owner array such that
 +  any change in the _owner array will be also made to the _ownee array.
 +  In other words, the _owner array take ownership of the _ownee array.
 +/


 @safe @nogc pure
void takeOwnership_util_matrix(Mtype, T)(ref T[] _owner, ref Mtype _ownee, ref size_t _index)
{
    static if (is(Mtype : Matrix!T)) {
        takeOwnership_util!T(_owner, _ownee.params, _index);
    }
    else static if (is(Mtype : ReflectionMatrix!T)){
        takeOwnership_util!T(_owner, _ownee.vec.v, _index);
    }
    else static if (is(Mtype : DiagonalMatrix!T)){
        takeOwnership_util!T(_owner, _ownee.params, _index);
    }
    else static if (is(Mtype : UnitaryMatrix!T)){
        static if (is(Complex!T : T))
            assert(0, "Cannot serialize UnitaryMatrix in complex network for the moment.");
        else
            takeOwnership_util!T(_owner, _ownee.params, _index);
    }
    else static if (Mtype.stringof.startsWith("BlockMatrix")) {
        mixin("alias BlockMatType = "~Mtype.stringof.split("!")[1][1 .. $]~"!T;");
        foreach(tmp_block; _ownee.blocks){
            takeOwnership_util_matrix!(BlockMatType, T)(_owner, tmp_block, _index);
        }
    }
}


@safe @nogc pure
void takeOwnership_util(T)(ref T[] _owner, ref T[] _ownee, ref size_t _index)
{
    _owner[_index .. _index + _ownee.length] = _ownee[];
    _ownee = _owner[_index .. _index + _ownee.length];
    _index += _ownee.length;
}
unittest {
    write("                 takeOwnership (T[], T(], size_t))... ");
    size_t[] a = new size_t[4];

    size_t[] v1 = [1, 3];
    size_t[] v2 = [5, 7];

    size_t ind = 0;
    takeOwnership(a, v1, ind);
    takeOwnership(a, v2, ind);

    assert(a == [1, 3, 5, 7]);

    a[0] = 1000;
    a[3] = 2000;

    assert(ind == 4);
    assert(v1 == [1000, 3]);
    assert(v2 == [5, 2000]);

    writeln("Done.");
}