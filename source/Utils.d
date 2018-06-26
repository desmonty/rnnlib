module source.Utils;

import std.algorithm;
import std.complex: abs;
import std.conv: to;
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

bool isOneOf(string strKey, string[] strTable) {
    foreach(key; strTable)
        if(strKey == key)
            return true;
    return false;
}
unittest {
    enum string[3] kafka = ["ae", "ry", "up"];
    enum string k1 = "ar";
    enum string k2 = "up";

    static assert(!isOneOf(k1, kafka));
    static assert(isOneOf(k2, kafka));
    assert(!isOneOf(k1, kafka));
    assert(isOneOf(k2, kafka));
}

/++ FrameRange is used to enhanced clean up the following code.
 +  -By adding complexity ..?
 +  -Yes ! It is worth it because it will allow us to do a simple
 +    "foreach" and get both...
 +/
struct FrameRange(size_t[] _arr, size_t _width)
{
    enum size_t[] arr = _arr;
    enum size_t width = _width;

    int opApply(scope int delegate(size_t, size_t) dg)
    {

        // if the cut_frame is empty, we do not cut the frame
        // and so we just return the first position with the width.
        static if (!arr)
            return dg(width, 0);
        else {
            int result;
            size_t frame_len = arr[0];

            result = dg(frame_len, 0);
            if (result)
                return result;

            enum size_t len_1 = arr.length - 1;
            foreach (index; 0 .. len_1) {
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
}


/++ Function that create a general pooling function
 +  See https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
 +  for description.
 +
 +  Template Args:
 +      T: type of the vector's elements.
 +/
@safe pure
auto createPoolingFunction(T, alias reducer, size_t height, size_t width,
                           size_t stride_height, size_t stride_width,
                           size_t frame_height, size_t frame_width,
                           size_t[] cut_height, size_t[] cut_width)()
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
            reducer (alias): Function that takes two arguments and return a value
                             of the same type (e.g. max for max-pooling, ...)

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
    static if (cut_width) {
        static assert(minElement(cut_width), "cut_width cannot contains '0'");
        static assert(maxElement(cut_width) < width, "max(cut_width) must be < width-1");
        static assert(isSorted(cut_width), "cut_width must be sorted");

        static foreach(tmp; 0 .. (cut_width.length-1))
            static assert(cut_width[tmp] < cut_width[tmp+1],"cut_width cannot contains doublons");
    }
    static if (cut_height) {
        static assert(minElement(cut_height), "cut_height cannot contains '0'");
        static assert(maxElement(cut_height) < height, "max(cut_height) must be < height-1");
        static assert(isSorted(cut_height), "cut_height must be sorted");

        static foreach(tmp; 0 .. (cut_height.length-1))
            static assert(cut_height[tmp] < cut_height[tmp+1],"cut_height cannot contains doublons");
    }

    enum size_t lenRetVec = (cut_height.length + 1)
                          * (cut_width.length  + 1)
                          * (1 + (height - frame_height) / stride_height)
                          * (1 + (width  - frame_width)  / stride_width);

    return "
        auto res = new Vector! T("~to!string(lenRetVec)~", 0.1);

        size_t tmp_fr_x, tmp_fr_y,
               cell_h, cell_w,
               tmp_y, tmp_x,
               index,
               cur_pos,
               pos_x, pos_y,
               shift, tmp_inner;

        auto f_range_width = FrameRange!("~to!string(cut_width)~", "~to!string(frame_width)~")();
        auto f_range_height = FrameRange!("~to!string(cut_height)~", "~to!string(frame_height)~")();

        // we move the starting point of the frame using the first two 'for'
        for (tmp_y = 0; tmp_y <= "~to!string(height)~" - "~to!string(frame_height)~"; tmp_y += "~to!string(stride_height)~") {
            for (tmp_x = 0; tmp_x <= "~to!string(width)~" - "~to!string(frame_width)~"; tmp_x += "~to!string(stride_width)~") {
                // We iterate over the frame's cells using the last two 'for'
                foreach(cell_h, tmp_fr_y; f_range_height) {
                    foreach(cell_w, tmp_fr_x; f_range_width) {
                        // we create the range that will iterte over the
                        // values in the cell.
                        cur_pos = (tmp_fr_y + tmp_y) * "~to!string(width)~" + tmp_x + tmp_fr_x;
                        shift = "~to!string(width)~" - cell_w;

                        // And finally we just reduce the range using the
                        // the user defined reducer.

                        tmp_inner = cur_pos;
                        for (pos_y = 0; pos_y <= (cell_h-1); ++pos_y) {
                            for (pos_x = 0; pos_x <= (cell_w-1); ++pos_x) {
                                if (cur_pos == tmp_inner)
                                    res[index] = _v[tmp_inner];
                                else
                                    res[index] = "~reducer("res[index]","_v[tmp_inner]")~";
                                tmp_inner += 1;
                            }
                            tmp_inner += shift;
                        }
                        ++index;
                    }
                }
            }
        }

        return res;
    ";
}
unittest {
    write("Unittest: Utils: createPoolingFunction ...");

    enum size_t[] av = [1, 3];
    enum size_t[] lenav = [1, 2, 2];
    enum size_t[] valav = [0, 1, 3];
    enum size_t width = 5;

    auto fr = FrameRange!(av, width)();

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
    enum auto tmp1 = createPoolingFunction!(double, (a, b) => "max("~a~", "~b~")", 10, 10, 2, 2, 4, 4, [2], [2]);

    // max pooling, stride 2, square frame, no cut
    enum auto tmp2 = createPoolingFunction!(double, (a, b) => "max("~a~", "~b~")", 10, 10, 2, 2, 4, 4, [], []);

    // min pooling, rectangular stride, rectangular frame, no cut
    enum auto tmp3 = createPoolingFunction!(double, (a, b) => "min("~a~", "~b~")", 10, 10, 3, 2, 2, 4, [], []);

    // sum pooling, no stride, sqared frame, no cut
    enum auto tmp4 = createPoolingFunction!(double, (a, b) => a~"+"~b, 10, 10, 2, 2, 2, 2, [], []);

    // max pooling, stride 2, square frame, multi-cut
    enum auto tmp5 = createPoolingFunction!(double, (a, b) => "max("~a~", "~b~")", 10, 10, 2, 2, 10, 10, [2,7], [2,7]);

    // max pooling, no stride, square frame, multi-cut
    enum auto tmp6 = createPoolingFunction!(double, (a, b) => "max("~a~","~b~")", 10, 10, 10, 10, 10, 10, [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]);

    auto layer1 = new FunctionalLayer!(double, tmp1);
    auto layer2 = new FunctionalLayer!(double, tmp2);
    auto layer3 = new FunctionalLayer!(double, tmp3);
    auto layer4 = new FunctionalLayer!(double, tmp4);
    auto layer5 = new FunctionalLayer!(double, tmp5);
    auto layer6 = new FunctionalLayer!(double, tmp6);

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

    double[] s4 = [1.0, 3.0, 6.0, 8.0, 8.0,
                   11.0, 10.0, 14.0, 16.0, 0.0,
                   3.4, 4.0, 7.5, 9.6, 9.7,
                   -12304.0, -18.0, 13.0, 17.0, -10.0,
                   4.0, 8.0, -9.0, 8.0, 11.0];

    double[] s5 = [1, 4, 5, 6, 9, 5.5, 1, 7, 5];

    s[] -= res.v[];
    assert(abs(s.sum) <= 0.001);

    s2[] -= re2.v[];
    assert(abs(s2.sum) <= 0.001);

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
unittest {
    write("                 takeOwnership_util_matrix... ");

    auto v = new Vector!real(4, 1.0);
    auto v2 = new Vector!real(2, 1.0);

    auto r = new ReflectionMatrix!real(4, 1.0);
    auto r1 = new ReflectionMatrix!real([1.0, 1.0, 1.0, 1.0]);

    auto d = new DiagonalMatrix!real(4, 2.0);

    auto u = new UnitaryMatrix!real(4, 1.0);
    auto u1 = new UnitaryMatrix!real(4); u1.params[] = 1.0; u1.perm = u.perm;

    auto gm = new Matrix!real(2, 2.0);
    auto gm1 = new Matrix!real(2); gm1.params[] = 1.0;

    auto b = new BlockMatrix!(Matrix!real)(4, 4, 2, [gm, gm], false);
    auto b1 = new BlockMatrix!(Matrix!real)(4, 4, 2, [gm1, gm1], false);

    auto array_bouïlla = new real[4 + 4 + 4*7 + 4 + 4*2];
    size_t indexouille = 0;

    takeOwnership_util_matrix!(ReflectionMatrix!real, real)(array_bouïlla, r, indexouille);
    takeOwnership_util_matrix!(DiagonalMatrix!real, real)(array_bouïlla, d, indexouille);
    takeOwnership_util_matrix!(UnitaryMatrix!real, real)(array_bouïlla, u, indexouille);
    takeOwnership_util_matrix!(Matrix!real, real)(array_bouïlla, gm, indexouille);
    takeOwnership_util_matrix!(BlockMatrix!(Matrix!real), real)(array_bouïlla, b, indexouille);

    array_bouïlla[] = 1.0;

    auto res_1 = r * v;
    res_1 = r1 * res_1;
    res_1 -= v;
    assert(res_1.norm!"L2" <= 0.00001);

    auto res_2 = d * v;
    res_2 -= v;
    assert(res_2.norm!"L2" <= 0.00001);

    auto res_3 = u * v;
    res_3 -= u1 * v;
    assert(res_3.norm!"L2" <= 0.00001);

    auto res_4 = gm * v2;
    res_4 -= gm1 * v2;
    assert(res_4.norm!"L2" <= 0.00001);

    auto res_5 = b * v;
    res_5 -= b1 * v;
    assert(res_5.norm!"L2" <= 0.00001);

    writeln("Done.");
}

@safe @nogc pure
void takeOwnership_util(T)(ref T[] _owner, ref T[] _ownee, ref size_t _index)
{
    _owner[_index .. _index + _ownee.length] = _ownee[];
    _ownee = _owner[_index .. _index + _ownee.length];
    _index += _ownee.length;
}
unittest {
    write("                 takeOwnership_util... ");
    size_t[] a = new size_t[4];

    size_t[] v1 = [1, 3];
    size_t[] v2 = [5, 7];

    size_t ind = 0;
    takeOwnership_util(a, v1, ind);
    takeOwnership_util(a, v2, ind);

    assert(a == [1, 3, 5, 7]);

    a[0] = 1000;
    a[3] = 2000;

    assert(ind == 4);
    assert(v1 == [1000, 3]);
    assert(v2 == [5, 2000]);

    writeln("Done.");
}