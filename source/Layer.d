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




/+  Function that create a general pooling function
    See https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    for description.

    Template Args:
        S: type of the indices.
        T: type of the vector's elements.
 +/
auto createPoolingfunction(S, T)(in S height, in S width,
                                 in S stride_height, in S stride_width,
                                 in S frame_height, in S frame_width,
                                 in S[] cut_height, in S[] cut_width,
                                 T delegate(InputRange!T) reducer)
{
    /+  This function create a delegate.
        That delegate take as input a vector and apply
        a pooling of specified form to it.
        The first element of the vector is assumed to be the Top Left "pixel"
        and the rest are seen in a "Left-Right/Top-Left" fashion.

        Args:
            height (S): height of the picture (let's assume it's a pic). 
            width (S): width of the picture (let's assume it's a pic). 
            stride_height (S): number of pixel to move to take the next frame
            stride_width (S): number of pixel to move to take the next frame
            frame_height (S): height of the frame to look at
            frame_width (S): width of the frame to look at
            cut_height (S[]): List of indices to cut the frame (see below).
            cut_width (S[]): List of indices to cut the frame (see below).
            reducer (T delegate(InputRange)): Function that takes an InputRange
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
    enforce(minElement(cut_width), "cut_width cannot contains '0'");
    enforce(minElement(cut_height), "cut_height cannot contains '0'");
    enforce(maxElement(cut_width) < width, "max(cut_width) must be < width-1");
    enforce(maxElement(cut_height) < height, "max(cut_height) must be < height-1");
    enforce(isSorted(cut_width), "cut_width must be sorted");
    enforce(isSorted(cut_height), "cut_height must be sorted");

    for (S tmp = 0; tmp < cut_width.length-1;)
        enforce(cut_width[tmp] < cut_width[++tmp],
               "cut_width cannot contains doublons");
    for (S tmp = 0; tmp < cut_height.length-1;)
        enforce(cut_height[tmp] < cut_height[++tmp],
               "cut_height cannot contains doublons");

    // TODO: test the magic formula
    S lenRetVec = (cast(S) cut_height.length + 1)
                  *(cast(S) cut_width.length  + 1)
                  *(1 + (height - frame_height + 1) / stride_height)
                  *(1 + (width  - frame_width  + 1) / stride_width);

    return delegate(in Vector!(S, T) _v) {
        auto res = new Vector!(S, T)(lenRetVec, 0.1);

        /++ CellRange is used to get all the values inside a cell that
         +  will be reduced together.
         +  It allows us to let the user defines its own reducer in a very
         +  simple way.
         +/ 
        static class CellRange: InputRange!T{
          private
          {
            S pos_x, pos_y,
              cur_pos,
              cell_w, cell_h,
              shift, width;

            const Vector!(S,T) vec;

            bool is_empty = false;
          }

            this(in S _pos_x, in S _pos_y, in S _cell_w,
                 in S _cell_h, in S _width,
                 ref const Vector!(S,T) _v)
            {
                cur_pos = _pos_y * _width + _pos_x;
                pos_x = 0;
                pos_y = 0;
                width = _width;
                cell_h = _cell_h - 1;
                cell_w = _cell_w - 1;
                vec = _v;

                shift = width - cell_w;
            }

            @property
            const
            T front()
            {
                return vec[cur_pos];
            }

            T moveFront()
            {
                return vec[cur_pos];
            }

            void popFront()
            {
            }

            @property
            const
            bool empty()
            {
                return is_empty;
            }

            int opApply(scope int delegate(T) dg)
            {
                int result;

                S tmp = cur_pos;
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

                S tmp = cur_pos;
                T tmp_val;
                auto shiftm1 = shift - 1;

                outer: for (pos_y = 0; pos_y <= cell_h; ++pos_y) {
                    for (pos_x = 0; pos_x <= cell_w; ++pos_x) {
                        tmp_val = vec[tmp];
                        result = dg(tmp, tmp_val);
                        tmp += 1;

                        if (result)
                            break outer;
                    }
                    tmp += shiftm1;
                }
                return result;
            }
        }

        /++ -FrameRange is used to enhanced clean up the following code.
         +  -By adding complexity ..?
         +  -Yes ! It is worth it because it will allow us to do a simple
         +    "foreach" and get both...
         +/
        static struct FrameRange {
            private {
                const S[] arr;
                const S width;
            }

            this(ref const S[] _arr, const S _width) {
                arr = _arr;
                width = _width;
            }
            
            int opApply(scope int delegate(S) dg)
            {
                int result;

                result = dg(0);
                if (result)
                    return result;

                immutable size_t len = arr.length;
                for (size_t index = 0; index < len; ++index) {
                    result = dg(arr[index]);
                    if (result)
                        break;
                }
                return result;
            }
            
            int opApply(scope int delegate(S, S) dg)
            {
                int result;

                S frame_len = arr[0];

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

        S tmp_fr_x, tmp_fr_y,
          cell_h, cell_w,
          tmp_y, tmp_x,
          index = 0;

        CellRange cell_range;
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
                        cell_range = new CellRange(tmp_fr_x + tmp_x,
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
    alias S = int;
    static struct FrameRange {
        private {
            const(S)[] arr;
            const(S) width;
        }

        this(ref in S[] _arr, in S _width) {
            arr = _arr;
            width = _width;
        }
        
        int opApply(scope int delegate(S) dg)
        {
            int result;

            result = dg(0);
            if (result)
                return result;

            immutable size_t len = arr.length;
            for (size_t index = 0; index < len; ++index) {
                result = dg(arr[index]);
                if (result)
                    break;
            }
            return result;
        }
        
        int opApply(scope int delegate(S, S) dg)
        {
            int result;

            S frame_len = arr[0];

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

    int[] av = [1, 3];
    int[] lenav = [1, 2, 2];
    int[] valav = [0, 1, 3];
    int width = 5;

    auto fr = FrameRange(av, width);

    int tmp_index = 0;
    foreach(a, b; fr){
        assert((a == lenav[tmp_index]) && (b == valav[tmp_index]));
        tmp_index += 1;
    }
    tmp_index = 0;
    foreach(b; fr){
        assert(b == valav[tmp_index]);
        tmp_index += 1;
    }
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
        - Matrix Layer
        - convnet
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
    static if (is(Complex!T : T))
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
    Vector!(S,T) delegate(Vector!(S,T), Parameter[]) func;

    /// Used by the optimizer to know if it must optimize the layer.
    bool isLearnable = false;

    /// Used
    void set_name(string _name)
    {
        if (_name !is null)
            name = _name;
    }

    abstract Vector!(S,T) compute(Vector!(S,T));
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
    /+ This implements common functions that are not implemented already.
       It includes the following:
            -SoftMax
            -relu
            -modRelu
     +/
    this(string easyfunc, in S size_in=0)
    {
        switch (easyfunc)
        {
            case "relu":
                static if (!is(Complex!T : T)) {
                    func =
                        delegate(Vector!(S,T) _v, Parameter[] _p) {
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
                    isLearnable = true;
                    params = new Parameter[1];
                    params[0] = new Vector!(S,Tc)(size_in, 1.0);
                    
                    func =
                        delegate(Vector!(S,T) _v, Parameter[] _p) {
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
                static if (!is(Complex!T : T))
                    func =
                        delegate(Vector!(S,T) _v, Parameter[] _p) {
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
    this(Vector!(S,T) delegate(Vector!(S,T)) _func)
    {
        func = delegate(Vector!(S,T) _v, Parameter[] _p) {
            return _func(_v);
        };
    }

    // The function to apply to the vector. Can be anything. FUNCTION
    this(Vector!(S,T) function(Vector!(S,T)) _func)
    {
        this(toDelegate(_func));
    }

    // Create an element-wise function that apply a provided
    // function to a vector. DELEGATE
    this(T delegate(T) _func)
    {
        func = delegate(Vector!(S,T) _v, Parameter[] _p) {
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
        this(delegate(Vector!(S,T) _v) {
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
    Vector!(S,T) compute(Vector!(S,T) _v)
    {
        return func(_v, params);
    }
}
unittest {
    write("Unittest FunctionalLayer ... ");

    alias Vec = Vector!(uint, Complex!double);
    alias Fl = FunctionalLayer!(uint, Complex!double);

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
    (cast(Vector!(uint, double)) f5.params[0])[0] = -0.9;
    (cast(Vector!(uint, double)) f5.params[0])[1] = -0.9;
    (cast(Vector!(uint, double)) f5.params[0])[2] = -0.9;
    (cast(Vector!(uint, double)) f5.params[0])[3] = -0.9;
    auto v5 = f5.compute(w);
    assert(abs(v5.sum) < 0.0001);

    // vector 'e' doesn't have the right length.
    assertThrown(f1.compute(e));
    // modRelu must be given the vector's size to create parameters.
    assertThrown(new Fl("modRelu"));
    // relu takes only real-valued vectors.
    assertThrown(new FunctionalLayer!(uint, Complex!double)("relu"));
    // softmax takes only real-valued vectors.
    assertThrown(new FunctionalLayer!(uint, Complex!double)("softmax"));
    // modRelu takes only complex-valued vectors.
    assertThrown(new FunctionalLayer!(uint, double)("modRelu"));
    // Incorrect function name.
    assertThrown(new FunctionalLayer!(uint, real)("this is incorrect."));

    auto vr = new Vector!(size_t, double)([0.0, 1.0, pi, -1.0]);

    // relu function.
    auto f6 = new FunctionalLayer!(size_t, double)("relu");
    auto vr6 = f6.compute(vr);
    assert(abs(vr6.sum - 1.0 - pi) <= 0.01);

    // softmax function.
    auto f7 = new FunctionalLayer!(size_t, double)("softmax");
    auto vr7 = f7.compute(vr);
    assert(abs(vr7.sum - 1.0) <= 0.001);

    // set the name of a layer.
    f1.set_name("f1");
    assert(f1.name == "f1");

    auto tmp = createPoolingfunction!(int, double)(10, 10, 2, 2, 4, 4, [2], [2],
                                                   delegate(InputRange!double _range) {
                                                        double s = _range.front;
                                                        _range.popFront();
                                                        foreach(a,e; _range)
                                                            s = max(s, e);
                                                        return s;
                                                   });

    auto layer1 = new FunctionalLayer!(int, double)(tmp);

    auto vec = new Vector!(int, double)([1.,0.,  0.,1.,  1.,0.,  0.,0.,  0.,5.,
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

    auto vv = new Vector!(int, double)(100, 0.2);

    write("Done.\n");
}


