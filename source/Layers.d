module source.Layers;

/+  The layers of the Neural Networks.

    Basically, each layer can be seen as a function,
    which take a vector and return another vecotr of
    a possibly different length.
    Those functions have parameters (matrix, bias, ..)
    which are specific to each kind of layer (linear, recurrent, ..).
    All the logic of how these layers are assembled is
    in the NeuralNet object.

    If they are trained with an evolutionary algorithm
    (Which is the primary goal), we will need to have
    some function to handle *mutation* and *crossover*.
    This should be enough for many of the optimization
    we will implement (PSA, GA, RS, ES, and other obscure
    acronym...).

    The gradient of the layer will be difficult to
    compute due to the will to play with heavily recurrent
    networks (which is not the common cas because of
    the use of gradient-based optimization).
    However, it would be very interesting to know the gradient
    of the NeuralNet and could be investigated in this project.


 +/

abstract class Layer(S,T)
{
    static if (T.stringof.startsWith("Complex"))
        mixin("alias Tc = "~(T.stringof[8 .. $])~";");
    else alias Tc = T;

    /// Name of the layer.
    string name;

    /// Type of the layer. 
    string typeId;

    S size_in;
    S size_out;

    void set_name(string _name)
    {
        if (_name !is null)
            name = _name;
    }

    abstract Vector!(S,T) apply();
}

/+ This layer implement a simple linear matrix transformation
   applied to the vector followed by adding a bias vector
   (which can be turner off). 
 +/
class LinearLayer(S,T) : Layer!(S,T)
{
    MatrixAbstract!(S,T) W;
    Vector!(S,T) bias;
    bool keep_bias;

    this(ref MatrixAbstract!(S,T) _W, bool _keep_bias = true)
    {
        W = _W;
        keep_bias = _keep_bias;
    }

    /// Random initialization of the matrix and vector.
    void init(in S _size_in, in S _size_out)
    {
        if (keep_bias) {
            bias = new Vector!(S,T)(size_out);
            bias.v[] = 0;
        }
    }

    /// Apply the function implemented by the layer to the vector.
    override
    auto apply(in Vector!(S,T) vec)
    {
        auto tmp_vec = new Vector!(S,T)(size_in);
        tmp *= W;
        if (keep_bias)
            tmp += bias;
        return tmp;
    }
}

class FunctionLayer(S,T) : Layer!(S,T)
{
    Vector!(S,T) delegate(Vector!(S,T) v) func;

    this(string easyfunc)
    {
        static if 
        func = delegate(Vector!(S,T) v) {

        };
        switch (easyfunc)
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
}
/+
class InputLayer : Layer
class RecurrentLayer : Layer
+/
