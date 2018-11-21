import std.conv: to;

/+ Used to create mutable version of an immutable type.
 +
 +/
mixin template makeMutable(string name_type, U: immutable(T), T)
{
    mixin("alias "~name_type~" = T;");
}

mixin template makeMutable(string name_type, U)
{
    mixin("alias "~name_type~" = U;");
}

/+ Function used to extract any given property of the layers
 + and return them in the form of a to!string(string[])
 +/
auto extract_property(string property, args...)()
{
    static if (args.length)
    {
        mixin makeMutable!("mutable_property_type", typeof(__traits(getMember, args[0], property)));
        mutable_property_type[args.length] unfolded_args;

        static foreach(i; 0 .. args.length)
        {
            unfolded_args[i] = __traits(getMember, args[i], property);
        }
        return unfolded_args;
    }
    else
    {
        return;
    }
}


/+ The shape of our Layers !
 + They will be templated struct.
 + Therefore, the template arguments of the neural networks will
 + actually be types, and all the infromation will be contained in
 + the template arguments passed to the layer.
 +
 +/
struct Linear(size_t _dim_out=0,
              bool _bias=false,
              string _name=null,
              alias _state=null,
              string[] _layers_in=null,
              string[] _layers_out=null)
{
    static immutable size_t dim_out = _dim_out;
    static immutable bool bias = _bias;
    static immutable string name = _name;
    static immutable typeof(_state) state = _state;
    static immutable string[] layers_in = _layers_in;
    static immutable string[] layers_out = _layers_out;
}


/+ Neural Network class
 +/
class NeuralNetwork(typeInput, size_t dimInput, Layers...)
{
    // Array of layers' name.
    static immutable string[Layers.length] layers_names = extract_property!("name",Layers);

    // Array of layers' bias existance.
    static immutable bool[Layers.length] layers_bias = extract_property!("bias",Layers);

    this()
    {
    }
}


void main(string[] args)
{
    static immutable auto nn = new NeuralNetwork!(
        real,
        10,
        Linear!(0, false, "Blue", [0, 1, 2], [], ["Unicorn"]),
        Linear!(0, false, "Unicorn", [0, 1, 2], ["Blue"], []),
    )();

    pragma(msg, nn.layers_names);
    pragma(msg, nn.layers_bias);
    static assert(nn.layers_names[0] == "Blue");
    assert(nn.layers_names[1] == "Unicorn");
}