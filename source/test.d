/+ Function used exclusively to extract the names of the layers and
 + return them in the form of a array.stringof
 +
 + Ideally, we need to find a we to generalize it to extract _any_
 + piece of information cntained in the layers.
 +/
auto extract_name(args...)()
{
    string unfolded_args= "[";

    static foreach(i; 0 .. args.length)
    {
        unfolded_args ~= "\"";
        unfolded_args ~= args[i].name;
        unfolded_args ~= "\",";
    }

    unfolded_args ~= "]";
    return unfolded_args;
}

/+ The shape of our Layers !
 + They will be templated struct.
 + Therefore, the template arguments of the neural networks will
 + actually be types, and all the infromation will be contained in
 + the template arguments passed to the layer.
 +
 +/
struct Linear(string _name)
{
    static immutable string name = _name;
}

/+ Neural Network class
 +/
class NeuralNetwork(Layers...)
{
    // Array of layers' name.
    mixin("static immutable string[Layers.length] layers_names = "
         ~extract_name!(Layers) 
         ~";");

    this()
    {
    }
}


void main(string[] args)
{
    static immutable auto nn = new NeuralNetwork!(
        Linear!("Blue"),
        Linear!("Unicorn"),
    )();

    pragma(msg, nn.layers_names);
    static assert(nn.layers_names[0] == "Blue");
    assert(nn.layers_names[1] == "Unicorn");
}