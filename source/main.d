module source.main;

import std.algorithm: copy, map;
import std.datetime;
import std.complex;
import std.math;
import std.random;
import std.stdio : writeln;

import source.Parameters;
import source.Matrix;

// source ~/dlang/ldc-1.7.0/activate
//ldc2 -O3 -ffast-math *.d
/+
abstract class Layer
{
    ///
    string name;

    /// Neural layers pointing to this layer.
    Layer[] inLayers;

    /// Neural layers the layer is pointing to.
    Layer[] outLayers;

    bool Ps_parameter;

    this(){}

    void set_name(string _name)
    {
        if (_name !is null)
            name = _name;
    }

    abstract void apply();
    abstract Vector pop();
}

class LinearLayer : Layer
{
    Matrix W;
    Vector bias;
    Vector hidden;


    void init()


    override Vector pop()
    {
        return hidden;
    }

    override void apply()
    {
        hidden[] = 0;
        foreach(layer; inLayers)
        {

        }
    }
}

class InputLayer : Layer
class FunctionLayer : Layer
class RecurrentLayer : Layer
+/



void main()
{
    if (false)
    {
        auto v = new Vector!(uint, float)(5);
        auto c = new Vector!(size_t, Complex!double)(5);
        auto m = new Matrix!(uint, float)(10);
        auto d = new DiagonalMatrix!(uint, Complex!double)(5);
        auto p = new PermutationMatrix!(uint, Complex!double)(5);
        auto b = new BlockMatrix!(uint, Complex!double)(10u, 5u, [p, d], true);
        auto r = new ReflectionMatrix!(long, Complex!float)(10);
        auto f = new FourierMatrix!(ushort, Complex!real)(cast(ushort) 65536u);
    }

}