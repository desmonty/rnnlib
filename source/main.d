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
    auto length = 15;

    writeln("Begin");
    auto stattime = Clock.currTime();

    if (false)
    {
        auto v = new Vector!(uint, float)(5);
        auto c = new Vector!(uint, Complex!double)(5);
        auto m = new Matrix!(uint, float)(10);
        auto d = new DiagonalMatrix!(uint, double)(5);
        auto p = new PermutationMatrix!(uint, Complex!double)(5);
        //auto b = new BlockMatrix!(uint, Complex!double)(10, 10, [p, c], true);
        auto r = new ReflectionMatrix!(long, Complex!float)(10);
        auto f = new FourierMatrix!(ushort, Complex!real)(cast(ushort) 65536u);
    }

    auto len = 10_000_000;

    auto v = new Vector!(size_t, real)(len, 0.01);
    auto w = new Vector!(size_t, real)(len, 0.01);

    writeln(v.dot(w));
    writeln("\a");

    auto m = new PermutationMatrix!(size_t, float)(3, 0.1);
    auto ve = new Vector!(size_t, float)(3);
    ve[0] = 0;
    ve[1] = 1;
    ve[2] = 40;

    writeln(ve.v);
    ve = m*ve;
    writeln(ve.v);
    writeln(m.perm);

    MatrixAbstract!(size_t, float) mp = m;
    ve = mp * ve;
    writeln(ve.v);


    writeln(typeof(m).stringof);
    writeln(typeof(mp).stringof);

    auto endttime = Clock.currTime();
    auto duration = endttime - stattime;
    writeln("Time ", duration);
    writeln("End");

}