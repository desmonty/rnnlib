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

    auto len = 100_000_000;

    auto vec = new Vector!(size_t, float)(len, 0.01);
    auto wec = new Vector!(size_t, float)(len, 0.01);

    writeln(vec.dot(wec));


    auto m1 = new Matrix!(uint, Complex!double)(4, 4, 0.1);
    auto m2 = new DiagonalMatrix!(uint, Complex!double)(4, 0.1);
    auto m3 = new ReflectionMatrix!(uint, Complex!double)(4, 0.1);
    auto m4 = new FourierMatrix!(uint, Complex!double)(4);
    auto bm = new BlockMatrix!(uint, Complex!double)(16, 4, [m1,m2,m3,m4], false);

    auto v = new Vector!(uint, Complex!double)(16);
    v[0]=complex(0,0);   v[1]=complex(1,0);
    v[2]=complex(2, 0);   v[3]=complex(3, 0);
    v[4]=complex(4, 0);   v[5]=complex(5, 0);
    v[6]=complex(6, 0);   v[7]=complex(7, 0);
    v[8]=complex(8, 0);   v[9]=complex(9, 0);
    v[10]=complex(10, 0); v[11]=complex(11, 0);
    v[12]=complex(12, 0); v[13]=complex(13, 0);
    v[14]=complex(14, 0); v[15]=complex(15, 0);

    writeln(v.v);
    v = bm * v;
    writeln(v.v);

    auto endttime = Clock.currTime();
    auto duration = endttime - stattime;
    writeln("Time ", duration);
    writeln("End");

}