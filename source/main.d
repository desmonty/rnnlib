module source.main;

import std.algorithm: copy, map;
import std.datetime;
import std.complex;
import std.functional: toDelegate;
import std.math;
import std.random;
import std.stdio : writeln;

import source.Layer;
import source.Parameter;
import source.Matrix;

void main()
{
    
    if (true)
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

    class A {
        real delegate(in real[], in bool[]) pure func;

        this(real function(real) pure _func)
        {
            this(toDelegate(_func));
        }

        this(real delegate(real) pure _func)
        {
            func = delegate(in real[] _v, in bool[] _b) pure {
                real s = 0;
                foreach(i; 0 .. _v.length){
                    if (_b[i])
                        s += _func(_v[i]);
                }
                return s;
            };
        }


        real apply(in real[] _v, in bool[] _b)
        {
            return func(_v, _b);
        }
    }

    real blue(real x) pure {
        return x*x - x - 1.0;
    }

    writeln(typeof(&blue).stringof);
    writeln(typeof(&std.math.cos).stringof);

    immutable real pi = 3.1415926535;
    immutable real[] arr = [ pi, -pi/2.0, 0.5 + sqrt(5.0)/2.0, 1.0];
    
    A a = new A(&blue);
    writeln(a.apply(arr, [false, false, true, true]));
   

    A b = new A(&std.math.cos);
    writeln(b.apply(arr, [true, true, false, false]));


    //
    int isSomething(int[] arra = null) {
        if (arra !is null)
            return 1;
        return 0;
    }

    assert(isSomething([1]));
    assert(!isSomething());
    assert(!isSomething(null));

}