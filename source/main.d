module source.main;

import std.algorithm: copy, map;
import std.datetime;
import std.complex;
import std.math;
import std.random;
import std.stdio : writeln;

import source.Layers;
import source.Parameters;
import source.Matrix;

// source ~/dlang/ldc-1.7.0/activate
//ldc2 -O3 -ffast-math *.d


unittest
{
    import std.stdio: writeln;

    writeln("-------------------------------");
    writeln("       Begin Unittesting       ");
    writeln("-------------------------------");
}

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

    writeln("-------------------------------");
    writeln("        End Unittesting        ");
    writeln("-------------------------------");
}