module source.main;

import std.algorithm: copy, map;
import std.datetime;
import std.complex;
import std.functional: toDelegate;
import std.math;
import std.random;
import std.stdio : writeln;

import source.Layer;
import source.Matrix;
import source.NeuralNetwork;
import source.Parameter;

void main()
{
    if (true)
    {
        auto v = new Vector!float(5);
        auto c = new Vector!(Complex!double)(5);
        auto m = new Matrix!float(10);
        auto d = new DiagonalMatrix!(Complex!double)(5);
        auto p = new PermutationMatrix!(Complex!double)(5);
        auto b = new BlockMatrix!(Complex!double)(10u, 5u, [p, d], true);
        auto r = new ReflectionMatrix!(Complex!float)(10);
        auto f = new FourierMatrix!(Complex!real)(cast(ushort) 65536u);
    }
}