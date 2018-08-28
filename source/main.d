module source.main;

import std.algorithm: copy, map;
import std.datetime;
import std.complex;
import std.functional: toDelegate;
import std.getopt;
import std.math;
import std.random;
import std.stdio : writeln;

import source.Layer;
import source.Matrix;
import source.NeuralNetwork;
import source.Parameter;
import source.Utils;

import source.optimizers.random_search;
import source.optimizers.nelder_mead;

/// GLOBALS

auto optimizers_tests = false;

///


void main(string[] args) {
    auto optinfo = getopt(args,
        "optimizers_tests", &optimizers_tests
    );

    if (optimizers_tests)
        run_optimizers_tests();
}

void run_optimizers_tests() {
    random_search_tests();
    nelder_mead_tests();
}