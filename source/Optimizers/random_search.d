module source.optimizers.random_search;

import core.stdc.math: isnan;
import std.algorithm: max;
import std.conv: to;
import std.functional: toDelegate;
import std.math: abs, pow, cos, exp, sqrt;
import std.mathspecial: normalDistributionInverse;
import std.random: uniform;
import std.stdio: writeln, write;

import source.Matrix;
import source.NeuralNetwork;
import source.Parameter;


T random_search(T)(ref Vector!T _v, T function(in Vector!T) _obj,
                   in size_t _num_iterations=50UL,
                   in size_t _patience=5UL,
                   in T _exponential_factor_radius=0.5,
                   in T _lower_bound_radius=1e-4,
                   in bool _random_start=true)
{
    return random_search!T(_v, toDelegate(_obj), _num_iterations,
                            _patience, _exponential_factor_radius,
                            _lower_bound_radius, _random_start);
}

T random_search(T)(ref Vector!T _v, T delegate(in Vector!T) _obj,
                   in size_t _num_iterations=50UL,
                   in size_t _patience=5UL,
                   in T _exponential_factor_radius=0.5,
                   in T _lower_bound_radius=1e-4,
                   in bool _random_start=true)
{
    /+
     + Arguments:
     +
     +  - _v (Vector!T): will contains a local minimum of the _obj.
     +
     +  - _obj (T delegate(Vector!T)): the function to minimize.
     +
     +  - _num_iterations (size_t): Maximum number of iterations of the algorithm.
     +
     +  - _patience (size_t): Reduce hypersphere radius if _patience itertations
     +                        has been executed without improvement.
     +
     +  - _exponential_factor_radius (T): value we use to rescale the size of step
     +                                    when '_patence' tries didn't provide us 
     +                                    with a better value.
     +
     +  - _lower_bound_radius (T): Stop criterion on the minimal acceptable scaling
     +                             one can use.
     +
     +  - _random_start (bool): Whether we should start the algorithm at the provided
     +                          vector or create a random one.
     +
     +/

    /// Initialization
    immutable T default_upper_bound = to!T( 1); 
    immutable T default_lower_bound = to!T(-1); 

    if (_random_start)
        foreach(i; 0 .. _v.length)
            _v[i] = uniform(default_lower_bound, default_upper_bound);

    auto current_value = _obj(_v);
    auto current_iteration = 0UL;
    auto current_radius = to!T(1.0);
    size_t current_patience = _patience;

    auto neighbour = _v.dup;
    auto neighbour_value = current_value;

    while_loop: while (current_iteration++ <_num_iterations) {
        // If we have no more patience, divide the radius by two.
        if (!current_patience){
            current_patience = _patience;
            current_radius *= _exponential_factor_radius;
            if (current_radius <= _lower_bound_radius)
                break while_loop;
        }

        /// Create neighbour
        // Sample from normal distribution
        foreach(i; 0 .. neighbour.length)
            neighbour[i] = to!T(normalDistributionInverse(uniform!"()"(0.0, 1.0)));// ~ Normal(0, 1)

        // make vector norm equal to 1.
        neighbour /= neighbour.norm!"L2"/current_radius;
        
        // move neighbour near the xurrent solution.
        neighbour += _v;

        /// Compute new value.
        neighbour_value = _obj(neighbour);

        /// If new value is better than the old, we move _v to the neighbour.
        // And become patient again.
        if (neighbour_value < current_value) {
            current_value = neighbour_value;
            _v.v[] = neighbour.v[];
            current_patience = _patience;
        }
        // Else we lose some patience.
        else {
            --current_patience;
        }
    }

    return current_value;
}
unittest {
    size_t dimension = 2;
    Vector!real origin = new Vector!real(dimension, 0.0);

    real func_obj(in Vector!real _v) {
        // We are sure that any point will do better than the origin here.
        return -_v.norm!"L2";
    }

    auto origin_tmp = origin.dup;

    
    /+ We make only one iteration and stop to see if the next
     + point is at the right distance from the previous one.
     + Furthermore, we take the mean of all the generated points
     + to see if it is close to the origin
     + (check ifuniform on the hypersphere)
     +/
    auto mean_vector = new Vector!real(dimension, 0.0);
    size_t num_vector = 1000;
    foreach(i; 0 .. num_vector) {
        random_search(origin_tmp, &func_obj, 1, 10, 0.5, 1e-3, false);
        assert(abs(origin_tmp.norm!"L2" - 1.0) <= 1e-3,
               "Error: Random_search: Wrong distance between consecutives point");
        mean_vector += origin_tmp;
        origin_tmp.v[] = origin.v[];
    }

    mean_vector /= num_vector;
    assert(mean_vector.norm!"L2" <= 1e-1,
           "Error: Random_search: Generation isn't uniform on the hypersphere.");

}
/*
void random_search_tests() {
    {// Three-hump camel function -  dimensions
        auto v = new Vector!real(2);
        
        real f3hc(in Vector!real _v) {
            real x = _v[0];
            real y = _v[1];
            return 1 - cos(x) * cos(y) * exp(- (x*x + y*y));
        }

        size_t num = 100;
        real succes = 0;

        foreach(i; 0 .. num)
        {
            auto res = random_search!real(v, &f3hc, 100_000_000, 100000);

            if ((abs(res) <= 0.01) && (v.norm!"L2" <= 0.01))
                succes++;
        }
        assert((succes/num) >= 0.95, "Only " ~ to!string(100*(succes/num)) ~ "% success.");
    }

    {// Sphere function - 100 dimensions
        auto v = new Vector!real(100);

        real sphere(in Vector!real _v) {
            real tmp = 0.0;
            foreach(val; _v.v)
                tmp += val*val;
            return tmp;
        }

        assert(sphere(new Vector!real([0.0])) == 0.0);
        assert(sphere(new Vector!real([1.0, -1.0])) == 2.0);

        size_t num = 25;
        real succes = 0;

        foreach(i; 0 .. num)
        {
            auto res = random_search!real(v, &sphere, 5000, 30);

            if ((abs(res) <= 0.01) && (v.norm!"max" <= 0.01))
                succes++;
        }
        assert((succes/num) > 0.94);
    }

    { // Really simple exemple: find the racine of a polynomial (phi !)
        auto v = new Vector!real(1);

        // racine!(a, b, c)(v) is zero iif v[i] is a racine of ax^2 + bx + c
        // for every "i" and > zero everywhere else.
        real racine(real a=1.0, real b=0.0, real c=0.0)(in Vector!real _v) {
            real tmp = 0.0;
            foreach(val; _v.v)
                tmp += pow(a*val*val + b*val + c, 2.0);
            return tmp;
        } 

        assert(racine(new Vector!real([0.0])) == 0.0);
        assert(racine!(1.0, -1.0)(new Vector!real([1.0])) == 0.0);

        auto res = random_search!real(v, &(racine!(1.0, -1.0, -1.0)),
                                      200UL, 5UL);

        assert(abs(res) <= 0.001);
        assert((abs(v[0] - (1.0 + sqrt(5.0))/2.0) <= 0.001) ||
               (abs(v[0] - (1.0 - sqrt(5.0))/2.0) <= 0.001));
    }

    { // train a very small neural network on linear + relu function
        size_t len = 10;
        size_t num_points = 500;
        
        // Create Data points.
        auto true_nn = new NeuralNetwork!float(len);
        true_nn.linear
               .relu
               .serialize;

        Vector!float[] x_train = new Vector!float[num_points];
        Vector!float[] y_train = new Vector!float[num_points];
        Vector!float[] y_tilde = new Vector!float[num_points];

        foreach(i; 0 .. num_points) {
            x_train[i] = new Vector!float(len, 1.0);
            y_train[i] = new Vector!float(len);
            y_tilde[i] = new Vector!float(len);
            true_nn.apply(x_train[i], y_train[i]);
        }

        // Create Neural Network.
        auto nn = new NeuralNetwork!float(len);
        nn.linear
          .relu
          .serialize;

        float loss_function_linRel(in Vector!float _v) {
            float loss_value = 0.0;

            // We equipe the neural network with the weigth given in parameters.
            nn.set_parameters(_v);

            // We loop over all data points and compute the sum of squared errors.
            foreach(i; 0 .. num_points) {
                nn.apply(x_train[i], y_tilde[i]);
                y_tilde[i] -= y_train[i];
                loss_value += y_tilde[i].norm!"L2";
            }

            return loss_value/num_points;
        }

        auto sol = new Vector!float(nn.serialized_data.length, 0.0);
        auto res = random_search!float(sol, &loss_function_linRel, 100_000_000, 500);

        write("Optimizers: Random_search: Linear.Relu: ");
        if (res < 1e-3)
            writeln("OK");
        else
            writeln("FAIL: ", res);
    }

    { // train a very small neural network on dot product function
        size_t len = 100;
        size_t num_points = 1000;
        
        // Create Data points.
        auto m = new Matrix!float(1, len, 1.0);

        Vector!float[] x_train = new Vector!float[num_points];
        Vector!float[] y_train = new Vector!float[num_points];
        Vector!float[] y_tilde = new Vector!float[num_points];

        foreach(i; 0 .. num_points) {
            x_train[i] = new Vector!float(len, 1.0);
            y_tilde[i] = new Vector!float(1, 1.0);
            y_train[i] = m * x_train[i];
        }

        // Create Neural Network.
        auto nn = new NeuralNetwork!float(len);
        nn.linear(1).serialize();

        float loss_function_dot(in Vector!float _v) {
            float loss_value = 0.0;

            // We equipe the neural network with the weigth given in parameters.
            nn.set_parameters(_v);

            // We loop over all data points and compute the sum of squared errors.
            foreach(i; 0 .. num_points) {
                nn.apply(x_train[i], y_tilde[i]);
                y_tilde[i] -= y_train[i];
                loss_value += y_tilde[i].norm!"L2";
            }

            return loss_value/num_points;
        }

        auto sol = new Vector!float(nn.serialized_data.length, 0.0);
        auto res = random_search!float(sol, &loss_function_dot, 1_000_000_000, 200);

        write("Random_search: Dot Product: ");
        if (res < 1e-3)
            writeln("OK");
        else
            writeln("FAIL");
    }
}*/