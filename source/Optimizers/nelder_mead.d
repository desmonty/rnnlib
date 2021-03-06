module source.optimizers.nelder_mead;

import std.algorithm;
import std.conv: to;
import std.functional: toDelegate;
import std.math: abs, pow, sqrt, cos;
import std.range: enumerate;
import std.stdio: writeln, write;
import std.typecons: Yes;

import source.Matrix;
import source.NeuralNetwork;
import source.Parameter;


T nelder_mead(T)(ref Vector!T _v, T function(in Vector!T) _obj,
                 T reflection_coef=1.0,
                 T expension_coef=2.0,
                 T contraction_coef=0.5,
                 T shrink_coef=0.5) {
    return nelder_mead!T(_v, toDelegate(_obj),
                         reflection_coef, expension_coef,
                         contraction_coef, shrink_coef);
}

T nelder_mead(T)(ref Vector!T _v, T delegate(in Vector!T) _obj,
                 in T _random_bound=5.0,
                 in size_t _max_iterations=10000,
                 in T _reflection_coef=1.0,
                 in T _expension_coef=2.0,
                 in T _contraction_coef=0.5,
                 in T _shrink_coef=0.5)
{
    /+
     + Arguments:
     +
     +  - _v (Vector!T): will contains a local minimum of the _obj.
     +
     +  - _obj (T delegate(Vector!T)): the function to minimize.
     +
     +  - _random_bound (T): Absolute upper/lower bound for the creation
     +                       of the vectors.
     +
     +  - _max_iterations (size_t): Maximum number of iterations.
     +
     +  - reflection_coef (T): Meta-Parameter.
     +
     +  - expension_coef (T): Meta-Parameter.
     +
     +  - contraction_coef (T): Meta-Parameter.
     +
     +  - shrink_coef (T): Meta-Parameter.
     +/

    assert(_contraction_coef <= 0.5, "Contraction coef must be <= 0.5");
    assert(_contraction_coef > 0, "Contraction coef must be > 0");

    immutable size_t length = _v.length;
    immutable size_t num_points = _v.length + 1;

    /// Create Simplex

    auto simplex = new Vector!T[num_points];
    /+
    foreach(i; 0 .. (num_points))
        simplex[i] = new Vector!T(length, _random_bound);
    +/
    foreach(i; 0 .. (num_points-1)) {
        simplex[i] = new Vector!T(length, 0);
        simplex[i].v[i] = _random_bound;
    }
    simplex[num_points-1] = new Vector!T(length, 0);
    foreach(i; 0 .. length)
        simplex[num_points-1].v[i] = - _random_bound / sqrt(to!real(length));

    /// Compute values at simplex points.
    T[] simplex_values = new T[num_points];
    foreach(i; 0 .. num_points)
        simplex_values[i] = _obj(simplex[i]);

    /// Order simplex
    auto min_value = simplex_values.minElement;
    auto min_index = simplex_values.minIndex;

    auto tmp_max_value = new T[2];
    simplex_values.topNCopy!"a > b"(tmp_max_value, Yes.sortOutput);
    size_t[] index_max = new size_t[2];
    simplex_values.topNIndex!"a > b"(index_max, Yes.sortOutput);
    auto max1_value = tmp_max_value[0];
    auto max1_index = index_max[0];
    auto max2_value = tmp_max_value[1];
    auto max2_index = index_max[1];
            
    /// Init util vector
    auto temporary_vector = new Vector!T(length, _random_bound);
    auto reflection_vector = new Vector!T(length, _random_bound);
    auto expension_vector = new Vector!T(length, _random_bound);
    auto contraction_vector = new Vector!T(length, _random_bound);
    auto shrink_vector = new Vector!T(length, _random_bound);
    T temporary_value = to!T(0);
    T reflection_value = to!T(0);
    T expension_value = to!T(0);
    T contraction_value = to!T(0);
    T shrink_value = to!T(0);

    // Centroid
    auto temporary_centroid_vector = new Vector!T(length, 0);
    auto centroid_vector = new Vector!T(length, 0);

    foreach(i; 0 .. num_points){ // Centroid doesn't take worst point into account
        if (i != max1_index)
            temporary_centroid_vector += simplex[i];
    }

    /// Core Loop
    bool stop_criterion = false;
    bool free_pass = false;
    size_t ind = 0;
    while(!stop_criterion && ind < _max_iterations) {
        /// Compute centroid
        centroid_vector.v[] = temporary_centroid_vector.v[];
        centroid_vector /= length;
        
        free_pass = false;
        
        /// Reflection - 1st case
        reflection_vector.v[] = centroid_vector.v[];
        reflection_vector -= simplex[max1_index];
        reflection_vector *= _reflection_coef;
        reflection_vector += centroid_vector;

        reflection_value = _obj(reflection_vector);
        

        // if f(x_1) <= f(x_r) < f(x_n)
        if ((reflection_value < max2_value)
         && (reflection_value >= min_value)) {
            // Readjust centroid vector.
            temporary_centroid_vector += reflection_vector;

            // Replace last point in simplex.
            simplex[max1_index].v[] = reflection_vector.v[];
            simplex_values[max1_index] = reflection_value;

            min_value = simplex_values.minElement;
            min_index = simplex_values.minIndex;
            simplex_values.topNCopy!"a > b"(tmp_max_value, Yes.sortOutput);
            simplex_values.topNIndex!"a > b"(index_max, Yes.sortOutput);
            max1_value = tmp_max_value[0];
            max1_index = index_max[0];
            max2_value = tmp_max_value[1];
            max2_index = index_max[1];
            free_pass = true;
        }

        /// Expension
        if (!free_pass) {
            if (reflection_value < min_value) {
                expension_vector.v[] = reflection_vector.v[];
                expension_vector -= centroid_vector;
                expension_vector *= _expension_coef;
                expension_vector += centroid_vector;

                expension_value = _obj(expension_vector);

                // We replace the worst point by the expension vector
                if (expension_value < reflection_value) {
                    // Readjust centroid vector.
                    temporary_centroid_vector += expension_vector;

                    simplex[max1_index].v[] = expension_vector.v[];
                    simplex_values[max1_index] = expension_value;

                    min_value = simplex_values.minElement;
                    min_index = simplex_values.minIndex;
                    simplex_values.topNCopy!"a > b"(tmp_max_value, Yes.sortOutput);
                    simplex_values.topNIndex!"a > b"(index_max, Yes.sortOutput);
                    max1_value = tmp_max_value[0];
                    max1_index = index_max[0];
                    max2_value = tmp_max_value[1];
                    max2_index = index_max[1];
                }
                // Reflection - 2nd case
                // We replace the worst point by the reflection vector
                else {
                    // Readjust centroid vector.
                    temporary_centroid_vector += reflection_vector;

                    simplex[max1_index].v[] = reflection_vector.v[];
                    simplex_values[max1_index] = reflection_value;

                    min_value = simplex_values.minElement;
                    min_index = simplex_values.minIndex;
                    simplex_values.topNCopy!"a > b"(tmp_max_value, Yes.sortOutput);
                    simplex_values.topNIndex!"a > b"(index_max, Yes.sortOutput);
                    max1_value = tmp_max_value[0];
                    max1_index = index_max[0];
                    max2_value = tmp_max_value[1];
                    max2_index = index_max[1];
                }
                free_pass = true;
            }
        }

        /// Contraction
        if (!free_pass) {
            // Internal contraction
            auto blie = "";
            if (max1_value <= reflection_value) {
                contraction_vector.v[] = simplex[max1_index].v[];
                contraction_vector -= centroid_vector;
                contraction_vector *= _contraction_coef;
                contraction_vector += centroid_vector;
            }
            // External contraction
            else {
                contraction_vector.v[] = reflection_vector.v[];
                contraction_vector -= centroid_vector;
                contraction_vector *= _contraction_coef;
                contraction_vector += centroid_vector;
            }

            contraction_value = _obj(contraction_vector);
            if (contraction_value < max1_value) {
                // Readjust centroid vector.
                temporary_centroid_vector += contraction_vector;

                // Replace last point in simplex.
                simplex[max1_index].v[] = contraction_vector.v[];
                simplex_values[max1_index] = contraction_value;

                min_value = simplex_values.minElement;
                min_index = simplex_values.minIndex;
                simplex_values.topNCopy!"a > b"(tmp_max_value, Yes.sortOutput);
                simplex_values.topNIndex!"a > b"(index_max, Yes.sortOutput);
                max1_value = tmp_max_value[0];
                max1_index = index_max[0];
                max2_value = tmp_max_value[1];
                max2_index = index_max[1];
                free_pass = true;
            }
        }

        /// Shrink
        if (!free_pass) {
            // We need to recompute the centroid
            temporary_centroid_vector.v[] = simplex[min_index].v[];
            foreach(i; 0 .. num_points) {
                if (i != min_index) {
                    simplex[i] -= simplex[min_index];
                    simplex[i] *= _shrink_coef;
                    simplex[i] += simplex[min_index];
                    simplex_values[i] = _obj(simplex[i]);
                    temporary_centroid_vector += simplex[i];
                }
            }

            min_value = simplex_values.minElement;
            min_index = simplex_values.minIndex;
            simplex_values.topNCopy!"a > b"(tmp_max_value, Yes.sortOutput);
            simplex_values.topNIndex!"a > b"(index_max, Yes.sortOutput);
            max1_value = tmp_max_value[0];
            max1_index = index_max[0];
            max2_value = tmp_max_value[1];
            max2_index = index_max[1];
        }

        // Stopping criterion 1:
        // Stop when the centroid and the first vector are close

        // Remove worst point from centroid
        temporary_centroid_vector -= simplex[max1_index];

        temporary_vector.v[] = temporary_centroid_vector.v[];
        temporary_vector.v[] /= length;
        temporary_vector -= simplex[min_index];
        stop_criterion = (temporary_vector.norm!"L2" <= 0.0001);

        ind++;
    }

    /// Return
    _v.v[] = simplex[min_index].v[];
    return min_value;
}
unittest {
    void test_action_NM(Vector!float expected_solution,
                        string str_action,
                        Vector!float point_training=null)
    {
        if (!point_training)
            point_training = expected_solution;

        // Create Data points.
        Vector!float[] x_train = new Vector!float[1];
        Vector!float[] y_train = new Vector!float[1];
        Vector!float[] y_tilde = new Vector!float[1];

        x_train[0] = point_training;
        y_train[0] = new Vector!float(2);
        y_tilde[0] = new Vector!float(2);
        y_train[0].v[] = x_train[0].v[];

        // Create Neural Network.
        auto nn = new NeuralNetwork!float(2);
        nn.func!("b.v[] = p0.v[];", Vector!float)
                (2, null, null, null, null, [2], [0.0])
          .serialize;

        float loss_function_linRel(in Vector!float _v) {
            float loss_value = 0.0;

            // We equipe the neural network with the weigth given in parameters.
            nn.set_parameters(_v);

            // Compute the squared errors.
            nn.apply(x_train[0], y_tilde[0]);
            y_tilde[0] -= y_train[0];
            loss_value += y_tilde[0].norm!"L2";

            return loss_value;
        }

        auto sol = new Vector!float(nn.serialized_data.length, 0.0);
        auto res = nelder_mead!float(sol, &loss_function_linRel, 1.0, 1);

        sol -= expected_solution;
        assert (sol.norm!"L2" < 1e-3, "Optimizers: Nelder_Mead: " ~ str_action);
    }

    void test_action_NM_shrink(Vector!float expected_solution,
                               string str_action,)
    {
        float loss_function_cosabs(in Vector!float _v) {
            auto v0 = 0.5*(_v[1] - _v[0]);
            auto v1 = 0.5*(1+sqrt(2.0))*(_v[0]+_v[1]-1.0);
            auto z = sqrt(v0*v0 + v1*v1);
            return z - cos(2.0*3.1415926535*z);
        }

        auto sol = new Vector!float(2, 0.0);
        auto res = nelder_mead!float(sol, &loss_function_cosabs, 1.0, 1);

        sol -= expected_solution;
        assert (sol.norm!"L2" < 1e-3, "Optimizers: Nelder_Mead: " ~ str_action);
    }

    auto vec_r1 = new Vector!float([0, 1]);
    auto vec_r2 = new Vector!float([1.0+1.0/sqrt(2.0), 1.0+1.0/sqrt(2.0)]);
    auto vec_e = new Vector!float([1.5+sqrt(2.0), 1.5+sqrt(2.0)]);
    auto vec_c_i = new Vector!float([0.25-1.0/(2.0*sqrt(2.0)),
                                     0.25-1.0/(2.0*sqrt(2.0))]);
    auto vec_c_e = new Vector!float([0.75 + 1.0/(2.0*sqrt(2.0)),
                                     0.75 + 1.0/(2.0*sqrt(2.0))]);
    auto vec_s = new Vector!float([0.5, 0.5]);

    test_action_NM(vec_r1, "Reflection - 1st case", new Vector!float([0.5, 1 + 1/sqrt(2.0)]));
    test_action_NM(vec_r2, "Reflection - 2nd case");
    test_action_NM(vec_e, "Expension");
    test_action_NM(vec_c_i, "Contraction (Internal)", new Vector!float([0.1, 0.1]));
    test_action_NM(vec_c_e, "Contraction (External)", new Vector!float([0.9, 0.9]));
    test_action_NM_shrink(vec_s, "Shrink");
}

void nelder_mead_tests() {
    {// coscosexp function -  dimensions
        auto v = new Vector!real(2);
        
        import std.math: cos, exp;
        
        real f3hc(in Vector!real _v) {
            real x = _v[0];
            real y = _v[1];
            return 1 - cos(x) * cos(y) * exp(- (x*x + y*y));
        }

        size_t num = 25;
        real succes = 0;

        foreach(i; 0 .. num)
        {
            auto res = nelder_mead!real(v, &f3hc, 1.0);

            if ((abs(res) <= 0.01) && (v.norm!"L2" <= 0.01))
                succes++;
        }
        assert((succes/num) >= 0.95,
               "Optimizers: Nelder_Mead: coscosexp: " ~ to!string(100*(succes/num))~"%");
    }

    {// Sphere function
        auto v = new Vector!real(150);

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
            auto res = nelder_mead!real(v, &sphere);

            if ((abs(res) <= 0.01) && (v.norm!"max" <= 0.01))
                succes++;
        }
        assert((succes/num) >= 0.95,
              "Optimizers: Nelder_Mead: Sphere: " ~ to!string(100*(succes/num))~"%");
    }

    {// train a very small neural network on linear + relu function
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
        auto res = nelder_mead!float(sol, &loss_function_linRel, 2.0);

        assert(res < 1e-3, "Optimizers: Nelder_Mead: Linear.Relu");
    }

    {// train a very small neural network on dot product function
        size_t len = 100;
        size_t num_points = 5000;
        
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
        auto res = nelder_mead!float(sol, &loss_function_dot, 2.0, 50000);

        
        assert (res < 1e-3, "Optimizers: Nelder_Mead: Dot Product");
    }
}
