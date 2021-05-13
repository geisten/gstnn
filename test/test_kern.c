//
// Created by germar on 08.04.21.
//

#include "../kern.c"
#include "test.h"

TEST_INIT();

static void test_trans() {
    float x[]          = {0.4f, 0.8f, 0.1f, 0.66f, 0.2f};
    float y_expected[] = {0.7639f, -0.582f, 0.102f, -0.582f,
                          0.072f,  -0.582f, -0.582f};
    float y[7];
    float w[] = {
        0.3f, 0.4f,  0.199f, 0.4f,  0.2f,   //
        0.1f, -0.8f, 0.5f,   -0.2f, 0.5f,   //
        0.5f, -0.1f, -0.5f,  0.2f,  -0.5f,  //
        0.1f, -0.8f, 0.5f,   -0.2f, 0.5f,   //
        0.6f, 0.3f,  1.0f,   -0.8f, 0.1f,   //
        0.1f, -0.8f, 0.5f,   -0.2f, 0.5f,   //
        0.1f, -0.8f, 0.5f,   -0.2f, 0.5f,   //
    };
    const int M = ARRAY_LENGTH(x);
    const int N = ARRAY_LENGTH(y);
    trans(1, M, N, w, x, y);
    vec_write_f32(stdout, N, y, "calculated result");
    test(vec_is_equal_f32(N, y_expected, y, 0.001) && "Calculate y = x * w");
}

static void test_train_sgd() {
    float x[] = {0.4f, 0.8f, 0.1f, 0.66f, 0.2f};
    float y[] = {0.7639f, -0.582f, 0.102f, -0.582f, 0.072f, -0.582f, -0.582f};
    float w[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
    };

    float w_expected[] = {
        -0.30556f, -0.61112f, -0.07639f, -0.504174f, -0.15278f,  //
        0.2328f,   0.4656f,   0.0582f,   0.38412f,   0.1164f,    //
        -0.0408f,  -0.0816f,  -0.0102f,  -0.06732f,  -0.0204f,   //
        0.2328,    0.4656f,   0.0582f,   0.38412f,   0.1164f,    //
        -0.0288f,  -0.0576f,  -0.0072f,  -0.04752f,  -0.0144f,   //
        0.2328f,   0.4656f,   0.0582f,   0.38412f,   0.1164f,    //
        0.2328f,   0.4656f,   0.0582f,   0.38412f,   0.1164f,    //
    };

    const int M = ARRAY_LENGTH(x);
    const int N = ARRAY_LENGTH(y);
    train_sgd(1, M, N, x, y, 1.0, w);
    vec_write_f32(stdout, N * M, w, "calculated result");
    test(vec_is_equal_f32(N * M, w_expected, w, 0.001) &&
         "Calculate w -= N * x^T * y");
}

/*
 * TODO check mom and veloc result with expected values
 */
static void test_train_adam() {
    float x[] = {0.4f, 0.8f, 0.1f, 0.66f, 0.2f};
    float y[] = {0.7639f, -0.582f, 0.102f, -0.582f, 0.072f, -0.582f, -0.582f};
    float w[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
    };

    float w_expected[] = {
        -1.00000, -1.00000, -1.00000, -1.00000, -1.00000, 1.00000,  1.00000,
        1.00000,  1.00000,  1.00000,  -1.00000, -1.00000, -0.99995, -1.00000,
        -0.99999, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -0.99999,
        -1.00000, -0.99990, -1.00000, -0.99998, 1.00000,  1.00000,  1.00000,
        1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    };

    float mom[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
    };

    float vel[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  //
    };

    const int M = ARRAY_LENGTH(x);
    const int N = ARRAY_LENGTH(y);
    train_adam(1, M, N, x, y, 1.0f, 1.0f, 0.9f, .99f, 1e-8f, w, mom, vel);
    vec_write_f32(stdout, N * M, w, "calculated result");
    test(vec_is_equal_f32(N * M, w_expected, w, 0.001) &&
         "Calculate adam optimization");
}

static void test_weight_delta() {
    float x[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.2f};
    float y[] = {0.7639f, -0.582f, 0.102f, -0.582f, 0.072f, -0.582f, -0.582f};
    float w[] = {
        -0.30556f, -0.61112f, -0.07639f, -0.504174f, -0.15278f,  //
        0.2328f,   0.4656f,   0.0582f,   0.38412f,   0.1164f,    //
        -0.0408f,  -0.0816f,  -0.0102f,  -0.06732f,  -0.0204f,   //
        0.2328f,   0.4656f,   0.0582f,   0.38412f,   0.1164f,    //
        -0.0288f,  -0.0576f,  -0.0072f,  -0.04752f,  -0.0144f,   //
        0.2328f,   0.4656f,   0.0582f,   0.38412f,   0.1164f,    //
        0.2328f,   0.4656f,   0.0582f,   0.38412f,   0.1164f,    //
    };

    float x_expected[] = {-0.781610884f, -1.563221768f, -0.195402721f,
                          -1.2896579586f, -0.390805442f};

    const int M = ARRAY_LENGTH(x);
    const int N = ARRAY_LENGTH(y);
    loss(1, M, N, w, y, x);
    vec_write_f32(stdout, M, x, "calculated result");
    test(vec_is_equal_f32(M, x_expected, x, 0.001) &&
         "Calculate dx = dy * w^T");
}

static void test_dropout() {
    float y[] = {0.7639f, -0.582f, 0.102f, -0.582f, 0.072f, -0.582f, -0.582f};
    float d[ARRAY_LENGTH(y)];
    dropout(ARRAY_LENGTH(y), y, 0.5, d);
    uint32_t counter = 0;
    for (uint32_t i = 0; i < ARRAY_LENGTH(y); i++) {
        if (d[i] == 0) {
            ++counter;
        } else if (d[i] != y[i]) {
            test(false && "Result of dropout should be either 0 or y[i]");
        }
    }
    vec_write_f32(stdout, ARRAY_LENGTH(y), d, "vector with drop out values");
}

static void test_argmax() {
    float y[] = {0.7639f, -0.582f, 0.102f, -0.582f, 0.072f, -0.582f, -0.582f};
    float max;
    uint32_t max_pos = argmax(ARRAY_LENGTH(y), y, &max);
    test(max == 0.7639f && "Argmax should return the max element value");
    test(max_pos == 0 &&
         "Argmax should return the position of the max element");

    float y2[] = {-0.7639f, -0.582f, 0.102f, -0.582f, 0.072f, 0.582f, 0.582f};
    max_pos    = argmax(ARRAY_LENGTH(y2), y2, &max);
    test(max == 0.582f &&
         "Argmax should return the first max element value (2)");
    test(max_pos == 5 &&
         "Argmax should return the position of the max element (2)");
}

static void test_softmax() {
    float x[] = {0.7639f, -0.582f, 0.102f, -0.582f, 0.072f, -0.582f, -0.582f};
    float xs[ARRAY_LENGTH(x)];
    softmax(ARRAY_LENGTH(x), x, xs);
    float sum = 0.0f;
    for (uint32_t i = 0; i < ARRAY_LENGTH(x); i++) {
        sum += xs[i];
    }
    printf("sum of softmax vector: %f", sum);
    test(sum >= 0.999f && sum <= 1.0f &&
         "The sum of the softmax vector elements should be equal 1");
}

int main() {
    srandom(time(NULL));
    test_trans();
    test_train_sgd();
    test_weight_delta();
    test_dropout();
    test_train_adam();
    test_argmax();
    test_softmax();
    return TEST_RESULT;
}
