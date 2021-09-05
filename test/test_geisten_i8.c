//
// Created by germar on 31.07.21.
//
#include "../geisten_i8.h"
#include "test.h"

TEST_INIT();

#define ARRAY_LENGTH(_arr) (sizeof(_arr) / sizeof((_arr)[0]))

static inline void print_i8(FILE *fp, int8_t value) {
    fprintf(fp, "%s0.%04d, ", value & 0x80 ? "-" : "",
            (int)(((abs(value) & 0x7F) * 10000) / (1 << 7)));
}

static void vec_write_i8(FILE *fp, uint64_t size, const int8_t arr[static size],
                         const char str[]) {
    fprintf(fp, "%s: [", str);
    for (uint64_t i = 0; i < size; i++) {
        print_i8(fp, arr[i]);
    }
    fprintf(fp, "]\n");
}

static void vec_write_i8n(FILE *fp, uint64_t size,
                          const int8_t arr[static size], const char str[]) {
    fprintf(fp, "%s: [", str);
    for (uint64_t i = 0; i < size; i++) {
        fprintf(fp, "%d, ", (int)arr[i]);
    }
    fprintf(fp, "]\n");
}

static void test_8bit_sigmoid() {
    FILE *file = fopen("test/sigmoid.csv", "w");
    if (!file) {
        err(EXIT_FAILURE, "Unable to create new file");
    }
    for (int8_t i = INT8_MIN; i < INT8_MAX; i++) {
        fprintf(file, "%f, %f, %d, %d\n", FX2FLOAT(i), FX2FLOAT(sigmoid(i)), i,
                sigmoid(i));
    }
    fclose(file);
}

static void test_8bit_derived_sigmoid() {
    FILE *file = fopen("test/derived_sigmoid.csv", "w");
    if (!file) {
        err(EXIT_FAILURE, "Unable to create new file");
    }
    for (int8_t i = INT8_MIN; i < INT8_MAX; i++) {
        fprintf(file, "%f, %f, %d, %d\n", FX2FLOAT(i),
                FX2FLOAT(derived_sigmoid(sigmoid(i))), i,
                derived_sigmoid(sigmoid(i)));
    }
    fclose(file);
}

static void test_8bit_relu() {
    FILE *file = fopen("test/relu.csv", "w");
    if (!file) {
        err(EXIT_FAILURE, "Unable to create new file");
    }
    for (int8_t i = INT8_MIN; i < INT8_MAX; i++) {
        fprintf(file, "%f, %f, %d, %d\n", FX2FLOAT(i), FX2FLOAT(relu(i)), i,
                relu(i));
    }
    fclose(file);
}

static void test_8bit_derived_relu() {
    FILE *file = fopen("test/derived_relu.csv", "w");
    if (!file) {
        err(EXIT_FAILURE, "Unable to create new file");
    }
    for (int8_t i = INT8_MIN; i < INT8_MAX; i++) {
        fprintf(file, "%f, %f, %d, %d\n", FX2FLOAT(i),
                FX2FLOAT(derived_relu(relu(i))), i, derived_relu(relu(i)));
    }
    fclose(file);
}

static void test_8bit() {
    int8_t x[128];
    int8_t x_neg[128];

    const int M = ARRAY_LENGTH(x);

    for (int i = 0; i < M; i++) {
        x[i]     = i;
        x_neg[i] = -i;
    }

    vec_write_i8n(stdout, M, x, "all 8 bit numbers");
    vec_write_i8(stdout, M, x, "all 8 bit numbers");
    vec_write_i8(stdout, M, x_neg, "all negative 8 bit fix point numbers");

    int8_t x_scalar[] = {20};
    int8_t y_scalar[] = {20};
}

static void test_trans() {
    int8_t x[]          = {FLOAT2FX(0.4f), FLOAT2FX(0.8f), FLOAT2FX(0.1f),
                  FLOAT2FX(0.66f), FLOAT2FX(0.2f)};
    int8_t y_expected[] = {FLOAT2FX(0.7639f), FLOAT2FX(-0.582f),
                           FLOAT2FX(0.102f),  FLOAT2FX(-0.582f),
                           FLOAT2FX(0.072f),  FLOAT2FX(-0.582f),
                           FLOAT2FX(-0.582f)};
    int8_t y[7];
    int8_t w[] = {
        FLOAT2FX(0.3f),  FLOAT2FX(0.4f),  FLOAT2FX(0.199f),
        FLOAT2FX(0.4f),  FLOAT2FX(0.2f),  //
        FLOAT2FX(0.1f),  FLOAT2FX(-0.8f), FLOAT2FX(0.5f),
        FLOAT2FX(-0.2f), FLOAT2FX(0.5f),  //
        FLOAT2FX(0.5f),  FLOAT2FX(-0.1f), FLOAT2FX(-0.5f),
        FLOAT2FX(0.2f),  FLOAT2FX(-0.5f),  //
        FLOAT2FX(0.1f),  FLOAT2FX(-0.8f), FLOAT2FX(0.5f),
        FLOAT2FX(-0.2f), FLOAT2FX(0.5f),  //
        FLOAT2FX(0.6f),  FLOAT2FX(0.3f),  FLOAT2FX(1.0f),
        FLOAT2FX(-0.8f), FLOAT2FX(0.1f),  //
        FLOAT2FX(0.1f),  FLOAT2FX(-0.8f), FLOAT2FX(0.5f),
        FLOAT2FX(-0.2f), FLOAT2FX(0.5f),  //
        FLOAT2FX(0.1f),  FLOAT2FX(-0.8f), FLOAT2FX(0.5f),
        FLOAT2FX(-0.2f), FLOAT2FX(0.5f),  //
    };
    const int M = ARRAY_LENGTH(x);
    const int N = ARRAY_LENGTH(y);
    linear(1, M, N, w, x, y);

    vec_write_i8(stdout, N, y_expected, "expected result");
    vec_write_i8(stdout, N, y, "calculated result");
    test(vec_is_equal(N, y_expected, y, 3) && "Calculate y = x * w");

    uint8_t result = mult(20, 80);
    print_i8(stdout, result);
    printf("\n");
    print_i8(stdout, 20);
    printf("\n");
    print_i8(stdout, -20);
    printf("\n");
}

static void test_train_sgd() {
    int8_t x[] = {FLOAT2FX(0.4f), FLOAT2FX(0.8f), FLOAT2FX(0.1f),
                  FLOAT2FX(0.66f), FLOAT2FX(0.2f)};
    int8_t y[] = {97, -74, 13, -74, 9, -74, -74};
    int8_t w[] = {
        0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0,  //
    };

    int8_t w_expected[] = {
        -39, -78, -10, -64, -19,  //
        30,  59,  7,   49,  15,   //
        -5,  -10, -1,  -9,  -3,   //
        30,  59,  7,   49,  15,   //
        -4,  -7,  -1,  -6,  -2,   //
        30,  59,  7,   49,  15,   //
        30,  59,  7,   49,  15,   //
    };

    const int M = ARRAY_LENGTH(x);
    const int N = ARRAY_LENGTH(y);
    vec_write_i8(stdout, N * M, w_expected, "expected result");
    train_sgd(1, M, N, x, y, FX_DIV, w);
    vec_write_i8(stdout, N * M, w, "calculated result");
    test(vec_is_equal(N * M, w_expected, w, 2) && "Calculate w -= N * x^T * y");
}

static void test_weight_delta() {
    int8_t x[] = {0, 0, 0, 0, 25};
    int8_t y[] = {
        (int8_t)(roundf(0.7639f * FX_DIV)), (int8_t)roundf(-0.582f * FX_DIV),
        (int8_t)roundf(0.102f * FX_DIV),    (int8_t)roundf(-0.582f * FX_DIV),
        (int8_t)roundf(0.072f * FX_DIV),    (int8_t)roundf(-0.582f * FX_DIV),
        (int8_t)roundf(-0.582f * FX_DIV)};
    int8_t w[] = {
        (int8_t)roundf(-0.30556f * FX_DIV), (int8_t)roundf(-0.61112f * FX_DIV),
        (int8_t)roundf(-0.07639f * FX_DIV), (int8_t)roundf(-0.504174f * FX_DIV),
        (int8_t)roundf(-0.15278f * FX_DIV),  //
        (int8_t)roundf(0.2328f * FX_DIV),   (int8_t)roundf(0.4656f * FX_DIV),
        (int8_t)roundf(0.0582f * FX_DIV),   (int8_t)roundf(0.38412f * FX_DIV),
        (int8_t)roundf(0.1164f * FX_DIV),  //
        (int8_t)roundf(-0.0408f * FX_DIV),  (int8_t)roundf(-0.0816f * FX_DIV),
        (int8_t)roundf(-0.0102f * FX_DIV),  (int8_t)roundf(-0.06732f * FX_DIV),
        (int8_t)roundf(-0.0204f * FX_DIV),  //
        (int8_t)roundf(0.2328f * FX_DIV),   (int8_t)roundf(0.4656f * FX_DIV),
        (int8_t)roundf(0.0582f * FX_DIV),   (int8_t)roundf(0.38412f * FX_DIV),
        (int8_t)roundf(0.1164f * FX_DIV),  //
        (int8_t)roundf(-0.0288f * FX_DIV),  (int8_t)roundf(-0.0576f * FX_DIV),
        (int8_t)roundf(-0.0072f * FX_DIV),  (int8_t)roundf(-0.04752f * FX_DIV),
        (int8_t)roundf(-0.0144f * FX_DIV),  //
        (int8_t)roundf(0.2328f * FX_DIV),   (int8_t)roundf(0.4656f * FX_DIV),
        (int8_t)roundf(0.0582f * FX_DIV),   (int8_t)roundf(0.38412f * FX_DIV),
        (int8_t)roundf(0.1164f * FX_DIV),  //
        (int8_t)roundf(0.2328f * FX_DIV),   (int8_t)roundf(0.4656f * FX_DIV),
        (int8_t)roundf(0.0582f * FX_DIV),   (int8_t)roundf(0.38412f * FX_DIV),
        (int8_t)roundf(0.1164f * FX_DIV),  //
    };

    int8_t x_expected[] = {(int8_t)roundf(-0.781610884f * FX_DIV),
                           (int8_t)roundf(-1.563221768f * FX_DIV),
                           (int8_t)roundf(-0.195402721f * FX_DIV),
                           (int8_t)roundf(-1.2896579586f * FX_DIV),
                           (int8_t)roundf(-0.390805442f * FX_DIV)};

    const int M = ARRAY_LENGTH(x);
    const int N = ARRAY_LENGTH(y);
    loss(1, M, N, w, y, x);
    vec_write_i8(stdout, M, x_expected, "expected result");
    vec_write_i8(stdout, M, x, "calculated result");
    test(vec_is_equal(M, x_expected, x, 4) && "Calculate dx = dy * w^T");
}

static void test_dropout() {
    int8_t y[] = {(int8_t)roundf(0.7639f * FX_DIV), (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(0.102f * FX_DIV),  (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(0.072f * FX_DIV),  (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(-0.582f * FX_DIV)};
    int8_t d[ARRAY_LENGTH(y)];
    dropout(ARRAY_LENGTH(y), y, 0.5, d);
    uint32_t counter = 0;
    for (uint32_t i = 0; i < ARRAY_LENGTH(y); i++) {
        if (d[i] == 0) {
            ++counter;
        } else if (d[i] != y[i]) {
            test(false && "Result of dropout should be either 0 or y[i]");
        }
    }
    vec_write_i8(stdout, ARRAY_LENGTH(y), y, "original vector");
    vec_write_i8(stdout, ARRAY_LENGTH(y), d, "vector with drop out values");
}

static void test_argmax() {
    int8_t y[] = {(int8_t)roundf(0.7639f * FX_DIV), (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(0.102f * FX_DIV),  (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(0.072f * FX_DIV),  (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(-0.582f * FX_DIV)};
    int8_t max;
    uint32_t max_pos = argmax(ARRAY_LENGTH(y), y, &max);
    test(max == (int8_t)roundf(0.7639f * FX_DIV) &&
         "Argmax should return the max element value");
    test(max_pos == 0 &&
         "Argmax should return the position of the max element");

    int8_t y2[] = {
        (int8_t)roundf(-0.7639f * FX_DIV), (int8_t)roundf(-0.582f * FX_DIV),
        (int8_t)roundf(0.102f * FX_DIV),   (int8_t)roundf(-0.582f * FX_DIV),
        (int8_t)roundf(0.072f * FX_DIV),   (int8_t)roundf(0.582f * FX_DIV),
        (int8_t)roundf(0.582f * FX_DIV)};
    max_pos = argmax(ARRAY_LENGTH(y2), y2, &max);
    test(abs(max - (int8_t)roundf(0.582f * FX_DIV)) < 1 &&
         "Argmax should return the first max element value (2)");
    test(max_pos == 5 &&
         "Argmax should return the position of the max element (5)");
}

static void test_softmax() {
    int8_t x[] = {(int8_t)roundf(0.7639f * FX_DIV), (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(0.102f * FX_DIV),  (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(0.072f * FX_DIV),  (int8_t)roundf(-0.582f * FX_DIV),
                  (int8_t)roundf(-0.582f * FX_DIV)};
    int8_t xs[ARRAY_LENGTH(x)];
    softmax(ARRAY_LENGTH(x), x, xs);
    float sum = 0.0f;
    for (uint32_t i = 0; i < ARRAY_LENGTH(x); i++) {
        sum += xs[i] / (float)INT8_MAX;
    }
    printf("sum of softmax vector: %f\n", sum);
    test(sum >= 0.9605f && sum <= 1.0f &&
         "The sum of the softmax vector elements should be equal 1");
}

int main() {
    srandom(time(NULL));
    test_8bit_sigmoid();
    test_8bit_derived_sigmoid();
    test_8bit_derived_relu();
    test_8bit_relu();
    test_8bit();
    test_trans();
    test_train_sgd();
    //  test_train_adam();
    test_weight_delta();
    test_dropout();
    test_argmax();
    test_softmax();
    return TEST_RESULT;
}