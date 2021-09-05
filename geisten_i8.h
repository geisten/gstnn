/*

 */

#include <err.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef WITH_THREADS
#include <omp.h>
#define MP_LOOP() PRAGMA(omp for simd)
#else
#define MP_LOOP()
#endif

#define PRAGMA(X) _Pragma(#X)

#define LHS_CACHE 128

#define FX_DIV 64
#define FX_DIVF 64.0
#define FX2FLOAT(_fx) (float)((_fx) / FX_DIVF)
#define FLOAT2FX(_fl) (int)((_fl)*FX_DIVF)

static inline int8_t mult(int8_t multiplier, int8_t multiplicand) {
    int r = (int)multiplier * (int)multiplicand;
    return (int8_t)((r >> 6) +
                    ((r >> 5) & 1));  //Q number format rounding operation
}

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
/*
 * Transform the input vector to the output stream.
 * The m x n matrix is stored with column arrays.
 *
 * y[k * n + j] = x[k * m + i] * w[m * j + i]
 * y(l, n) = x(l,m) * w(m, n)
 * (https://en.wikipedia.org/wiki/Q_%28number_format%29#Math_operations)
 */
static void linear(uint32_t l, uint32_t m, uint32_t n, const int8_t w[m * n],
                   const int8_t x[l * m], int8_t y[l * n]) {
    uint32_t i, j, k, c, d;
    const int8_t *restrict wr;
    int8_t *restrict yr;
    const int8_t *restrict xr;
    int16_t acc[LHS_CACHE];

    for (k = 0; k < l; k++) {
        for (j = 0, yr = &y[k * n], xr = &x[k * m], d = MIN(LHS_CACHE, n);
             j < n; j += d, d = MIN(LHS_CACHE, n - j)) {
            for (c = 0; c < d; c++) {
                for (i = 0, acc[c] = 0, wr = &w[m * (c + j)]; i < m; i++) {
                    acc[c] += mult(wr[i], xr[i]);
                }

               // fprintf(stderr,"%d,%d, %d\n",i, j+c, acc[c] >> 1);
                yr[c + j] = (int8_t)acc[c] >> 1;
            }
        }
    }
}
#pragma clang diagnostic pop

/*
 * Original:
 * w(m,n) -= N * x(l,m)^T * dy(l,n)
 * w[m * $j + i] -= N * x[m * $k + i] * dy[n * k + $j]
 *
 * Optimized:
 * w(m,n)^T -= N * dy(l,n)^T * x(l,m)
 * w[m * j + $i] -= N * dy[n * $k + j] * x[m * k + $i]
 */
static void train_sgd(uint32_t batch_len, uint32_t m, uint32_t n,
                      const int8_t x[m * batch_len],
                      const int8_t dy[n * batch_len], int8_t rate,
                      int8_t w[m * n]) {
    uint32_t i, k;
    int8_t *restrict wr;
    const int8_t *restrict xr;
    int8_t a;

    for (uint32_t j = 0; j < n; j++) {
        for (k = 0, wr = &w[m * j]; k < batch_len; k++) {
            for (i = 0, xr = &x[k * m], a = (int8_t)mult(dy[k * n + j], rate);
                 i < m; i++) {
                wr[i] -= mult(a, xr[i]) ;
            }
        }
    }
}

/*
 * Original:
 * dx(l,m) = dy(l,n) * w(m,n)^T
 * dx[m * k + $i] = dy[n * k + $j] * w[m * $j + i]
 */
static void loss(uint32_t l, uint32_t m, uint32_t n, const int8_t w[m * n],
                 const int8_t dy[l * n], int8_t dx[l * m]) {
    const int8_t *restrict dyr;
    int8_t *restrict dxr;
    uint32_t i, j, k, d, c;
    int acc[LHS_CACHE];
    for (k = 0; k < l; k++) {
        for (i = 0, dyr = &dy[k * n], d = MIN(LHS_CACHE, m); i < m;
             i += d, d = MIN(LHS_CACHE, m - i)) {
            for (c = 0, dxr = &dx[k * m + i]; c < d; c++) {
                MP_LOOP()
                for (j = 0, acc[c] = 0; j < n; ++j) {
                    acc[c] += mult(dyr[j], w[m * j + i + c]);
                }
                dxr[c] = (int8_t)acc[c] << 1;
            }
        }
    }
}

/*
 * Calculate the difference between two vectors of size 'size'.
 *
 * size:   The size of the vectors.
 * vec1:   The first vector
 * vec2:   The second vector
 *
 * Return the mean difference
 */
double vec_delta(uint32_t size, const int8_t vec1[size],
                 const int8_t vec2[size], int8_t deltas[size]) {
    double error = 0;
    for (size_t i = 0; i < size; i++) {
        deltas[i] = vec1[i] - vec2[i];
        error += powf(FX2FLOAT(deltas[i]), 2.0f);
    }
    return error / (double)size;
}

static bool vec_is_equal(uint32_t n, const int8_t a[n], const int8_t b[n],
                         int8_t epsilon) {
    for (uint32_t i = 0; i < n; i++) {
        if (abs(a[i] - b[i]) >= epsilon) return false;
    }
    return true;
}

/*
 * In each iteration of the while loop two normal random variables are
 * generated. On the first call of the function, two normal random variables are
 * generated.
 * On the after call, the second generated number will be returned.
 */
static float rand_normal(float mu, float sigma) {
    float U1, U2, W, mult;
    static float X1, X2;
    static bool call = false;

    if (call == true) {
        call = !call;
        return mu + sigma * X2;
    }

    do {
        U1 = -1 + ((float)random() / (float)(RAND_MAX)) * 2;
        U2 = -1 + ((float)random() / (float)(RAND_MAX)) * 2;
        W  = powf(U1, 2) + powf(U2, 2);
    } while (W >= 1 || W == 0);

    mult = sqrtf((-2 * logf(W)) / W);
    X1   = U1 * mult;
    X2   = U2 * mult;

    call = !call;
    return (mu + sigma * X1);
}

static uint32_t argmax(uint32_t len, const int8_t x[len], int8_t *max) {
    *max             = x[0];
    uint32_t max_pos = 0;
    for (uint32_t i = 0; i < len; i++) {
        if (x[i] > *max) {
            *max    = x[i];
            max_pos = i;
        }
    }
    return max_pos;
}

static void softmax(uint32_t len, const int8_t x[len], int8_t xs[len]) {
    int8_t max;
    argmax(len, x, &max);
    float sum = .0f;
    for (uint32_t i = 0; i < len; i++) {
        float xv = expf((float)(x[i] - max) / FX_DIVF);
        xs[i]    = (int8_t)(xv * FX_DIV);
        sum += xv;
    }
    for (uint32_t i = 0; i < len; i++) {
        xs[i] = (int8_t)((float)xs[i] / sum);
    }
}

// List of activation functions
// Convention: derived activation functions are prefixed with label 'derived'
/*
 * The sigmoid function
 */
static int8_t sigmoid(int8_t x) {
    return (int8_t)(FX_DIVF / (1.0 + expf(FX2FLOAT(-x))));
}

/*
 * The derived of the sigmoid function
 */
static int8_t derived_sigmoid(int8_t y) {
    return mult(y, (int8_t)(FX_DIV - y));
}

static int8_t relu(int8_t x) { return (int8_t)((x > 0) * x); }

static int8_t derived_relu(int8_t x) { return (int8_t)(x > 0); }

static int8_t gtanh(int8_t x) { return FX_DIV * tanhf(x / FX_DIVF); }

static int8_t derived_gtanh(int8_t x) {
    return (int8_t)(FX_DIV * (1.0 - pow(gtanh(x) / FX_DIVF, 2.0)));
}

static void vec_func(uint32_t len, int8_t result[len], int8_t (*f)(int8_t)) {
    for (uint32_t d = 0; d < len; d++) {
        result[d] = f(result[d]);
    }
}

static void sigmoid_vec(uint32_t len, int8_t result[len]) {
    vec_func(len, result, sigmoid);
}

static void relu_vec(uint32_t len, int8_t result[len]) {
    vec_func(len, result, relu);
}

static void tanh_vec(uint32_t len, int8_t result[len]) {
    vec_func(len, result, gtanh);
}

static void vec_derived(uint32_t len, const int8_t result[len],
                        int8_t delta[len], int8_t (*f)(int8_t)) {
    for (uint32_t d = 0; d < len; d++) {
        delta[d] = mult(f(result[d]), delta[d]);
    }
}

static void relu_derived(uint32_t len, const int8_t result[len],
                         int8_t delta[len]) {
    vec_derived(len, result, delta, derived_relu);
}

static void tanh_derived(uint32_t len, const int8_t result[len],
                         int8_t delta[len]) {
    vec_derived(len, result, delta, derived_gtanh);
}

static void sigmoid_derived(uint32_t len, const int8_t result[len],
                            int8_t delta[len]) {
    vec_derived(len, result, delta, derived_sigmoid);
}

static void weights_init(uint32_t m, uint32_t n, int8_t matrix[m * n]) {
    for (uint32_t i = 0; i < m * n; i++) {
        matrix[i] = 0;
    }
}

static void weights_norm_init(uint32_t in_size, uint32_t out_size,
                              int8_t weights[in_size * out_size]) {
    srandom(time(NULL));
    for (unsigned long long i = 0; i < (in_size * out_size); i++) {
        weights[i] = FLOAT2FX(rand_normal(0.0f, sqrt(2.0 / in_size)));
    }
}

static void image_weights_init(uint32_t channels, uint32_t m, uint32_t n,
                               int8_t image[channels * m * n]) {
    for (uint32_t i = 0; i < channels; i++) {
        weights_init(m, n, &image[i * m * n]);
    }
}

static inline int bernoulli(float p /* must between [0,1] */) {
    return (p < (((float)random() / (float)(RAND_MAX))));
}

static void dropout(uint32_t len, const int8_t vec[len], float p,
                    int8_t result[len]) {
    for (uint32_t i = 0; i < len; i++) {
        result[i] = (int8_t)(vec[i] * bernoulli(p));
    }
}

static void filter_init(uint32_t len, int8_t filter[len * len]) {
    weights_norm_init(len, len, filter);
}

/**
 * prune() - cutting off the unnecessary connections in a weights matrix
 * Pruning means cutting off the unnecessary connections in the neural network to reduce its size.
 * The major issue in the pruning process is to identify the "unnecessary".
 * Pruning can be executed before, during, and after training.
 *
 * The pruning operation implemented here hides weights below a certain threshold.
 * For example, if a weight lies in an interval [-0.5, 0.5] (this corresponds to d = 0.5), it will be hidden.
 *
 */
static void prune(uint32_t l, uint32_t m, uint32_t n,
                  int8_t w[restrict l * m * n], int8_t d) {
    for (uint32_t i = 0; i < l * m * n; i++) {
        w[i] *=
            d < abs(w[i]);  // NOLINT(cppcoreguidelines-narrowing-conversions)
    }
}

//static void quantize(uint32_t l, uint32_t m, uint32_t n,
//                     const int8_t w[restrict l * m * n], uint32_t centroids_len,
//                     int8_t centroids[centroids_len],
//                     int8_t w_quantized[restrict l * m * n]) {}
//
//static void quantization_train_sgd(uint32_t l, uint32_t m, uint32_t n,
//                                   const int8_t w[restrict l * m * n],
//                                   uint32_t centroids_len,
//                                   int8_t centroids[centroids_len],
//                                   int8_t w_quantized[restrict l * m * n]) {}

static void conv(
    uint32_t channels, uint32_t m, uint32_t n,
    const float image[channels * m * n], uint32_t filter_size,
    uint32_t filter_range,
    const int8_t filter[filter_size][channels * filter_range * filter_range],
    uint32_t stride,
    int8_t feature_maps[channels * ((m - filter_range) / stride + 1) *
                        ((n - filter_range) / stride + 1)]) {
    image_weights_init(channels, (m - filter_range + 1), (n - filter_range + 1),
                       feature_maps);
    for (uint32_t f = 0; f < filter_size; f++) {
        for (uint32_t channel = 0; channel < channels; channel++) {
            for (uint32_t y = 0, out_y = 0; y + filter_range <= n;
                 y += stride, out_y += filter_range) {
                for (uint32_t filter_y = 0; filter_y < filter_range;
                     filter_y++) {
                    for (uint32_t x = 0, out_x = 0; x + filter_range <= m;
                         x += stride, out_x += filter_range) {
                        for (uint32_t filter_x = 0; filter_x < filter_range;
                             filter_x++) {
                            feature_maps[(out_y + filter_y) *
                                             ((m - filter_range) / stride + 1) *
                                             channel +
                                         out_x + filter_range] +=
                                filter[f][filter_y * filter_range * channel +
                                          filter_x] *
                                image[(y + filter_y) * m * channel + x +
                                      filter_x];
                        }
                    }
                }
            }
        }
    }
}

static void pool(uint32_t batch_size, uint32_t channels, uint32_t m, uint32_t n,
                 const int8_t image[batch_size * channels * m * n],
                 uint32_t pool_range,
                 int8_t feature_maps[batch_size * channels * (m / pool_range) *
                                     (n / pool_range)]) {
    for (uint32_t f = 0; f < batch_size; f++) {
        for (uint32_t channel = 0; channel < channels; channel++) {
            for (uint32_t y = 0, out_y = 0; y + pool_range <= n; y++, out_y++) {
                for (uint32_t x = 0, out_x = 0; x + pool_range <= m;
                     x++, out_x++) {
                    float max = 0.0f;
                    for (uint32_t filter_y = 0; filter_y < pool_range;
                         filter_y++, out_y += pool_range) {
                        for (uint32_t filter_x = 0; filter_x < pool_range;
                             filter_x++) {
                            if (max < image[(y + filter_y) * m * channel + x +
                                            filter_x]) {
                                max = image[(y + filter_y) * m * channel + x +
                                            filter_x];
                            }
                        }
                    }
                    feature_maps[out_y * (m / pool_range) * channel + out_x] =
                        max;
                }
            }
        }
    }
}
