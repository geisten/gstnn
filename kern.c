/*

 */

#include "kern.h"

#include <err.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef WITH_THREADS
#include <omp.h>
#define MP_LOOP() PRAGMA(omp for simd)
#else
#define MP_LOOP()
#endif

#define PRAGMA(X) _Pragma(#X)

/*
 * Transform the input vector to the output stream.
 * The m x n matrix is stored with column arrays.
 *
 * y[k * n + j] = x[k * m + i] * w[m * j + i]
 * y(l, n) = x(l,m) * w(m, n)
 *
 */
void trans(uint32_t batch_len, uint32_t m, uint32_t n, const float *w,
           const float *x, float *y) {
    uint32_t i, j, k;
    const float *restrict wr;
    float *restrict yr;
    const float *restrict xr;

    for (k = 0; k < batch_len; k++) {
        for (j = 0, yr = &y[k * n], xr = &x[k * m]; j < n; j++) {
            for (i = 0, wr = &w[m * j], yr[j] = 0; i < m; i++) {
                yr[j] += wr[i] * xr[i];
            }
        }
    }
}

/*
 * Original:
 * w(m,n) -= N * x(l,m)^T * dy(l,n)
 * w[m * $j + i] -= N * x[m * $k + i] * dy[n * k + $j]
 *
 * Optimized:
 * w(m,n)^T -= N * dy(l,n)^T * x(l,m)
 * w[m * j + $i] -= N * dy[n * $k + j] * x[m * k + $i]
 */
void train_sgd(uint32_t batch_len, uint32_t m, uint32_t n, const float *vec,
               const float *delta,
                   float rate, float *mtrx) {
    uint32_t i, k;
    float *restrict wr;
    const float *restrict xr;
    float a;

    for (uint32_t j = 0; j < n; j++) {
        for (k = 0, wr = &mtrx[m * j]; k < batch_len; k++) {
            for (i = 0, xr = &vec[k * m], a = rate * delta[k * n + j]; i < m; i++) {
                wr[i] -= a * xr[i];
            }
        }
    }
}

/*
 * Original:
 * dx(l,m) = dy(l,n) * w(m,n)^T
 * dx[m * k + $i] = dy[n * k + $j] * w[m * $j + i]
 */
void loss(uint32_t batch_len, uint32_t m, uint32_t n, const float *w,
          const float *dy, float *dx) {
    const float *restrict dyr;
    float *restrict dxr;
    uint32_t i, j, k;
    for (k = 0; k < batch_len; k++) {
        for (i = 0, dxr = &dx[k * m], dyr = &dy[k * n]; i < m; i++) {
            MP_LOOP()
            for (j = 0, dxr[i] = 0.0f; j < n; ++j) {
                dxr[i] += dyr[j] * w[m * j + i];
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
double vec_delta(uint32_t size, const float *vec1, const float *vec2,
                 float *deltas) {
    double error = 0.0;
    for (size_t i = 0; i < size; i++) {
        deltas[i] = vec1[i] - vec2[i];
        error += pow(deltas[i], 2.0);
    }
    return error / (float)size;
}

bool vec_is_equal_f32(uint32_t n, const float a[n], const float b[n],
                      float epsilon) {
    for (uint32_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) >= epsilon) return false;
        // better:
        // if(fabs(a[ii]-b[ii]) < 1e-10 * (fabs(a[ii]) + fabs(b[ii]))) {
        // with the appropriate tolerance
    }
    return true;
}

/*
 * In each iteration of the while loop two normal random variables are
 * generated. On the first call of the function, two normal random variables are
 * generated.
 * On the after call, the second generated number will be returned.
 */
float rand_normal(float mu, float sigma) {
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

void weights_norm_init(uint32_t in_size, uint32_t out_size,
                           float *weights) {
    srandom(time(NULL));
    for (unsigned long long i = 0; i < (in_size * out_size); i++) {
        weights[i] = rand_normal(0.0f, sqrtf(2.0f / in_size));
    }
}

void softmax_f32(uint32_t len, float vec[len]) {
    float max = FLT_MIN;
    for (uint32_t i = 0; i < len; i++) {
        if (vec[i] > max) {
            max = vec[i];
        }
    }
    float sum = .0f;
    for (uint32_t i = 0; i < len; i++) {
        vec[i] = expf(vec[i] - max);
        sum += vec[i];
    }
    for (uint32_t i = 0; i < len; i++) {
        vec[i] = vec[i] / sum;
    }
}

void weights_softmax_f32(uint32_t m, uint32_t n, float vec[m * n]) {
    for (uint32_t i = 0; i < n; i++) {
        softmax_f32(m, &vec[i * m]);
    }
}

// List of activation functions
// Convention: derived activation functions are prefixed with label 'derived'
/*
 * The sigmoid function
 */
float fsigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/*
 * The derived of the sigmoid function
 */
float derived_fsigmoid(float y) { return (y * (1.0f - y)); }

float frelu(float x) { return (float)(x > .0) * x; }

float derived_frelu(float x) { return (float)(x > .0); }

float ftanh(float x) { return tanhf(x); }

float derived_ftanh(float x) { return 1.0f - powf(ftanh(x), 2.0f); }

static void vec_func_f32(uint32_t len, float result[len], float (*f)(float)) {
    for (uint32_t d = 0; d < len; d++) {
        result[d] = f(result[d]);
    }
}

void sigmoid(uint32_t len, float *result) {
    vec_func_f32(len, result, fsigmoid);
}

void relu(uint32_t len, float *result) { vec_func_f32(len, result, frelu); }

void tanhg(uint32_t len, float *result) { vec_func_f32(len, result, ftanh); }

static void vec_derived_f32(uint32_t len, const float result[len],
                            float delta[len], float (*f)(float)) {
    for (uint32_t d = 0; d < len; d++) {
        delta[d] = f(result[d]) * delta[d];
    }
}

void relu_derived(uint32_t len, const float *result, float *delta) {
    vec_derived_f32(len, result, delta, derived_frelu);
}

void tanhg_derived(uint32_t len, const float *result, float *delta) {
    vec_derived_f32(len, result, delta, derived_ftanh);
}

void sigmoid_derived(uint32_t len, const float *result, float *delta) {
    vec_derived_f32(len, result, delta, derived_fsigmoid);
}

float *weights_create_or_load(const char *filename, uint32_t input_len,
                    uint32_t output_len) {
    float *weight;
    if (filename == NULL) {
        weight = reallocarray(NULL, input_len * output_len, sizeof(float));
        weights_norm_init(input_len, output_len, weight);
    } else {
        struct stat statbuf;
        size_t size_expected = input_len * output_len * sizeof(float);

        int fd = open(filename, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            /* failure */
            err(EXIT_FAILURE, "open file '%s'", filename);
        }
        /* find size of input file */
        if (fstat(fd, &statbuf) < 0) {
            err(EXIT_FAILURE, "fstat error");
        } else if (statbuf.st_size == 0) {
            // file does not exist before - create it with the required size
            if (ftruncate(fd, size_expected)) {
                err(EXIT_FAILURE, "file truncate");
            }
            weight = mmap(0, size_expected, PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            weights_norm_init(input_len, output_len, weight);
            return weight;
        } else if (statbuf.st_size > 0 &&
                   (unsigned long)statbuf.st_size < size_expected) {
            errx(EXIT_FAILURE, "invalid data size. Expected: %lu; given: %ld",
                 size_expected, statbuf.st_size);
        }
        weight = mmap(0, size_expected, PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
    }
    return weight;
}