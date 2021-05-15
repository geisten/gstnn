/*

 */

#include "kern.h"

#include <cblas.h>
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
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batch_len, n, m,
                1.0f, x, batch_len, w, m, 0.0f, y, batch_len);
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
void train_sgd(uint32_t batch_len, uint32_t m, uint32_t n,
               const float x[m * batch_len], const float y[n * batch_len],
               float rate, float w[m * n]) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, m, batch_len, -rate,
                y, n, x, m, 1.0f, w, m);
}

/*
 * adam optimizer for the weight matrix
 */
float train_adam(uint32_t batch_len, uint32_t m, uint32_t n,
                 const float x[restrict m * batch_len],
                 const float dy[restrict const n * batch_len], float counter,
                 float N, float beta1, float beta2, float epsilon,
                 float w[m * n], float mom[n * m], float veloc[n * m]) {
    float *restrict wr;
    float *restrict mr;
    float *restrict vr;
    const float *restrict xr;
    float dr;
    uint32_t k, i;
    for (uint32_t j = 0; j < n; j++) {
        for (k = 0, wr = &w[m * j], vr = &veloc[j * m], mr = &mom[j * m];
             k < batch_len; k++) {
            for (i = 0, xr = &x[k * m], dr = dy[k * n + j]; i < m; i++) {
                const float g      = dr * xr[i];
                mr[i]              = beta1 * mr[i] + ((1 - beta1) * g);
                const float mr_hat = mr[i] / (1 - powf(beta1, counter));
                vr[i] = beta2 * vr[i] + ((1 - beta2) * (powf(g, 2)));
                const float vr_hat = vr[i] / (1 - powf(beta2, counter));
                wr[i] -= N * mr_hat / (sqrtf(vr_hat + epsilon));
            }
        }
    }
    return counter + 1.0f;
}

/*
 * Original:
 * dx(l,m) = dy(l,n) * w(m,n)^T
 * dx[m * k + $i] = dy[n * k + $j] * w[m * $j + i]
 */
void loss(uint32_t batch_len, uint32_t m, uint32_t n, const float *w,
          const float *dy, float *dx) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_len, m, n, 1.0f,
                dy, n, w, m, .0f, dx, m);
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

void weights_norm_init(uint32_t in_size, uint32_t out_size, float *weights) {
    srandom(time(NULL));
    for (unsigned long long i = 0; i < (in_size * out_size); i++) {
        weights[i] = rand_normal(0.0f, sqrtf(2.0f / in_size));
    }
}

uint32_t argmax(uint32_t len, const float x[len], float *max) {
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

void softmax(uint32_t len, const float x[len], float xs[len]) {
    float max;
    argmax(len, x, &max);
    float sum = .0f;
    for (uint32_t i = 0; i < len; i++) {
        xs[i] = expf(x[i] - max);
        sum += xs[i];
    }
    for (uint32_t i = 0; i < len; i++) {
        xs[i] = xs[i] / sum;
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

/*
 * Allocate memory for a float matrix.
 */
float *matrix_alloc(uint32_t m, uint32_t n) {
    return reallocarray(NULL, m * n, sizeof(float));
}

void matrix_init(uint32_t m, uint32_t n, float matrix[m * n]) {
    for (uint32_t i = 0; i < m * n; i++) {
        matrix[i] = 0.0f;
    }
}

float *weights_create_or_load(const char *filename, uint32_t input_len,
                              uint32_t output_len) {
    float *weight;
    if (filename == NULL) {
        weight = matrix_alloc(input_len, output_len);
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

static inline float fbernoulli(float p /* must between [0,1] */) {
    return (float)(p < (((float)random() / (float)(RAND_MAX))));
}

void dropout(uint32_t len, const float vec[len], float p, float result[len]) {
    for (uint32_t i = 0; i < len; i++) {
        result[i] = vec[i] * fbernoulli(p);
    }
}
