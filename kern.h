/**
 * # The geisten kern (core) functions - version _0.6-0_
 *
 *
 *               ╚══╗ ║ ╔════╝
 *                  ╠═╩═╩╗
 *                  ║ ▒▒ ║
 *                  ╠═╦═╦╝
 *               ╔══╝ ╚═╚════╗
 *
 *  > By Germar Schlegel - 2021
 *
 * The geisten neural net (_gstnn_) kern module contains functions to run and train deep learning
 * neural networks.
 *
 *
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define KERN_VERSION "0.6-0"

/** ## Macros
 */
/**
 * ### ARRAY_LENGTH - Return the length (number of elements) of the c array.
 *
 * #### Parameters
 *
 *  - `arr` The c array
 */
#define ARRAY_LENGTH(_arr) (sizeof(_arr) / sizeof((_arr)[0]))
#define Derived(_func) _func##_derived

/** ## Functions
 */
/**
 * ### trans()
 *
 * Transform input data array to output array via a weight matrix.
 *
 * #### Parameters
 *
 *  - `batch_len` The number of parallel processed input data.
 *  - `m` The number of input (matrix) rows.
 *  - `n` The number of output (matrix) columns.
 *  - `w` The m x n weight matrix.
 *  - `x` The input vector of length `m`.
 *  - `y` The output (result) vector  of length `n`.
 */
void trans(uint32_t batch_len, uint32_t m, uint32_t n, const float w[m * n],
           const float x[m * batch_len], float y[n * batch_len]);

/**
 * ### train_sgd()
 *
 * Trains the weight matrix by the delta output array.
 *
 * The Stochastic gradient descent (**SGD**) is an iterative
 * method for optimizing an objective function with suitable smoothness
 * properties
 * (See [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)).
 *
 * #### Parameters
 *
 *  - `batch_len` The number of parallel input data.
 *  - `m` The number of input (matrix) rows.
 *  - `n` The number of output (matrix) columns.
 *  - `x` The input vector of length `m`.
 *  - `dy` The delta output (difference between output and expected output) vector  of length `n`.
 *  - `rate` The leaning rate
 *  - `w` The m x n weight matrix.
 */
void train_sgd(uint32_t batch_len, uint32_t m, uint32_t n,
               const float x[m * batch_len], const float dy[n * batch_len],
               float rate, float w[m * n]);

/**
 * ### loss()
 *
 * Calculates `dx` from the output `dy`.
 *
 * #### Parameters
 *
 *  - `batch_len` The number of parallel input data.
 *  - `m` The number of input (matrix) rows.
 *  - `n` The number of output (matrix) columns.
 *  - `w` The m x n weight matrix.
 * - `dy` The delta output (difference between output and expected output)
 * vector  of length `n`.
 * - `dx` The calculated delta input vector  of length `m`.
 */
void loss(uint32_t batch_len, uint32_t m, uint32_t n, const float *w,
          const float *dy, float *dx);

/**
 * ### vec_delta()
 *
 * Calculates the delta between `v1` and `v2`.
 * The result is written into the vector `d`
 *
 * #### Parameters
 *
 *  - `size` The length of the vectors.
 *  - `v1` The first vector.
 *  - `v2` The second vector.
 *  - `d` The calculated delta of v1-v2.
 *  Returns the error of the delta vector (`d`)
 */
double vec_delta(uint32_t size, const float v1[size], const float v2[size],
                 float d[size]);

/**
 * ### vec_is_equal_f32()
 *
 * Check if two vectors are equal within a tolerance range.
 *
 * #### Parameters
 *
 *  - `n` The length of the vectors.
 *  - `a` The first vector.
 *  - `b` The second vector.
 *  - `epsilon` The allowed tolerance.
 *  Returns true if the both vectors are equal.
 */
bool vec_is_equal_f32(uint32_t n, const float a[n], const float b[n],
                      float epsilon);

/**
 * ### weights_norm_init()
 *
 * Initialize a weight matrix with random noise values.
 *
 * #### Parameters
 *
 *  - `m` The rows of the weight matrix.
 *  - `n` The columns of the weight matrix.
 *  - `w` The weight matrix.
 */
void weights_norm_init(uint32_t m, uint32_t n, float w[m * n]);

/** ## Activation functions and its derivatives
 */

/**
 * ### relu()
 *
 * Rectified Linear Units activation function. *
 * The formula is: max(0,z).
 * It’s not linear and provides the same benefits as Sigmoid but with better performance.
 *
 * #### Parameters
 *
 *  - `len` The length of the vector.
 *  - `y` The vector.
 */
void relu(uint32_t len, float y[len]);

/**
 * ### tanhg()
 *
 * Tanh activation function.
 *
 * Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear.
 * But unlike Sigmoid, its output is zero-centered. Therefore, in practice
 * the tanh non-linearity is always preferred to the sigmoid nonlinearity.
 *
 * #### Parameters
 *
 *  - `len` The length of the vector.
 *  - `y` The vector.
 */
void tanhg(uint32_t len, float y[len]);

/**
 * ### tanhg()
 *
 * Sigmoid activation function.
 *
 * Sigmoid takes a real value as input and outputs another value between 0 and 1.
 * It’s non-linear, continuously differentiable, monotonic, and has a fixed output range.
 *
 * #### Parameters
 *
 *  - `len` The length of the vector.
 *  - `y` The vector.
 */
void sigmoid(uint32_t len, float y[len]);

void relu_derived(uint32_t len, const float *result, float *delta);

void tanhg_derived(uint32_t len, const float *result, float *delta);

void sigmoid_derived(uint32_t len, const float *result, float *delta);

/**
 * ### weights_create_or_load()
 *
 * Create or load the matrix fom a memory mapped file or direct from memory.
 *
 * If filename == NULL, the matrix will be allocated from memory. The weights matrix will be lost when closing the program
 * If the filename != NULL, the matrix will be saved and updated in the file
 *
 * #### Parameters
 *
 *  - `filename` The file name to save the weights matrix.
 *  - `input_len` The rows of the weight matrix.
 *  - `input_len` The columns of the weight matrix.
 *
 *  Returns the allocated matrix memory.
 */
float *weights_create_or_load(const char *filename, uint32_t input_len,
                              uint32_t output_len);

/**
 * ### dropout()
 *
 * Set random elements in the vector `vec` to _0_. It is allowed for `vec` and `result` to be identical (the same array).
 *
 * #### Parameters
 *
 *  - `len` The length of the vector.
 *  - `vec` The original, input vector.
 *  - `p` The related probability to set the element to _0_.
 *  - `result` The new vector with some of the elements is set to _0_.
 */
void dropout(uint32_t len, const float vec[len], float p, float result[len]) ;