//
// Created by germar on 01.04.21.
//

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "kern.h"

/**
 * num_type - The type of the neural network cell (weights, input, output, etc.)
 * Don't change this type!
 */
typedef float num_type;

/**
 * `BATCH_LENGTH` - The number (batch) of input vectors read simultaneously
 */
#define BATCH_LENGTH 1

/**
 * INPUT_LENGTH - The length of the input array.
 */
#define INPUT_LENGTH 784

/**
 * OUTPUT_LENGTH - The length of the output array.
 */
#define OUTPUT_LENGTH 10

/**
 * The sgd learn rate
 * The learning rate is a hyperparameter that controls how much to change
 * the model in response to the estimated error each time the model weights
 * are updated.
 */
#define LEARN_RATE 0.05f

/**
 * The length of the hidden layer
 */
#define HIDDEN_LENGTH 280

#define HIDDEN_ACTIVATION relu
#define HIDDEN_ACTIVATION_DERIVED Derived(relu)
#define HIDDEN_WEIGHTS_FILENAME "data/weights_hidden.gstnn"
num_type *hidden_weights;
num_type hidden_output[HIDDEN_LENGTH * BATCH_LENGTH];
num_type hidden_delta[HIDDEN_LENGTH * BATCH_LENGTH];

#define OUTPUT_ACTIVATION sigmoid
#define OUTPUT_ACTIVATION_DERIVED Derived(sigmoid)
#define OUTPUT_WEIGHTS_FILENAME "data/weights_output.gstnn"
num_type *output_weights;
num_type output[OUTPUT_LENGTH * BATCH_LENGTH];
num_type output_delta[OUTPUT_LENGTH * BATCH_LENGTH];

/**
 * `layer_construct` - Construct the neural network layers.
 */
static void layer_construct() {
    if (NULL == (hidden_weights = weights_create_or_load(
                     HIDDEN_WEIGHTS_FILENAME, INPUT_LENGTH, HIDDEN_LENGTH))) {
        err(EXIT_FAILURE, "allocate hidden weights memory");
    }
    if (NULL == (output_weights = weights_create_or_load(
                     OUTPUT_WEIGHTS_FILENAME, HIDDEN_LENGTH, OUTPUT_LENGTH))) {
        err(EXIT_FAILURE, "allocate output weights memory");
    }
}

/**
 * `layer_destruct` - Destruct the created neural network.
 */
static void layer_destruct() {
    if (HIDDEN_WEIGHTS_FILENAME != NULL) {
        munmap(hidden_weights, INPUT_LENGTH * HIDDEN_LENGTH);
    } else {
        free(hidden_weights);
    }
    if (OUTPUT_WEIGHTS_FILENAME != NULL) {
        munmap(output_weights, HIDDEN_LENGTH * OUTPUT_LENGTH);
    } else {
        free(output_weights);
    }
}

/**
 * `predict` - Predict the output based on the given input.
 *
 * - `input`: The input vector
 */
static void predict(const num_type input[INPUT_LENGTH]) {
    trans(BATCH_LENGTH, INPUT_LENGTH, HIDDEN_LENGTH, hidden_weights, input,
          hidden_output);
    HIDDEN_ACTIVATION(BATCH_LENGTH * HIDDEN_LENGTH, hidden_output);
    trans(BATCH_LENGTH, HIDDEN_LENGTH, OUTPUT_LENGTH, output_weights,
          hidden_output, output);
    OUTPUT_ACTIVATION(BATCH_LENGTH * OUTPUT_LENGTH, output);
}

/**
 * `errorl` - Calculates the error between the output and the expected (target) array.
 *
 * - `target`: The target vector
 * - `error`: The calculated error value
 * Returns the number of hits (output == target)
 */
static uint64_t errorl(const num_type target[OUTPUT_LENGTH * BATCH_LENGTH],
                       double *error) {
    error[0] =
        vec_delta(OUTPUT_LENGTH * BATCH_LENGTH, output, target, output_delta);
    uint64_t hits = 0;

    size_t max_pos  = SIZE_MAX;
    float max_value = -INFINITY;
    for (size_t pos = 0; pos < OUTPUT_LENGTH; pos++) {
        if (output[pos] > max_value) {
            max_pos   = pos;
            max_value = output[pos];
        }
        if (target[pos] == 1.0f) {
            fprintf(stderr, "-->%zu <=> %zu/%f - [%f]  ::", pos, max_pos,
                    max_value, output[pos]);
            if (pos == max_pos) {
                ++hits;
            }
            break;
        }
    }
    return hits;
}

/**
 * `train` - Train the weight matrix based on the target vector
 *
 * - `input`: The input vector
 */
static void train(const num_type input[INPUT_LENGTH * BATCH_LENGTH]) {
    loss(BATCH_LENGTH, HIDDEN_LENGTH, OUTPUT_LENGTH, output_weights,
         output_delta, hidden_delta);

    HIDDEN_ACTIVATION_DERIVED(BATCH_LENGTH * HIDDEN_LENGTH, hidden_output,
                              hidden_delta);
    train_sgd(BATCH_LENGTH, INPUT_LENGTH, HIDDEN_LENGTH, input, hidden_delta,
              LEARN_RATE, hidden_weights);
    OUTPUT_ACTIVATION_DERIVED(BATCH_LENGTH * OUTPUT_LENGTH, output,
                              output_delta);
    train_sgd(BATCH_LENGTH, HIDDEN_LENGTH, OUTPUT_LENGTH, hidden_output,
              output_delta, LEARN_RATE, output_weights);
}
