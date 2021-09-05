//
// Created by germar on 10.08.21.
//
//
// Created by germar on 02.04.21.
//
#include <err.h>
#include <fcntl.h>
#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../geisten_i8.h"
#include "../stats.h"
#include "../stopwatch.h"

#define USAGE_FMT "%s [-t FILE] [-h] [-f]"

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

/**
 * num_type - The type of the neural network cell (weights, input, output, etc.)
 * Don't change this type!
 */
typedef int8_t num_type;
#define NUM_TYPE_MIN (-FX_DIV)
#define NUM_TYPE_MAX FX_DIV

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
//#define LEARN_RATE (int8_t)(0.05f * INT8_MAX)
#define LEARN_RATE FLOAT2FX(0.05)
/**
 * The length of the hidden layer
 */
#define HIDDEN_LENGTH 280

//#define HIDDEN_ACTIVATION relu_vec
//#define HIDDEN_ACTIVATION_DERIVED Derived(relu)
#define HIDDEN_WEIGHTS_FILENAME "data/weights_hidden.gstnn"

num_type *hidden_weights;
num_type hidden_output[HIDDEN_LENGTH * BATCH_LENGTH];
num_type hidden_delta[HIDDEN_LENGTH * BATCH_LENGTH];

//#define OUTPUT_ACTIVATION sigmoid_vec
//#define OUTPUT_ACTIVATION_DERIVED Derived(sigmoid)
#define OUTPUT_WEIGHTS_FILENAME "data/weights_output.gstnn"
num_type *output_weights;
num_type output[OUTPUT_LENGTH * BATCH_LENGTH];
num_type output_delta[OUTPUT_LENGTH * BATCH_LENGTH];

/*
 * Allocate memory for a num_type matrix.
 */
num_type *matrix_alloc(uint32_t m, uint32_t n) {
    return reallocarray(NULL, m * n, sizeof(num_type));
}

num_type *weights_create_or_load(const char *filename, uint32_t input_len,
                                 uint32_t output_len) {
    num_type *weight;
    if (filename == NULL) {
        weight = matrix_alloc(input_len, output_len);
        weights_norm_init(input_len, output_len, weight);
    } else {
        struct stat statbuf;
        size_t size_expected = input_len * output_len * sizeof(num_type);

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

static void vec_print(FILE *fp, uint32_t len, num_type vec[len]) {
    fprintf(fp, "[");
    for (uint32_t i = 0; i < len; i++) {
        fprintf(fp, "%d, ", vec[i]);
    }
    fprintf(fp, "]\n");
}

/**
 * `layer_construct` - Construct the neural network layer.
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
    linear(BATCH_LENGTH, INPUT_LENGTH, HIDDEN_LENGTH, hidden_weights, input,
           hidden_output);
    // fprintf(stderr, "process:\n");
    // vec_print(stderr, BATCH_LENGTH * HIDDEN_LENGTH, hidden_output);
    relu_vec(BATCH_LENGTH * HIDDEN_LENGTH, hidden_output);
    //   vec_print(stderr, BATCH_LENGTH * HIDDEN_LENGTH, hidden_output);
    linear(BATCH_LENGTH, HIDDEN_LENGTH, OUTPUT_LENGTH, output_weights,
           hidden_output, output);
    //   vec_print(stderr, BATCH_LENGTH * OUTPUT_LENGTH, output);
    sigmoid_vec(BATCH_LENGTH * OUTPUT_LENGTH, output);
    //   vec_print(stderr, BATCH_LENGTH * OUTPUT_LENGTH, output);
}

/**
 * `prediction_error` - Calculates the error between the output and the expected (target) array.
 *
 * - `target`: The target vector
 * - `error`: The calculated error value
 * Returns the number of hits (output == target)
 */
static uint64_t prediction_error(const num_type *target, double *error) {
    num_type *my_target = (num_type *)target;
    for (size_t pos = 0; pos < OUTPUT_LENGTH; pos++) {
        if (target[pos] != 0) {
            my_target[pos] = FX_DIV;
            break;
        }
    }
    error[0] =
        vec_delta(OUTPUT_LENGTH * BATCH_LENGTH, output, target, output_delta);
    uint64_t hits = 0;

    size_t max_pos     = SIZE_MAX;
    num_type max_value = NUM_TYPE_MIN;
    for (size_t pos = 0; pos < OUTPUT_LENGTH; pos++) {
        //   fprintf(stderr, "%d, ", output[pos]);
        if (output[pos] > max_value) {
            max_pos   = pos;
            max_value = output[pos];
        }
    }
    for (size_t pos = 0; pos < OUTPUT_LENGTH; pos++) {
        if (target[pos] != 0) {
            fprintf(stderr, "%zu, %zu, ", pos, max_pos);
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

    relu_derived(BATCH_LENGTH * HIDDEN_LENGTH, hidden_output, hidden_delta);
    train_sgd(BATCH_LENGTH, INPUT_LENGTH, HIDDEN_LENGTH, input, hidden_delta,
              LEARN_RATE, hidden_weights);
    sigmoid_derived(BATCH_LENGTH * OUTPUT_LENGTH, output, output_delta);
    train_sgd(BATCH_LENGTH, HIDDEN_LENGTH, OUTPUT_LENGTH, hidden_output,
              output_delta, LEARN_RATE, output_weights);
}

static void report_duration(double duration) {
    fprintf(stderr, "%5.2e\n", duration);
}

static void report_training(uint64_t total, uint64_t hits, double error,
                            double duration) {
    fprintf(stderr, "%1.4e, %.4f, %.3f, %5.2e\n", error,
            (double)hits / (double)total,
            (100.0 * (1.0 - (double)hits / (double)total)), duration);
}

void usage(char *progname) {
    fprintf(stderr, USAGE_FMT "\n", progname);
    exit(EXIT_FAILURE);
    /* NOTREACHED */
}

FILE *target_stream = NULL;
FILE *input_stream  = NULL;

int main(const int argc, char *argv[]) {
    int opt;

    // Handle the command line input
    while ((opt = getopt(argc, argv, "ht:")) != EOF) {
        switch (opt) {  // NOLINT(hicpp-multiway-paths-covered)
            case 't':
                target_stream = fopen(optarg, "r");
                if (target_stream == NULL) {
                    err(EXIT_FAILURE, "open target file");
                }
                break;
            case 'h':
            default:
                usage(basename(argv[0]));
                break;
        }
    }
    input_stream = stdin;
    for (int i = optind; i < argc; i++) {
        input_stream = fopen(argv[i], "r");
        break;  //stop after first file parameter is read
    }

    // Create the data arrays
    num_type input[INPUT_LENGTH * BATCH_LENGTH];
    num_type target[OUTPUT_LENGTH * BATCH_LENGTH];
    double batch_error = 0.0;

    layer_construct();

    uint64_t hits = 0, total = 0;

    while (fread(input, sizeof(num_type), ARRAY_LENGTH(input), input_stream) ==
           ARRAY_LENGTH(input)) {
        struct timespec route_period = stopwatch_start();

        predict(input);

        // Process (train) only if target_stream file is set (and open) to get
        // the expected output
        if (target_stream != NULL) {
            if (fread(target, sizeof(num_type), ARRAY_LENGTH(target),
                      target_stream) != ARRAY_LENGTH(target)) {
                err(EXIT_FAILURE, "loading target array");
            }

            hits += prediction_error(target, &batch_error);
           // vec_print(stderr, ARRAY_LENGTH(output), output);
            train(input);

            report_training(++total, hits, batch_error,
                            stopwatch_stop_us(route_period));
        } else {
            // print statistics to stderr
            report_duration(stopwatch_stop_us(route_period));
        }

        // Write the resulting output array to stdout
        if (fwrite(output, sizeof(num_type), OUTPUT_LENGTH * BATCH_LENGTH,
                   stdout) != OUTPUT_LENGTH * BATCH_LENGTH) {
            err(EXIT_FAILURE, "writing output array");
        }
    }

    // Finally, close the intput/output streams and clean up the memory usage
    // Write the training report to stdout and close the target stream if
    // training is enabled
    if (NULL != target_stream) {
        fclose(target_stream);
    }
    layer_destruct();
    if (input_stream != stdin) fclose(input_stream);

    return EXIT_SUCCESS;
}
