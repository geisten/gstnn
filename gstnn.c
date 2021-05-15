//
// Created by germar on 02.04.21.
//
#include <err.h>
#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "config.h"
#include "stats.h"
#include "stopwatch.h"

#define USAGE_FMT "%s [-t FILE] [-h] [-f]"

static void report_print(uint64_t total, uint64_t hits, struct stats error,
                         struct stats duration) {
    double time_rsdev  = stats_rsdev_unbiased(&duration);
    double error_rsdev = stats_rsdev_unbiased(&error);
    fprintf(stderr, "%1.4e, %4.02f, %.4f, %.3f, %5.2e, %4.02f",
            stats_mean(&error), error_rsdev, (double)hits / (double)total,
            (100.0 * (1.0 - (double)hits / (double)total)),
            stats_mean(&duration), time_rsdev);
}

void usage(char *progname) {
    fprintf(stderr, USAGE_FMT "\n", progname);
    exit(EXIT_FAILURE);
    /* NOTREACHED */
}

FILE *target_stream = NULL;
FILE *input_stream  = NULL;
bool freeze         = false;

int main(const int argc, char *argv[]) {
    int opt;

    // Handle the command line input
    while ((opt = getopt(argc, argv, "hft:")) != EOF) {
        switch (opt) {  // NOLINT(hicpp-multiway-paths-covered)
            case 't':
                target_stream = fopen(optarg, "r");
                if (target_stream == NULL) {
                    err(EXIT_FAILURE, "open target file");
                }
                break;
            case 'f':
                freeze = true;
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

    struct stats avg_duration = {};
    struct stats error_stats  = {};
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
            batch_error = prediction_error(target);

            if (!freeze) {
                train(input);
            }

            stats_collect2(&avg_duration, stopwatch_stop_us(route_period));
            stats_collect2(&error_stats, batch_error);

            report_print(++total, hits, error_stats, avg_duration);
            hits = monitor(ARRAY_LENGTH(target), target, error_stats,
                           avg_duration, total, hits);
            //create new line and close the monitor output
            fprintf(stderr, "\n");
        }

        // Write the resulting output array to stdout
        if (fwrite(output, sizeof(num_type), OUTPUT_LENGTH * BATCH_LENGTH,
                   stdout) != OUTPUT_LENGTH * BATCH_LENGTH) {
            err(EXIT_FAILURE, "writing output array");
        }
    }

    // Write the training report to stdout and close the target stream if
    // training is enabled
    if (NULL != target_stream) {
        fclose(target_stream);
    }

    layer_destruct();

    if (input_stream != stdin) fclose(input_stream);
    return EXIT_SUCCESS;
}
