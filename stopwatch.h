/**
 * ======================================================================
 * @file stopwatch.h
 *
 * @copyright Copyright 2020 Germar Schlegel. All rights reserved.
 * This project is released under the MIT License.
 *
 * @author Germar Schlegel
 * @date 2020-07-01 15:03:43
 * @brief tools to measure the duration of processes of the geisten deep
 * learning lib.
 *
 * Contains all functions to measure time relevant tasks
 * ======================================================================
 */

#pragma once

#include <time.h>

/**
 * stopwatch_start() - Start the stopwatch.
 *
 * Returns The timespec instance.
 */
static inline struct timespec stopwatch_start() {
    struct timespec now;
    timespec_get(&now, TIME_UTC);
    return now;
}

/**
 * Compute a time difference in microseconds.
 *
 * This uses a `double` to compute the time. If we want to
 * be able to track times without further loss of precision
 * and have `double` with 52 bit mantissa, this
 * corresponds to a maximal time difference of about 4.5E6
 * seconds, or 52 days.
 * Return The time in seconds.
 *
 * -`timespec` The last measured time.
 * Returns The time in microseconds
 */
static inline double stopwatch_stop_us(struct timespec timespec) {
    struct timespec result;
    timespec_get(&result, TIME_UTC);
    return (double)(result.tv_sec - timespec.tv_sec) * 1E+6 +
           (double)(result.tv_nsec - timespec.tv_nsec) * 1E-3;
}

/**
 * Compute a time difference in seconds.
 *
 *
 * -`timespec` The last measured time.
 * Returns The time in seconds
 */
static inline double stopwatch_stop_s(struct timespec timespec) {
    struct timespec result;
    timespec_get(&result, TIME_UTC);
    return (double)(result.tv_sec - timespec.tv_sec);
}

