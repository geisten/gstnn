//
// Created by germar on 01.07.20.
//

#include <glab/stopwatch.h>
#include "test.h"

TEST_INIT();

inline void stopwatch_startp(struct timespec *ts) {
    timespec_get(ts, TIME_UTC);
}

/**
 * Compute a time difference in microseconds.
 *
 * This uses a @c double to compute the time. If we want to
 * be able to track times without further loss of precision
 * and have @c double with 52 bit mantissa, this
 * corresponds to a maximal time difference of about 4.5E6
 * seconds, or 52 days.
 * Return The time in seconds.
 *
 * @param later The last measured time.
 * @param sooner The first measured time.
 * @return The time in microseconds
 */
inline double stopwatch_stopp_us(struct timespec *start,
                                 struct timespec *stop) {
    timespec_get(stop, TIME_UTC);
    return (double)(stop->tv_sec - start->tv_sec) * 1E+6 +
           (double)(stop->tv_nsec - start->tv_nsec) * 1E-3;
}

static void test_stopwatch_performance() {
    // first test as implemented
    double average = .0;
    const int number_of_tests = 10000;
    const struct timespec sleep[] = {{0, 100L}};
    for (int i = 0; i < number_of_tests; i++) {
        struct timespec sw = stopwatch_start();
        nanosleep(sleep, NULL);
        average += stopwatch_stop_us(sw);
    }
    printf("it took %f us to measure 10us %d times\n", average,
           number_of_tests);

    average = 0.0;
    struct timespec sw, st;
    for (int i = 0; i < number_of_tests; i++) {
        stopwatch_startp(&sw);
        nanosleep(sleep, NULL);
        average += stopwatch_stopp_us(&sw, &st);
    }

    printf("it took %f us to measure 10us %d times with alternative approach\n",
           average, number_of_tests);
}

int main() {
    test_stopwatch_performance();
    return TEST_RESULT;
}