/**
 * ======================================================================
 *
 * All rights reserved.
 * This project is released under the MIT License.
 *
 * @date 2020-07-01 15:00:18
 * @brief tools to measure the statistics of the geisten deep learning lib.
 * The name of the original author is lost ¯\_(⊙︿⊙)_/¯
 *
 * Contains all functions to measure the statistics of the geisten API
 * ======================================================================
 */

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#pragma once

/**
 * @file
 * @brief Collect some simple statistics online as we go.
 *
 * This uses a generalization of Welford's trick to compute running
 * mean and variance from Philippe Pébay. Formulas for robust, one-pass parallel
 * computation of covariances and arbitrary-order statistical moments. Technical
 * Report SAND2008-6212, SANDIA, 2008. URL
 * http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf.
 */

/**
 * @brief A simple data structure to collect the 0th to 3rd moment of
 * a statistic.
 *
 * @warning Since this also uses a @c double for the number of
 * samples, the validity of all this is restricted to about
 * @f$2^{50}@f$ samples.
 */
struct stats {
    double moment[4];
};

/**
 * @brief Return the number of samples that had been entered into the
 * statistic @a c.
 */
inline double stats_samples(struct stats c[static 1]) { return c->moment[0]; }

/**
 * @brief Return the mean value of the samples that had been entered
 * into the statistic @a c.
 */
inline double stats_mean(struct stats c[static 1]) { return c->moment[1]; }

/**
 * @brief Return the variance of the samples that had been entered
 * into the statistic @a c.
 */
inline double stats_var(struct stats c[static 1]) {
    return c->moment[2] / stats_samples(c);
}

/**
 * @brief Return the standard deviation of the samples that had been
 * entered into the statistic @a c.
 */
static inline double stats_sdev(struct stats c[static 1]) {
    return sqrt(stats_var(c));
}

/**
 * @brief Return the relative standard deviation of the samples that
 * had been entered into the statistic @a c.
 */
static inline double stats_rsdev(struct stats c[static 1]) {
    return sqrt(stats_var(c)) / stats_mean(c);
}

/**
 * @brief Return the normalized skew of the samples that had been
 * entered into the statistic @a c.
 */
static inline double stats_skew(struct stats c[static 1]) {
    double var = stats_var(c);
    return (c->moment[3] / pow(var, 1.5)) / stats_samples(c);
}

/**
 * @brief Return the unbiased variance of the samples that had been
 * entered into the statistic @a c.
 *
 * Use Bessel's correction to have an estimation of the unbiased
 * variance of the overall population.
 */
static inline double stats_var_unbiased(struct stats c[static 1]) {
    return c->moment[2] / (stats_samples(c) - 1);
}

/**
 * @brief Return the unbiased standard deviation of the samples that
 * had been entered into the statistic @a c.
 *
 * Use Bessel's correction to have an less biased estimation of the
 * variance of the overall population.
 */
static inline double stats_sdev_unbiased(struct stats c[static 1]) {
    return sqrt(stats_var_unbiased(c));
}

/**
 * @brief Return the unbiased relative standard deviation of the
 * samples that had been entered into the statistic @a c.
 */
static inline double stats_rsdev_unbiased(struct stats c[static 1]) {
    return stats_rsdev(c) * (1 + 1 / (4 * stats_samples(c)));
}

/**
 * @brief Add value @a val to the statistic @a c.
 */
inline void stats_collect(struct stats c[static 1], double val,
                          unsigned moments) {
    double n = stats_samples(c);
    double n0 = n - 1;
    double n1 = n + 1;
    double delta0 = 1;
    double delta = val - stats_mean(c);
    double delta1 = delta / n1;
    double delta2 = delta1 * delta * n;
    switch (moments) {
    default:
        c->moment[3] += (delta2 * n0 - 3 * c->moment[2]) * delta1;
        // fall through
    case 2:
        c->moment[2] += delta2;
        // fall through
    case 1:
        c->moment[1] += delta1;
        // fall through
    case 0:
        c->moment[0] += delta0;
    }
}

inline void stats_collect0(struct stats c[static 1], double val) {
    stats_collect(c, val, 0);
}

inline void stats_collect1(struct stats c[static 1], double val) {
    stats_collect(c, val, 1);
}

inline void stats_collect2(struct stats c[static 1], double val) {
    stats_collect(c, val, 2);
}

inline void stats_collect3(struct stats c[static 1], double val) {
    stats_collect(c, val, 3);
}

