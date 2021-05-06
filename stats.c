#include "stats.h"

void stats_collect(struct stats c[static 1], double val, unsigned moments);
void stats_collect0(struct stats c[static 1], double val);
void stats_collect1(struct stats c[static 1], double val);
void stats_collect2(struct stats c[static 1], double val);
void stats_collect3(struct stats c[static 1], double val);
double stats_samples(struct stats c[static 1]);
double stats_mean(struct stats c[static 1]);
double stats_var(struct stats c[static 1]);
double stats_sdev(struct stats c[static 1]);
double stats_rsdev(struct stats c[static 1]);
double stats_skew(struct stats c[static 1]);
double stats_var_unbiased(struct stats c[static 1]);
double stats_sdev_unbiased(struct stats c[static 1]);
double stats_rsdev_unbiased(struct stats c[static 1]);
