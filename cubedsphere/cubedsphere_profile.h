#ifndef GAMERA_CS_PROFILE_H
#define GAMERA_CS_PROFILE_H
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
/* Observation only: no barriers, reductions, or changes to solver state. */
typedef struct {
  int initialized, enabled;
  long steps;
  double last, totals[20];
} gamera_cs_profile;
static inline void cs_profile_begin(gamera_cs_profile *p) {
  if (!p->initialized) {
    p->enabled = getenv("GAMERA_CS_PROFILE_TILE") != NULL;
    p->initialized = 1;
  }
  if (p->enabled) p->last = MPI_Wtime();
}
static inline void cs_profile_mark(gamera_cs_profile *p, int stage) {
  if (p->enabled) {
    const double now = MPI_Wtime();
    p->totals[stage] += now - p->last;
    p->last = now;
  }
}
static inline int cs_profile_status(gamera_cs_profile *p, int stage, int status) {
  cs_profile_mark(p, stage);
  return status;
}
static inline void cs_profile_report(const gamera_cs_profile *p,
    const char *kind, int rank, const char *const *names, int count) {
  if (!p->enabled) return;
  char line[2048];
  int used = snprintf(line, sizeof(line), "%s rank=%d steps=%ld", kind, rank, p->steps);
  for (int i = 0; i < count && used < (int)sizeof(line)-64; ++i)
    used += snprintf(line+used, sizeof(line)-(size_t)used, " %s=%.9f", names[i], p->totals[i]);
  fprintf(stderr, "%s\n", line);
}
#endif
