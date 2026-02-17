#ifndef CALIB_SCOPE_H
#define CALIB_SCOPE_H

static const int DFT = 8 * 1024;
#include "common_lib.h"

typedef struct {
  uint tx;
  uint rx;
  uint freq;
  uint chirp;
  uint amplitude;
  uint sinus_freq;
  uint dft;
} config_t;

typedef struct {
  config_t *c;
  openair0_device_t *rfdevice;
  int antennas;
  int dft_sz;
  c16_t **samplesRx;
  c16_t **samplesTx;
  pthread_mutex_t rxMutex;
  pthread_mutex_t txMutex;
} threads_t;
void CalibrationInitScope(threads_t *p);
#endif
