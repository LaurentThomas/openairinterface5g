#ifndef CALIB_SCOPE_H
#define CALIB_SCOPE_H

enum e_pattern_scheme { e_SINUS, e_CHIRP, e_QPSK, e_QAM_16, e_QAM_64, e_QAM_256, e_RAMP, e_SIGNATURE, e_MAX_PATT_SCHEME };

typedef struct {
  uint tx;
  uint rx;
  uint freq;
  enum e_pattern_scheme tx_pattern;
  uint amplitude;
  uint sinus_freq;
  uint dft;
  char * file;
  char * dump_iq;
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
