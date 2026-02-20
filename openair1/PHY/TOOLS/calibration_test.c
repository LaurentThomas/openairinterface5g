/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define __USE_GNU
#include <stdint.h>
#include "openair1/PHY/defs_common.h"
#include <radio/COMMON/common_lib.h>
#include <executables/softmodem-common.h>
#include <openair1/PHY/TOOLS/calibration_scope.h>
#include "nfapi/oai_integration/vendor_ext.h"
#include "common/config/config_userapi.h"
#include <arpa/inet.h>
#include <pthread.h>

int oai_exit=false;
unsigned int mmapped_dma=0;
uint32_t timing_advance;
int8_t threequarter_fs;
uint64_t downlink_frequency[MAX_NUM_CCs][4];
int64_t uplink_frequency_offset[MAX_NUM_CCs][4];
int cpu_meas_enabled;
THREAD_STRUCT thread_struct;
uint32_t target_ul_mcs = 9;
uint32_t target_dl_mcs = 9;
uint64_t dlsch_slot_bitmap = (1<<1);
uint64_t ulsch_slot_bitmap = (1<<8);
uint32_t target_ul_bw = 50;

uint32_t target_dl_bw = 50;
uint32_t target_dl_Nl;
uint32_t target_ul_Nl;
char *uecap_file;
uint32_t dlsch_slot_modval;
uint32_t ulsch_slot_modval;
#include <executables/nr-softmodem.h>

int read_recplayconfig(recplay_conf_t **recplay_conf, recplay_state_t **recplay_state) {return 0;}
void nfapi_setmode(nfapi_mode_t nfapi_mode) {}
void set_taus_seed(unsigned int seed_init){};

// configmodule_interface_t *uniqCfg = NULL;
openair0_timestamp_t rx_timestamp = 0;
openair0_timestamp_t tx_timestamp = 0;
openair0_timestamp_t last_hole = 0;
pthread_cond_t tx_trig;

void *write_thread(void *arg)
{
  threads_t *params = (threads_t *)arg;
  c16_t **samplesTx = params->samplesTx;
  uint64_t ts = 0;
  const float WAVE_AMP = params->c->amplitude;
  const float sin_freq = params->c->sinus_freq;

  switch (params->c->tx_pattern) {
    case e_CHIRP: {
      double Fs = 122880.0;
      double f0 = -30 * 1000.0; // start freq
      double f1 = 30 * 1000.0; // end freq
      double T = params->dft_sz / Fs;
      double k = (f1 - f0) / T; // Hz/s sweep rate

      for (int i = 0; i < params->dft_sz; i++) {
        double t = ts / Fs;
        double phase = 2 * M_PI * (f0 * t + 0.5 * k * t * t);
        samplesTx[0][i].r = WAVE_AMP * cos(phase);
        samplesTx[0][i].i = WAVE_AMP * sin(phase);
        ts++;
      }
    } break;
    case e_QPSK: {
      __attribute__((aligned(32))) c16_t freq_signal[params->dft_sz] = {};
      int val = 0;
      for (int carrier = 0; carrier < params->dft_sz; carrier++) {
        const float sqrt2 = 0.70711;
        int amp = WAVE_AMP * sqrt2 * sqrt2;
        int i = rand() % 4;
        val ^= 1 << i;
        freq_signal[carrier] = (c16_t){(1 - 2 * (val & 1)) * amp, (1 - 2 * ((val >> 1) & 1)) * amp};
        ts++;
      }
      dft(get_dft(params->dft_sz), (int16_t *)freq_signal, (int16_t *)samplesTx[0], 1);
    } break;
    case e_QAM_16: {
      __attribute__((aligned(32))) c16_t freq_signal[params->dft_sz] = {};
      for (int carrier = 0; carrier < params->dft_sz; carrier++) {
        const float sqrt2 = 0.70711;
        const float sqrt10 = 0.31623;
        int amp = WAVE_AMP * sqrt10 * sqrt2;
        int i = rand() % 16;
        freq_signal[carrier] = (c16_t){(1 - 2 * (i & 1)) * (2 - (1 - 2 * ((i >> 2) & 1))) * amp,
                                       (1 - 2 * ((i >> 1) & 1)) * (2 - (1 - 2 * ((i >> 3) & 1))) * amp};
        ts++;
      }
      dft(get_dft(params->dft_sz), (int16_t *)freq_signal, (int16_t *)samplesTx[0], 1);
    } break;
    case e_QAM_256: {
      __attribute__((aligned(32))) c16_t freq_signal[params->dft_sz] = {};
      for (int carrier = 0; carrier < params->dft_sz; carrier++) {
        const float sqrt2 = 0.70711;
        const float sqrt170 = 0.076696;
        int amp = WAVE_AMP * sqrt170 * sqrt2;
        int i = rand() % 256;
        freq_signal[carrier] = (c16_t){
            ((1 - 2 * (i & 1)) * (8 - (1 - 2 * ((i >> 2) & 1)) * (4 - (1 - 2 * ((i >> 4) & 1)) * (2 - (1 - 2 * ((i >> 6) & 1))))))
                * amp,
            (1 - 2 * ((i >> 1) & 1))
                * (8 - (1 - 2 * ((i >> 3) & 1)) * (4 - (1 - 2 * ((i >> 5) & 1)) * (2 - (1 - 2 * ((i >> 7) & 1))))) * amp};
        ts++;
      }
      dft(get_dft(params->dft_sz), (int16_t *)freq_signal, (int16_t *)samplesTx[0], 1);
    } break;
    case e_SINUS:
      for (int i = 0; i < params->dft_sz; i++) {
        // Better to select a frequency having an integer division with the sampling rate to avoid having DFT leakage later on
        //  .r = cos and .i = sin -> having a positive spectrum
        //  For negative spectrum -> .r = sin and .i = cos
        samplesTx[0][i].r = WAVE_AMP * cos((ts * M_PI * 2 * sin_freq) / 122880000);
        samplesTx[0][i].i = WAVE_AMP * sin((ts * M_PI * 2 * sin_freq) / 122880000); // samplesTx[0][i].r;
        // Hamming Window - to allow some pseudo-continuity between batches as this is not a continuously generated signal as in
        // real life samplesTx[0][i].r = (samplesTx[0][i].r) * (0.54 - 0.46 * cos(2 * M_PI * i / (params->dft_sz-1)));
        // samplesTx[0][i].i = (samplesTx[0][i].i) * (0.54 - 0.46 * cos(2 * M_PI * i / (params->dft_sz-1)));
        // samplesTx[0][i]=(c16_t){i,-params->dft_sz+i};
        ts++;
      }
      break;
    default:
      abort();
  }

  double avg = 0;
  for (int i = 0; i < params->dft_sz; i++) {
    avg += sqrt(squaredMod(samplesTx[0][i]));
  }
  printf("avg: %f \n", avg / params->dft_sz);
  uint64_t count = 0;
  struct timespec last_second;
  clock_gettime(CLOCK_REALTIME, &last_second);

  openair0_timestamp_t last_tx_timestamp = 0, new_tx = 0;
  // this is tx ahead in main application, the driver has it's tx ahead that shuld be smaller to prevent starvation
  const int tx_ahead =  params->dft_sz * 20;
  char *flag = getenv("HOLE");
  while (!oai_exit) {
    do {
      AssertFatal(!pthread_mutex_lock(&params->txMutex), "");
      AssertFatal(!pthread_cond_wait(&tx_trig, &params->txMutex), "");
      new_tx = tx_timestamp & ~31;
      AssertFatal(!pthread_mutex_unlock(&params->txMutex), "");
    } while (last_tx_timestamp == new_tx);
    if (last_tx_timestamp +  params->dft_sz != new_tx)
      LOG_D(HW, "not continuous %ld\n", new_tx - (last_tx_timestamp + params->dft_sz));
    if (abs(last_tx_timestamp - new_tx) > 1228800) {
      LOG_W(HW, "large tx gap %ld\n", new_tx - (last_tx_timestamp + params->dft_sz));
      last_tx_timestamp = new_tx - params->dft_sz;
    }
    do {
      last_tx_timestamp += params->dft_sz;
      c16_t tmp[25];
      int loc=rand()%8000;
      if (flag && count % 1935 == 0) {
	memcpy(tmp, samplesTx[0]+loc,sizeof(tmp));
	memset(samplesTx[0]+loc, 0,sizeof(tmp));
	AssertFatal(!pthread_mutex_lock(&params->txMutex), "");
	last_hole=last_tx_timestamp + loc + tx_ahead;
	LOG_W(HW,"Set hole for: %lu\n", last_hole);
	AssertFatal(!pthread_mutex_unlock(&params->txMutex), "");
      }
      params->rfdevice
          ->trx_write_func(params->rfdevice, last_tx_timestamp + tx_ahead, (void **)samplesTx, params->dft_sz, params->antennas, 0);
      if(flag && count % 1935 == 0)
	memcpy(samplesTx[0]+loc, tmp, sizeof(tmp));
      count++;
    } while (last_tx_timestamp < new_tx);
    last_tx_timestamp = new_tx;
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    if (now.tv_sec != last_second.tv_sec) {
      LOG_I(HW, "write thread wrote %lu times in one second\n", count);
      last_second = now;
      count = 0;
    }
  }
  return NULL;
}

void *read_thread(void *arg)
{
  threads_t *params = (threads_t *)arg;
  c16_t **samplesRx = params->samplesRx;
  uint64_t count = 0;
  int warmup=0;
  struct timespec last_second;
  clock_gettime(CLOCK_REALTIME, &last_second);
  while (!oai_exit) {
    uint64_t old = rx_timestamp;
     __attribute__((aligned(32))) c16_t rx[ params->dft_sz ];
     c16_t *rxptr=rx;
    int ret =
        params->rfdevice->trx_read_func(params->rfdevice, &rx_timestamp, (void **)&rxptr, params->dft_sz, params->antennas);
    if (old + params->dft_sz != rx_timestamp)
      LOG_E(HW, "not continuous rx %ld\n", rx_timestamp - (old + params->dft_sz));
    if (ret != params->dft_sz)
      printf("read of :%d\n", ret);
    count++;
    AssertFatal(!pthread_mutex_lock(&params->rxMutex), "");
    memcpy(samplesRx[0],rx, sizeof(rx));
    // LOG_E(HW,"signal: %lu\n", tx_timestamp);
    /*
    for (int i = 0; i < params->dft_sz; i++)
      params->samplesRx[0][i] = (c16_t){params->samplesRx[0][i].r >>2, params->samplesRx[0][i].i >>2};
    */
    double min=UINT64_MAX;
    int min_pos=0;
    if (getenv("HOLE")) {
      for (int i = 0; i < params->dft_sz-25; i++) {
	double local=0;
	for (int j=i; j<i+10; j++) {
	  local+=params->samplesRx[0][j].r*params->samplesRx[0][j].r+params->samplesRx[0][j].i*params->samplesRx[0][j].i;
	}
	if (local < min ) {
	  min=local;
	  min_pos=i;
	}
      }
    }
    AssertFatal(!pthread_mutex_lock(&params->txMutex), "");
    tx_timestamp = rx_timestamp;
    if (min < 1000) {
      LOG_I(HW, "found hole %lu, programmed for %lu, received %ld later\n", min_pos+rx_timestamp, last_hole,min_pos+rx_timestamp - last_hole  );
    }
    warmup++;
    if (warmup > 1024)
      AssertFatal(!pthread_cond_signal(&tx_trig), "");
    AssertFatal(!pthread_mutex_unlock(&params->txMutex), "");
    AssertFatal(!pthread_mutex_unlock(&params->rxMutex), "");
    //    dft(get_dft(len), (int16_t *)form->timeDomain, (int16_t *)form->freqDomain, 1);
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    if (now.tv_sec != last_second.tv_sec) {
      printf("read thread got %lu blocks in one second, samples per block: %d, nb samples: %lu\n", count, ret, count*ret);
      count=0;
      last_second.tv_sec++;
      #if 0
      FILE *fd = fopen("trace.iq", "w+");
      if (!fd)
        abort();

      /* We should advance +1 only if header was detected in previous steps
       * Which will make the read working even without timestamped rxdata
       */
      c16_t *s = samplesRx[0]; /* Exclude the header from the samples */
      for (int i = 0; i < ret; i++) {
        /* We need to throw the entie 256-bits word if we detect the 64-bits header.
         * This may happens when receiving a big packet size in chuncks.
         */
        fprintf(fd, "%d %d %d\n", i, s[i].r, s[i].i);
      }
      #endif
    }
  }
  return NULL;
}

int main(int argc, char **argv) {
  /// static configuration for NR at the moment
  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == NULL) {
    exit_fun("[SOFTMODEM] Error, configuration module init failed\n");
  }
  set_softmodem_sighandler();
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  logInit();

  CONFIG_SETRTFLAG(CONFIG_NOEXITONHELP);
  get_common_options(uniqCfg);

  config_t c = {1, 1, 3750000, 1, 2047, 10000, 8192};
  paramdef_t cmdline_params[] = {
      {"tx", "enable tx", 0, .uptr = &c.tx, .defintval = 1, TYPE_UINT, 0},
      {"rx", "enable tx", 0, .uptr = &c.rx, .defintval = 1, TYPE_UINT, 0},
      {"freq", "center frequency in kHz", 0, .uptr = &c.freq, .defintval = 1, TYPE_UINT, 0},
      {"tx_pattern",
       "generate signal for a chirp (0), qam-64 (1) or pure sinewave (default)",
       0,
       .uptr = &c.tx_pattern,
       .defintval = 2,
       TYPE_UINT,
       0},
      {"amplitude", "signal amplitude (int16)", 0, .uptr = &c.amplitude, .defintval = 2047, TYPE_UINT, 0},
      {"sinus_freq", "if chirp is false, sinut frequency in KHz", .uptr = &c.sinus_freq, .defintval = 10000, TYPE_UINT, 0},
      {"dft", "dft size for signal frequency/time convertion", .uptr = &c.dft, .defintval = 8192, TYPE_UINT, 0},
  };
  config_process_cmdline(uniqCfg, cmdline_params, sizeofArray(cmdline_params), NULL);
  CONFIG_CLEARRTFLAG(CONFIG_NOEXITONHELP);
  
  lock_memory_to_ram();
  int h=open("/dev/cpu_dma_latency", 0666);
  int lat=2; // micro second
  assert(sizeof(lat)==write(h,&lat,sizeof(lat)));

  int sampling_rate = 30.72e6 * 6;

  int antennas = 1;
  uint64_t freq = c.freq * 1000;
  int rxGain = 90;
  int txGain = 0;
  int filterBand = 40e6;

  openair0_config_t openair0_cfg = {
      .duplex_mode = 0,
      .sample_rate = sampling_rate,
      .tx_sample_advance = 0,
      .rx_num_channels = antennas,
      .tx_num_channels = antennas,
      .rx_freq = {freq, freq, freq, freq},
      .tx_freq = {freq, freq, freq, freq},
      .rx_gain_calib_table = NULL,
      .rx_gain = {rxGain, rxGain, rxGain, rxGain},
      .tx_gain = {txGain, txGain, txGain, txGain},
      .rx_bw = filterBand,
      .tx_bw = filterBand,
      .clock_source = external, // internal gpsdo external
      .time_source = internal, // internal gpsdo external
      .sdr_addrs="addr=192.168.30.2",
      .autocal = {0},
      //! rf devices work with x bits iqs when oai have its own iq format
      //! the two following parameters are used to convert iqs
      .configFilename = "",
      .recplay_mode = 0,
      .recplay_conf = NULL,
  };
  //-----------------------
  
  openair0_device_t rfdevice = {
    .type = NONE_DEV,
    .transp_type = NONE_TP,
    .host_type = MIN_HOST_TYPE,
  };
  
  openair0_device_load(&rfdevice, &openair0_cfg);

  printf("generate a sinus wave at middle RB");
  load_dftslib();

  c16_t **samplesRx = malloc16(antennas * sizeof(c16_t *));
  for (int i = 0; i < antennas; i++) {
    samplesRx[i] = malloc16_clear(c.dft * sizeof(c16_t));
  }
  c16_t **samplesTx = malloc16(antennas * sizeof(c16_t *));
  for (int i = 0; i < antennas; i++) {
    samplesTx[i] = malloc16_clear(c.dft * sizeof(c16_t));
  }

  /* scopedata shall be filled from a software FIFO and not directly from the samples */
  threads_t params = (threads_t){&c, &rfdevice, antennas, c.dft, samplesRx, samplesTx};
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
  AssertFatal(!pthread_mutex_init(&params.txMutex, &attr), "");
  AssertFatal(!pthread_mutex_init(&params.txMutex, &attr), "");
  AssertFatal(!pthread_cond_init(&tx_trig, NULL), "");
  CalibrationInitScope(&params);
  rfdevice.trx_start_func(&rfdevice);

  pthread_t w_thread;
  if (c.tx)
    threadCreate(&w_thread, write_thread, &params, "write_thr", 3, OAI_PRIORITY_RT);
  pthread_t r_thread;
  if (c.rx)
    threadCreate(&r_thread, read_thread, &params, "read_thr", 2, OAI_PRIORITY_RT);
  (void)pthread_join(w_thread, NULL);
  (void)pthread_join(r_thread, NULL);

  return 0;
}
