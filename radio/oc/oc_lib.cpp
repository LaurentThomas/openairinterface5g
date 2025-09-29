/*
 * Licensed by open cells project
 */
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <complex>
#include <fstream>
#include <cmath>
#include <time.h>
#ifdef OAI_INTEGRATION
#include "common_lib.h"
#include "assertions.h"
#else
//#define LOG_E(m, a...) printf(a)
#include "common_lib.h"
#endif
#include "system.h"
#include <sys/resource.h>
#include "common/platform_types.h"
#include "openair1/PHY/sse_intrin.h"
#include "common/utils/LOG/log.h"
#include "common/utils/time_meas.h"

#define DEVICE_WRITE_DEFAULT "/dev/xdma0_h2c_0"
#define DEVICE_READ_DEFAULT "/dev/xdma0_c2h_0"
#define WRITE_BLOCK_NB_SAMPLES 2048
#define NB_BLOCKS_PER_READ 16
#define READ_BLOCK_NB_SAMPLES 2048
#define PKT_HEADER_NB_SAMPLES 7
#define PKT_FOOTER_NB_SAMPLES 1
#define PKT_OVERHEAD_NB_SAMPLES (PKT_HEADER_NB_SAMPLES + PKT_FOOTER_NB_SAMPLES)
static const uint64_t magic_tx = 0xA5A50be3A5A5A5A5LL;
static const uint64_t magic_rx = 0xA5A50be3A5A5A5A5LL;
static const uint32_t magic_footer1 = 0xce11;
static const uint32_t magic_footer2 = 0x5A;

typedef struct {
  uint64_t control;
  uint32_t packetSeqNum: 16;
  uint32_t packetSz: 16;
  uint32_t seqId: 2;
  uint32_t filler: 6;
  uint32_t markers: 8;
  uint32_t filler2: 16;
  uint32_t txGain: 24;
  uint32_t filler3: 8;
  uint32_t ppsOffset;
  uint64_t timestamp;
} __attribute__((packed)) headerTx_t;

typedef struct {
  uint64_t control;
  uint16_t packetSeqNum;
  uint16_t packetSz;
  uint8_t RXen     : 1;
  uint8_t TXen     : 1;
  uint8_t PLLlocked: 1;
  uint8_t ADCsync  : 1;
  uint8_t DACsync  : 1;
  uint8_t TXlate   : 1;
  uint8_t TXseqerr : 1;
  uint8_t filler   : 1;
  uint8_t InbandReadAddr;
  uint16_t InbandReadValue;
  uint32_t ppsOffset: 30;
  uint32_t ppsAlive: 1;
  uint32_t gpsLock: 1;
  uint64_t timestamp;
} __attribute__((packed)) headerRx_t;

typedef struct {
  uint16_t control1;
  uint8_t control2;
  uint8_t atomicPacket: 1;
  uint8_t timerOverflow: 1;
  uint8_t filler: 6;
} __attribute__((packed)) footer_t;

typedef struct {
  headerRx_t h;
  uint32_t b[READ_BLOCK_NB_SAMPLES];
  footer_t f;
} __attribute__((packed)) packet_t;

static inline void dumpHD(std::string ctx, headerRx_t h)
{
  printf("header dump, %s\n", ctx.c_str());
  uint8_t *z = (uint8_t *)&h;
  for (int i = 0; i < PKT_HEADER_NB_SAMPLES; i++)
    printf("  %02x:%02x %02x:%02x\n", z[i * 4 + 0], z[i * 4 + 1], z[i * 4 + 2], z[i * 4 + 3]);
  printf(
	 "decoded magic: %lx\n"
	 "   dataSz(words): %u, packetSeq:%u\n"
	 "   InbandReadValue %x, InbandReadAddr:%x, DACsync %u, ADCsync: %u, PLLlocked:%u, TXen:%u, RXen:%u\n"
	 "   gpslocked %d, ppsalive %d, pps offset %d\n"
	 "   timestamp: %lu\n",
	 h.control,
	 h.packetSz,
	 h.packetSeqNum,
	 h.InbandReadValue,
	 h.InbandReadAddr,
	 h.DACsync,
	 h.ADCsync,
	 h.PLLlocked,
	 h.TXen,
	 h.RXen,
	 h.gpsLock,
	 h.ppsAlive,
	 h.ppsOffset,
	 h.timestamp);
}

typedef struct {
  char filename_write[FILENAME_MAX];
  char filename_read[FILENAME_MAX];
  int fd_write;
  int fd_read;
  int num_underflows;
  int num_overflows;
  int num_seq_errors;
  int64_t tx_count;
  int64_t rx_count;
  int wait_for_first_pps;
  int use_gps;
  openair0_timestamp_t rx_timestamp;
  openair0_timestamp_t tx_ts;
  c16_t **tx_block;
  size_t tx_block_sz;
  bool first_tx;
  bool rxMagicFound;
  uint seqNum;
  uint lastPpsOffset;
  int  nb_blocks_per_read;
  int remain_samples;
  packet_t *current_rx_packet;
} oc_state_t;

typedef struct {
  openair0_device_t *rfdevice;
  int antennas;
  int dft_sz;
} threads_t;

static int check_ref_locked(oc_state_t *s)
{
  return 0;
}

static int sync_to_gps(openair0_device_t *device)
{
  return 0;
}

void *write_thread(void *arg)
{
  // threads_t params = *(threads_t *)arg;

  return NULL;
}

void *read_thread(void *arg)
{
  // threads_t params = *(threads_t *)arg;
  return NULL;
}

#if 0
//TEST 4G 20MHz
int nsamps2 = (nsamps*4) / 8 ;
simde__m256i buff_tx[nsamps2];
simde__m256i *out=buff_tx;
int *in=(int *)buff[0];
// bring RX data into 12 LSBs for softmodem RX
for (uint j = 0; j < nsamps/2; j++) {
  simde__m256i tmp=simde_mm256_set_epi32(in[1],in[1],in[1],in[1], in[0],in[0],in[0],in[0]);
  in+=2;
  *out++ = simde_mm256_slli_epi16(tmp, 6);
 }
#endif

static int32_t signalEnergy(int32_t *input, uint32_t length)
{
  // init
  simde__m128 mm0 = simde_mm_setzero_ps();

  // Acc
  for (uint32_t i = 0; i < (length >> 2); i++) {
    simde__m128i in = simde_mm_loadu_si128((simde__m128i *)input);
    mm0 = simde_mm_add_ps(mm0, simde_mm_cvtepi32_ps(simde_mm_madd_epi16(in, in)));
    input += 4;
  }

  // leftover
  float leftover_sum = 0;
  c16_t *leftover_input = (c16_t *)input;
  uint16_t lefover_count = length - ((length >> 2) << 2);
  for (int32_t i = 0; i < lefover_count; i++) {
    leftover_sum += leftover_input[i].r * leftover_input[i].r + leftover_input[i].i * leftover_input[i].i;
  }

  // Ave
  float sums[4];
  simde_mm_store_ps(sums, mm0);
  return (uint32_t)((sums[0] + sums[1] + sums[2] + sums[3] + leftover_sum) / (float)length);
}

// DC-filter: 0 will be done in FPGA after seeing 128-consecutive samples having the same value
static inline int write_block(oc_state_t *s, c16_t *samples, uint sz)
{
  uint blockSz = sz * sizeof(c16_t) + sizeof(headerTx_t);
  uint8_t buf[blockSz];
  *(headerTx_t *)buf = (headerTx_t){.control = magic_tx,
                                    .packetSeqNum = 0x1211,
                                    .packetSz = 0x0800,
                                    .seqId = 1,
                                    .filler = 0x02,
                                    .markers = 0xb1,
                                    .filler2 = 0xabcd,
                                    .txGain = 0x112233,
                                    .filler3 = 0xf0,
                                    .ppsOffset = 0x28272625,
                                    .timestamp = (uint64_t)s->tx_ts};
  memcpy(buf + sizeof(headerTx_t), samples, sz * sizeof(c16_t));
  uint64_t start = rdtsc_oai();
  size_t wrote = write(s->fd_write, buf, blockSz);
  uint64_t end = rdtsc_oai();
  if (end - start > 300 * 5000)
    LOG_E(HW, "one write to xdma took %ld µs, ts:%lu\n", (end - start) / 5000, s->tx_ts);
  if (wrote != blockSz)
    LOG_E(HW, "write to SDR failed, request: %u, wrote %ld\n", blockSz, wrote);
  if (wrote < 0)
    LOG_E(HW, "write to %s failed, errno %d:%s\n", s->filename_write, errno, strerror(errno));
  s->tx_ts += sz;
  s->tx_block_sz = 0;
  s->tx_count++;
  LOG_D(HW, "wrote at ts: %lu, energy: %u\n", s->tx_ts, signalEnergy((int32_t *)samples, sz));
  return sz;
}

static int oc_write(openair0_device_t *device, openair0_timestamp_t timestamp, void **buff, int nsamps, int cc, int flags)
{
  oc_state_t *s = (oc_state_t *)device->priv;
  timestamp -= device->openair0_cfg->command_line_sample_advance + device->openair0_cfg->tx_sample_advance;

  if (s->first_tx) {
    s->tx_ts = timestamp;
    s->first_tx = false;
  }

  int64_t gap = timestamp - s->tx_ts;
  if (gap < 0) {
    LOG_E(HW, "out of sequence\n");
    gap = 0;
  }

  if (gap)
    LOG_D(HW, "gap of %ld\n", gap);

  int wr_sz = nsamps;
  while (wr_sz) {
    int tmp = std::min(wr_sz, WRITE_BLOCK_NB_SAMPLES);
    int sz = write_block(s, ((c16_t *)buff[0]) + nsamps - wr_sz, tmp);
    if (sz != tmp)
      LOG_E(HW, "ask to write %d, res is %d\n", tmp, sz);
    wr_sz -= sz;
  }
  s->tx_ts = timestamp + nsamps;
  return nsamps;
}

static void initial_block_align (oc_state_t *s) {
  printf("Synchronizing rx\n");
  __attribute__ ((aligned(32))) uint32_t b[READ_BLOCK_NB_SAMPLES + PKT_OVERHEAD_NB_SAMPLES];
  int idx = 0;
  int old = 0;
  uint64_t bytes = 0;
  //while (1) {
  ssize_t ret = read(s->fd_read, b, sizeof(b));
    if (ret != sizeof(b)) {
      printf("Error reading %ld bytes: %ld\n",sizeof(b), ret);
      usleep(10000);
      return;
    }
    int i;
    headerRx_t *rx = NULL;
    for (i = 0; i < READ_BLOCK_NB_SAMPLES + PKT_HEADER_NB_SAMPLES; i++)
      if (b[i] == (magic_rx&UINT32_MAX) && b[i + 1] == ((magic_rx>>32)&UINT32_MAX)) {
	printf("found first magic at %d (inblock: %d), dist: %d, bytes %lu\n",
	       idx * (READ_BLOCK_NB_SAMPLES+ PKT_OVERHEAD_NB_SAMPLES) + i,
	       i,
	       idx * (READ_BLOCK_NB_SAMPLES + PKT_OVERHEAD_NB_SAMPLES) + i - old,
	       bytes);
  rx = (headerRx_t *)(b + i);
  dumpHD("first header:", *rx);
	break;
      }
    if (i == (READ_BLOCK_NB_SAMPLES + PKT_HEADER_NB_SAMPLES)) {
      printf("%%error magic not found\n");
      return;
    }
    ret = read(s->fd_read, b, i * sizeof(*b));
    s->seqNum = rx->packetSeqNum + 1;
    s->rx_timestamp = (int64_t)rx->timestamp + rx->packetSz;
    s->rx_count = 1;
    //}
}

static bool get_blocks(oc_state_t *s) {
  static struct timespec last_second={}, origin={};
  static struct timespec now={};
  static uint64_t tot_samples= 0;

  int readSz= sizeof(*s->current_rx_packet)* s->nb_blocks_per_read;
  packet_t* p= s->current_rx_packet;
  ssize_t ret = read(s->fd_read, p, readSz);
  if (ret != readSz || p[0].h.control != magic_rx) {
    printf((char *)"Error reading header asked for %d bytes, got %ld, magic: %lx\n", readSz, ret,  p[0].h.control);
    dumpHD("lost good header:", p[0].h);
    // abort();
    s->rx_count = -1;
    return false;
  }

  clock_gettime(CLOCK_REALTIME, &now);
  if ( last_second.tv_sec==0) {
    last_second=now;
    origin=now;
  }
  tot_samples+= NB_BLOCKS_PER_READ * READ_BLOCK_NB_SAMPLES;
  if (now.tv_sec != last_second.tv_sec) {
    printf("avg rate:%f\n",  (float)tot_samples/(now.tv_sec*1000000 - origin.tv_sec*1000000 +  now.tv_nsec/1000.0 - origin.tv_nsec/1000.0 ));
    last_second.tv_sec++;
  }

  for (int i = 0; i <s->nb_blocks_per_read ; i++) {
    if (s->rx_timestamp != (int64_t)p[i].h.timestamp)
      printf("expected ts: %lu got %lu, diff %ld, seq num %d, atomicPacket %d, timerOverflow %d\n",
	     s->rx_timestamp,
	     p[i].h.timestamp,
	     (int64_t)p[i].h.timestamp - s->rx_timestamp,
	     s->seqNum,
	     (~p[i].f.atomicPacket &0x01),
	     p[i].f.timerOverflow);
    s->rx_timestamp=p[i].h.timestamp + READ_BLOCK_NB_SAMPLES;
    if (llabs((int64_t)s->lastPpsOffset - (int64_t)p[i].h.ppsOffset) > 12)
      // pps_in is based on a real pulse from the GPS. It may have some jitter and drift during time.
      // Thus it is normal to have some diff between theoretical (expected) and real (measured) values.
      // >12 ->l means that we flag errors higher than +/- 0.1ppm.
      // When the board's FPGA starts, the vcxo is not yet discplined to the GPS, thus we may observe
      // some errors until the frequency error algorithm converges
      printf("expected pps offset %u got %u, diff %d, seq num %d\n",
	     s->lastPpsOffset,
	     p[i].h.ppsOffset,
	     p[i].h.ppsOffset- s->lastPpsOffset,
	     s->seqNum);
    s->lastPpsOffset = (p[i].h.ppsOffset + READ_BLOCK_NB_SAMPLES )% 122880000 ;
    if (!p[i].h.ppsAlive)
      printf("pps not alive\n");
    if (p[i].f.control1 != magic_footer1 || p[i].f.control2 != magic_footer2)
      printf("footer error\n");
    if (p[i].f.timerOverflow)
      printf("timerOverflow\n");
    if (p[i].f.atomicPacket) {
      printf("Not atomic\n");
      }
    static int last_filler = 0;
    if (last_filler != p[i].f.filler)
      printf("filler changed to %x\n", p[i].f.filler);
    last_filler = p[i].f.filler;
    if (s->seqNum % 65536 != p[i].h.packetSeqNum) {
      printf("expected packet sequence number %u got %u, diff %d\n",
	     s->seqNum,
	     p[i].h.packetSeqNum,
	     p[i].h.packetSeqNum - s->seqNum);
      s->seqNum = p[i].h.packetSeqNum;
      //s->rx_count = -1;
      //return false;
    }
    s->seqNum++;
  }
  s->remain_samples= sizeof(s->current_rx_packet->b) * s->nb_blocks_per_read / sizeof(*s->current_rx_packet->b);
  //printf("got %d samples\n",  s->remain_samples);
  return true;
}

static int oc_read(openair0_device_t *device, openair0_timestamp_t *ptimestamp, void **buff, int nsamps, int cc)
{
  oc_state_t *s = (oc_state_t *)device->priv;

  if (s->rx_count == -1)
    initial_block_align(s);
  if (s->rx_count == -1)
    return 0;

  c16_t **output=(c16_t**)buff;
  int remain_to_get=nsamps;
  while (remain_to_get > 0) {
    while (s->remain_samples > 0 && remain_to_get > 0) {
      packet_t* p=s->current_rx_packet;
      int nb_samples_per_packet=sizeof(p->b)/sizeof(*p->b);
      int nb_samples=nb_samples_per_packet*s->nb_blocks_per_read;
      int consumed_samples=nb_samples-s->remain_samples;
      int bloc=consumed_samples/nb_samples_per_packet;
      int nb_consumed_samples_in_block= consumed_samples%nb_samples_per_packet;
      int remaing_samples_in_block= nb_samples_per_packet-nb_consumed_samples_in_block;
      int toCopy=std::min(remaing_samples_in_block,remain_to_get );
      memcpy(output[0]+nsamps-remain_to_get,p[bloc].b+nb_consumed_samples_in_block,toCopy*sizeof(*p->b));
      s->remain_samples-=toCopy;
      remain_to_get -=toCopy;
      *ptimestamp = s->rx_timestamp;
    }
    if(  remain_to_get > 0 ) {
      static uint64_t a;
      uint64_t n=rdtsc_oai();
      if (n-a > 4200*100)
	printf("blocked for %ld\n", (n-a)/4200);
      a=n;
      if (!get_blocks(s)) {
	printf("getblocks returned bad\n");
	return 0;
      }
      uint64_t b=rdtsc_oai();
      if (b - a > 4200 * 600)
        printf("one call to xdma: %ld µs\n", (b - a) / 4200);
      a=b;
    }
  }
  return nsamps;
}

static int oc_set_freq(openair0_device_t *device, openair0_config_t *openair0_cfg)
{
  // oc_state_t *s = (oc_state_t *)device->priv;
  printf("Setting TX Freq %f, RX Freq %f, tune_offset: %f\n",
         openair0_cfg[0].tx_freq[0],
         openair0_cfg[0].rx_freq[0],
         openair0_cfg[0].tune_offset);
  return 0;
}

static int oc_set_gains(openair0_device_t *device, openair0_config_t *openair0_cfg)
{
  // oc_state_t *s = (oc_state_t *)device->priv;
  LOG_I(HW, "Setting RX gain to %f \n", openair0_cfg[0].rx_gain[0]);
  return 0;
}

static int oc_stop(openair0_device_t *device)
{
  return 0;
}

void set_rx_gain_offset(openair0_config_t *openair0_cfg, int chain_index, int bw_gain_adjust)
{
  int i = 0;
  // loop through calibration table to find best adjustment factor for RX frequency
  double min_diff = 6e9, diff, gain_adj = 0.0;

  if (bw_gain_adjust == 1) {
    switch ((int)openair0_cfg[0].sample_rate) {
    case 46080000:
      break;

    case 30720000:
      break;

    case 23040000:
      gain_adj = 1.25;
      break;

    case 15360000:
      gain_adj = 3.0;
      break;

    case 7680000:
      gain_adj = 6.0;
      break;

    case 3840000:
      gain_adj = 9.0;
      break;

    case 1920000:
      gain_adj = 12.0;
      break;

    default:
      LOG_E(HW, "unknown sampling rate %d\n", (int)openair0_cfg[0].sample_rate);
      // exit(-1);
      break;
    }
  }

  while (openair0_cfg->rx_gain_calib_table[i].freq > 0) {
    diff = fabs(openair0_cfg->rx_freq[chain_index] - openair0_cfg->rx_gain_calib_table[i].freq);
    LOG_I(HW,
          "cal %d: freq %f, offset %f, diff %f\n",
          i,
          openair0_cfg->rx_gain_calib_table[i].freq,
          openair0_cfg->rx_gain_calib_table[i].offset,
          diff);

    if (min_diff > diff) {
      min_diff = diff;
      openair0_cfg->rx_gain_offset[chain_index] = openair0_cfg->rx_gain_calib_table[i].offset + gain_adj;
    }

    i++;
  }
}

static int oc_get_stats(openair0_device_t *device)
{
  return (0);
}

static int oc_reset_stats(openair0_device_t *device)
{
  return (0);
}

int oc_write_init(openair0_device_t *device)
{
  LOG_E(HW, "trx_write_init should not be called, and is a design error even for USRP\n");
  return -1;
}

static int oc_start(openair0_device_t *device)
{
  oc_state_t *s = (oc_state_t *)device->priv;
  s->rx_count = -1;
  s->wait_for_first_pps = 1;
  s->first_tx = true;
  s->nb_blocks_per_read=NB_BLOCKS_PER_READ;
  int nb_tx = device->openair0_cfg->tx_num_channels;
  s->tx_block = (c16_t **)malloc(nb_tx * sizeof(*s->tx_block));
  // for (int i = 0; i < nb_tx; i++)
  // s->tx_block[i] = (c16_t *)malloc16(OC_BUFFER);
  s->current_rx_packet=(packet_t*)malloc16(s->nb_blocks_per_read*sizeof(* s->current_rx_packet));
  s->fd_write = open(s->filename_write, O_WRONLY);
  if (s->fd_write < 0) {
    LOG_E(HW, "Open %s failed, errno %d:%s\n", s->filename_write, errno, strerror(errno));
    exit(1);
  }
  s->fd_read = open(s->filename_read, O_RDONLY);
  if (s->fd_read < 0) {
    LOG_E(HW, "Open %s failed, errno %d:%s\n", s->filename_read, errno, strerror(errno));
    exit(1);
  }
  /*
    threads_t params = (threads_t){&rfdevice, antennas, DFT};
    pthread_t w_thread;
    threadCreate(&w_thread, write_thread, &params, "write_thr", -1, OAI_PRIORITY_RT);
    pthread_t r_thread;
    threadCreate(&r_thread, read_thread, &params, "read_thr", -1, OAI_PRIORITY_RT);
  */
  oc_set_gains(device, device->openair0_cfg);
  oc_set_freq(device, device->openair0_cfg);
  sync_to_gps(device);
  check_ref_locked(s);
  // Fixme: set sampling rate, lack of API in OAI
  return 0;
}

static void oc_end(openair0_device_t *device)
{
  if (device == NULL)
    return;
  oc_state_t *s = (oc_state_t *)device->priv;
  int nb_tx = device->openair0_cfg->tx_num_channels;
  if (s && nb_tx > 0) {
    for (int i = 0; i < nb_tx; i++)
      free(s->tx_block[i]);
    free(s->tx_block);
  }
}

extern "C" {

  int device_init(openair0_device_t *device, openair0_config_t *openair0_cfg)
  {
    LOG_I(HW, "openair0_cfg->clock_source == '%d' (internal = %d, external = %d)\n", openair0_cfg->clock_source, internal, external);
    oc_state_t *st;
    if (device->priv == NULL) {
      st = (oc_state_t *)calloc(1, sizeof(oc_state_t));
      device->priv = st;
      strcpy(st->filename_write, DEVICE_WRITE_DEFAULT);
      strcpy(st->filename_read, DEVICE_READ_DEFAULT);
      AssertFatal(st != NULL, "OC device: memory allocation failure\n");
    } else {
      LOG_E(HW, "multiple calls to device init detected\n");
      return 0;
    }
    device->openair0_cfg = openair0_cfg;
    device->trx_start_func = oc_start;
    device->trx_get_stats_func = oc_get_stats;
    device->trx_reset_stats_func = oc_reset_stats;
    device->trx_end_func = oc_end;
    device->trx_stop_func = oc_stop;
    device->trx_set_freq_func = oc_set_freq;
    device->trx_set_gains_func = oc_set_gains;
    device->trx_write_init = oc_write_init;
    device->trx_write_func = oc_write;
    device->trx_read_func = oc_read;
    if (device->openair0_cfg->recplay_mode == RECPLAY_RECORDMODE) {
      std::cerr << "OC device initialized in subframes record mode" << std::endl;
    }

    device->type = USRP_X300_DEV;

    struct {
      int sample_rate;
      int tx_sample_advance;
      double tx_bw;
      double rx_bw;
    } config_table[] = {{245760000, 15, 200e6, 200e6},
			{184320000, 15, 100e6, 100e6},
			{122880000, 15, 80e6, 80e6},
			{92160000, 15, 60e6, 60e6},
			{61440000, 15, 40e6, 40e6},
			{46080000, 15, 40e6, 40e6},
			{30720000, 15, 40e6, 40e6},
			{23040000, 15, 20e6, 20e6},
			{15360000, 15, 10e6, 10e6},
			{7680000, 50, 5e6, 5e6},
			{1920000, 50, 1.25e6, 1.25e6}};
    size_t i = 0;
    for (; i < sizeofArray(config_table); i++)
      if (config_table[i].sample_rate == (int)openair0_cfg[0].sample_rate) {
	openair0_cfg[0].tx_sample_advance = config_table[i].tx_sample_advance;
	openair0_cfg[0].tx_bw = config_table[i].tx_bw;
	openair0_cfg[0].rx_bw = config_table[i].rx_bw;
	break;
      }
    if (i == sizeofArray(config_table)) {
      LOG_E(HW, "unknown sampling rate: %d\n", (int)openair0_cfg[0].sample_rate);
      exit(-1);
    }
    return 0;
  }
}
