/*
 * Licensed by open cells project
 */
#include <iostream>
#include <complex>
#include <fstream>
#include <cmath>
#include <cassert>
#include <queue>
#include <condition_variable>
#include <mutex>

#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#ifdef OAI_INTEGRATION
#include "common_lib.h"
#include "assertions.h"
#else
// #define LOG_E(m, a...) printf(a)
#include "common_lib.h"
#endif
#include "system.h"
#include <sys/resource.h>
#include "common/platform_types.h"
#include "openair1/PHY/sse_intrin.h"
#include "common/utils/LOG/log.h"
#include "common/utils/time_meas.h"

// Thread-safe queue
template <typename T>
class TSQueue {
 private:
  // Underlying queue

  // mutex for thread synchronization
  std::mutex m_mutex;

  // Condition variable for signaling
  std::condition_variable m_cond;

 public:
  std::queue<T> m_queue;
  // Pushes an element to the queue
  void push(T item)
  {
    // Acquire lock
    std::unique_lock<std::mutex> lock(m_mutex);

    // Add item
    m_queue.push(item);

    // Notify one thread that
    // is waiting
    m_cond.notify_one();
  }

  // Pops an element off the queue
  T pop()
  {
    // acquire lock
    std::unique_lock<std::mutex> lock(m_mutex);

    // wait until queue is not empty
    m_cond.wait(lock, [this]() { return !m_queue.empty(); });

    // retrieve item
    T item = m_queue.front();
    m_queue.pop();

    // return item
    return item;
  }
};

#define DEVICE_WRITE_DEFAULT "/dev/xdma0_h2c_0"
#define DEVICE_READ_DEFAULT "/dev/xdma0_c2h_0"
#define NB_BLOCKS_PER_READ 8
#define READ_BLOCK_NB_SAMPLES 2048
#define NB_BLOCKS_PER_WRITE 8
#define WRITE_BLOCK_NB_SAMPLES 2048
#define PKT_HEADER_NB_SAMPLES 7
#define PKT_FOOTER_NB_SAMPLES 1
#define PKT_OVERHEAD_NB_SAMPLES (PKT_HEADER_NB_SAMPLES + PKT_FOOTER_NB_SAMPLES)
static const uint64_t magic_tx = 0xA5A50be3A5A5A5A5LL;
static const uint64_t magic_rx = 0xA5A50be3A5A5A5A5LL;
static const uint32_t magic_footer1 = 0xce11;
static const uint32_t magic_footer2 = 0x5A;
static const uint64_t tx_ahead = WRITE_BLOCK_NB_SAMPLES * NB_BLOCKS_PER_WRITE * 2;

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
  c16_t b[READ_BLOCK_NB_SAMPLES];
  footer_t f;
} __attribute__((packed)) rx_packet_t;

typedef struct {
  headerTx_t h;
  c16_t b[WRITE_BLOCK_NB_SAMPLES];
  // footer_t f;
} __attribute__((packed)) tx_packet_t;

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
  openair0_timestamp_t rx_ts_interface;
  openair0_timestamp_t tx_ts;
  uint64_t gap;
  tx_packet_t *tx_block;
  size_t tx_block_pos;
  TSQueue<tx_packet_t *> *ready_tx;
  TSQueue<uint64_t> *last_rx;
  TSQueue<rx_packet_t *> *read_queue;
  bool first_tx;
  bool rxMagicFound;
  uint seqNum;
  uint lastPpsOffset;
  int  nb_blocks_per_read;
  int remain_samples;
  rx_packet_t *rx_live;
  uint txLate;
  uint txErr;
  uint timerOverflow;
  uint atomicPacket;
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
  oc_state_t *s = (oc_state_t *)arg;
  uint64_t last_rx = s->last_rx->pop();
  do {
    tx_packet_t *p = s->ready_tx->pop();
    if (last_rx + tx_ahead < p->h.timestamp)
      LOG_D(HW, "tx is too ahead, waiting, %lu, %ld\n", p->h.timestamp, p->h.timestamp - last_rx);
    while (last_rx + tx_ahead < p->h.timestamp) {
      last_rx = s->last_rx->pop();
      LOG_D(HW, "pop rx: %lu, rx q sz %lu, tx q sz %lu\n", last_rx, s->last_rx->m_queue.size(), s->ready_tx->m_queue.size());
    }

    size_t wrote = write(s->fd_write, p, sizeof(tx_packet_t) * NB_BLOCKS_PER_WRITE);
    if (wrote != sizeof(tx_packet_t) * NB_BLOCKS_PER_WRITE)
      LOG_E(HW, "write to SDR failed, request: %lu, wrote %ld\n", sizeof(tx_packet_t) * NB_BLOCKS_PER_WRITE, wrote);
    if (wrote < 0)
      LOG_E(HW, "write to %s failed, errno %d:%s\n", s->filename_write, errno, strerror(errno));
    LOG_D(HW, "wrote: %lu\n", p->h.timestamp);
    free(p);
  } while (true);
  return NULL;
}

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
  if (!s->tx_block) {
    s->tx_block = (tx_packet_t *)malloc16(NB_BLOCKS_PER_WRITE * sizeof(tx_packet_t));
  }
  tx_packet_t *ant0 = s->tx_block + s->tx_block_pos;
  ant0->h = (headerTx_t){.control = magic_tx,
                         .packetSeqNum = s->txSeq++,
                         .packetSz = 0x0800,
                         .seqId = 1,
                         .filler = 0x02,
                         .markers = 0xb1,
                         .filler2 = 0xabcd,
                         .txGain = 0x112233,
                         .filler3 = 0xf0,
                         .ppsOffset = 0x28272625,
                         .timestamp = (uint64_t)s->tx_ts};
  for (uint i = 0; i < sz; i++)
    ant0->b[i] = (c16_t){(int16_t)(samples[i].r << 0), (int16_t)(samples[i].i << 0)};
  // memcpy(ant0->b, samples, sz * sizeof(c16_t));
  s->tx_ts += sz;
  s->tx_block_pos++;
  s->tx_count++;
  if (s->tx_block_pos == NB_BLOCKS_PER_WRITE) {
    s->ready_tx->push(s->tx_block);
    s->tx_block_pos = 0;
    s->tx_block = NULL;
  }
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
    LOG_I(HW, "gap of %ld\n", gap);
  else
    LOG_D(HW, ".\n");

  int wr_sz = nsamps;
  // LOG_E(HW, "ask to write %d\n", wr_sz);
  while (wr_sz > 0) {
    int tmp = std::min(wr_sz, WRITE_BLOCK_NB_SAMPLES);
    if (tmp != WRITE_BLOCK_NB_SAMPLES)
      LOG_E(HW, "Error block size: %d\n", nsamps);
    int sz = write_block(s, ((c16_t *)buff[0]) + nsamps - wr_sz, tmp);
    if (sz != tmp)
      LOG_E(HW, "ask to write %d, res is %d\n", tmp, sz);
    wr_sz -= sz;
  }
  s->tx_ts = timestamp + nsamps;
  return nsamps;
}

static void initial_block_align (oc_state_t *s) {
  LOG_I(HW, "Synchronizing rx\n");
  __attribute__ ((aligned(32))) uint32_t b[READ_BLOCK_NB_SAMPLES + PKT_OVERHEAD_NB_SAMPLES];
  int idx = 0;
  int old = 0;
  uint64_t bytes = 0;
  //while (1) {
  ssize_t ret = read(s->fd_read, b, sizeof(b));
  if (ret != sizeof(b)) {
    LOG_E(HW, "Error reading %ld bytes: %ld (%s)\n", sizeof(b), ret, strerror(errno));
    usleep(10000);
    return;
  }
  int i;
  headerRx_t *rx = NULL;
  for (i = 0; i < READ_BLOCK_NB_SAMPLES + PKT_HEADER_NB_SAMPLES; i++)
    if (b[i] == (magic_rx & UINT32_MAX) && b[i + 1] == ((magic_rx >> 32) & UINT32_MAX)) {
      LOG_D(HW,
            "found first magic at %d (inblock: %d), dist: %d, bytes %lu\n",
            idx * (READ_BLOCK_NB_SAMPLES + PKT_OVERHEAD_NB_SAMPLES) + i,
            i,
            idx * (READ_BLOCK_NB_SAMPLES + PKT_OVERHEAD_NB_SAMPLES) + i - old,
            bytes);
      rx = (headerRx_t *)(b + i);
      dumpHD("first header:", *rx);
      break;
    }
  if (i == (READ_BLOCK_NB_SAMPLES + PKT_HEADER_NB_SAMPLES)) {
    LOG_E(HW, "%%error magic not found\n");
    return;
  }
  ret = read(s->fd_read, b, i * sizeof(*b));
  s->seqNum = rx->packetSeqNum + 1;
  s->rx_timestamp = (int64_t)rx->timestamp + rx->packetSz;
  s->rx_ts_interface = rx->timestamp;
  s->rx_count = 1;
}

static bool get_blocks(oc_state_t *s, rx_packet_t *p)
{
  static struct timespec last_second={}, origin={};
  static struct timespec now={};
  static uint64_t tot_samples= 0;

  int readSz = sizeof(*s->rx_live) * s->nb_blocks_per_read;
  ssize_t ret = read(s->fd_read, p, readSz);
  if (ret != readSz || p[0].h.control != magic_rx) {
    LOG_E(HW, "Error reading header asked for %d bytes, got %ld, magic: %lx\n", readSz, ret, p[0].h.control);
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
    LOG_I(HW,
          "driver avg read rate:%f\n errors during last second: txLate %u, txSeqerr %u, timerOverflow %u, atomicPacket %u, present "
          "tx seq %u\n\n ",
          (float)tot_samples / (now.tv_sec * 1000000 - origin.tv_sec * 1000000 + now.tv_nsec / 1000.0 - origin.tv_nsec / 1000.0),
          s->txLate,
          s->txErr,
          s->timerOverflow,
          s->atomicPacket,
          s->txSeq);
    s->txLate = s->txErr = s->timerOverflow = s->atomicPacket = 0;
    last_second.tv_sec++;
  }

  for (int i = 0; i <s->nb_blocks_per_read ; i++) {
    if (s->rx_timestamp != (int64_t)p[i].h.timestamp)
      LOG_W(HW,
            "expected ts: %lu got %lu, diff %ld, seq num %d, atomicPacket %d, timerOverflow %d\n",
            s->rx_timestamp,
            p[i].h.timestamp,
            (int64_t)p[i].h.timestamp - s->rx_timestamp,
            s->seqNum,
            (~p[i].f.atomicPacket & 0x01),
            p[i].f.timerOverflow);
    s->rx_timestamp=p[i].h.timestamp + READ_BLOCK_NB_SAMPLES;
    /*
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
    */
    if (p[i].h.TXlate) {
      LOG_D(HW, "TXlate\n");
      s->txLate++;
    }
    if (p[i].h.TXseqerr) {
      LOG_W(HW, "TXseqerr, current tx seq is %u\n", s->txSeq);
      s->txErr++;
    }
    if (p[i].f.control1 != magic_footer1 || p[i].f.control2 != magic_footer2)
      LOG_W(HW, "footer error\n");
    if (p[i].f.timerOverflow) {
      LOG_D(HW, "timerOverflow\n");
      s->timerOverflow++;
    }
    if (p[i].f.atomicPacket) {
      s->atomicPacket++;
      LOG_D(HW, "Not atomic\n");
    }

    static int last_filler = 0;
    if (last_filler != p[i].f.filler)
      LOG_I(HW, "filler changed to %x\n", p[i].f.filler);
    last_filler = p[i].f.filler;
    if (s->seqNum % 65536 != p[i].h.packetSeqNum) {
      LOG_W(HW,
            "expected rx packet sequence number %u got %u, diff %d\n",
            s->seqNum,
            p[i].h.packetSeqNum,
            p[i].h.packetSeqNum - s->seqNum);
      s->seqNum = p[i].h.packetSeqNum;
      //s->rx_count = -1;
      //return false;
    }
    s->seqNum++;
  }
  s->last_rx->push(s->rx_timestamp);
  LOG_D(HW, "read: %lu\n", s->rx_timestamp);
  return true;
}

static int oc_read(openair0_device_t *device, openair0_timestamp_t *ptimestamp, void **buff, int nsamps, int cc)
{
  oc_state_t *s = (oc_state_t *)device->priv;
  c16_t **output=(c16_t**)buff;
  int remain_to_get=nsamps;
  while (remain_to_get > 0) {
    while (s->remain_samples > 0 && remain_to_get > 0) {
      if (s->gap) {
        c16_t *out = output[0] + nsamps - remain_to_get;
        while (s->gap && remain_to_get) {
          *out++ = {};
          remain_to_get--;
          s->gap--;
          s->rx_ts_interface++;
        }
      }
      rx_packet_t *last_rx = s->rx_live;
      int nb_samples_per_packet = sizeof(last_rx[0].b) / sizeof(last_rx[0].b[0]);
      int nb_samples=nb_samples_per_packet*s->nb_blocks_per_read;
      int consumed_samples = nb_samples - s->remain_samples;
      int nb_consumed_samples_in_block= consumed_samples%nb_samples_per_packet;
      int bloc = consumed_samples / nb_samples_per_packet;
      rx_packet_t *cur_pkt = last_rx + bloc;
      if (nb_consumed_samples_in_block == 0 && cur_pkt->h.timestamp != (uint64_t)s->rx_ts_interface) {
        s->gap = cur_pkt->h.timestamp - s->rx_ts_interface;
        if (s->gap > 0 && s->gap < 1228800) {
          LOG_W(HW, "gap of %ld, we fill \n", s->gap);
          continue;
        } else {
          LOG_W(HW, "gap of %ld, we break timestamp sequence %lu !=  %lu\n", s->gap, s->rx_ts_interface, cur_pkt->h.timestamp);
          s->rx_ts_interface = cur_pkt->h.timestamp;
          s->gap = 0;
        }
      }
      int remaing_samples_in_block= nb_samples_per_packet-nb_consumed_samples_in_block;
      int toCopy=std::min(remaing_samples_in_block,remain_to_get );
      memcpy(output[0] + nsamps - remain_to_get, cur_pkt->b + nb_consumed_samples_in_block, toCopy * sizeof(cur_pkt->b[0]));
      s->remain_samples-=toCopy;
      remain_to_get -= toCopy;
      s->rx_ts_interface += toCopy;
    }
    if(  remain_to_get > 0 ) {
      free(s->rx_live);
      s->rx_live = s->read_queue->pop();
      s->remain_samples = sizeof(s->rx_live->b) * s->nb_blocks_per_read / sizeof(*s->rx_live->b);
    }
  }
  *ptimestamp = s->rx_ts_interface;
  return nsamps;
}

static int oc_set_freq(openair0_device_t *device, openair0_config_t *openair0_cfg)
{
  // oc_state_t *s = (oc_state_t *)device->priv;
  LOG_I(HW,
        "Setting TX Freq %f, RX Freq %f, tune_offset: %f\n",
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
  s->nb_blocks_per_read = NB_BLOCKS_PER_READ;
  s->rx_live = (rx_packet_t *)malloc16(s->nb_blocks_per_read * sizeof(*s->rx_live));
  s->ready_tx = new TSQueue<tx_packet_t *>;
  s->read_queue = new TSQueue<rx_packet_t *>;
  s->last_rx = new TSQueue<uint64_t>;
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
  pthread_t w_thread;
  threadCreate(&w_thread, write_thread, s, (char *)"write_thr", -1, OAI_PRIORITY_RT);
  pthread_t r_thread;
  threadCreate(&r_thread, read_thread, s, (char *)"read_thr", -1, OAI_PRIORITY_RT);
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
  // oc_state_t *s = (oc_state_t *)device->priv;
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
