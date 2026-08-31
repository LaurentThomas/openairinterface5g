/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#if defined(__x86_64__) || defined(__i386__)
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>
#include "../sse_intrin.h"
#include "assertions.h"
#define OAIDFTS_MAIN
#include "tools_defs.h"
#include "LOG/log.h"
#include <pthread.h>

static pthread_mutex_t sr_twiddle_mutex = PTHREAD_MUTEX_INITIALIZER;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Q15_INV_SQRT2 ((int16_t)23170)
#define Q15_INV_SQRT3 ((int16_t)18919)
#define Q15_INV_SQRT8 ((int16_t)11585)
#define Q15_HALF_SQRT3 ((int16_t)28378)

#define Q15_INV_2SQRT3 ((int16_t)9459) /* 1 / (2 * sqrt(3)) */

#define Q15_HALF ((int16_t)16384) /* 1 / 2 */

#define Q15_ONE ((int16_t)32767)

#define DFT32_Q15_ONE ((int16_t)32767)

#define DFT32_Q15_COS_PI_16 ((int16_t)32138)
#define DFT32_Q15_SIN_PI_16 ((int16_t)6393)

#define DFT32_Q15_COS_3PI_16 ((int16_t)27246)
#define DFT32_Q15_SIN_3PI_16 ((int16_t)18205)

#define Q15_COS_PI_8 ((int16_t)30274) /* cos(pi/8) */
#define Q15_SIN_PI_8 ((int16_t)12540) /* sin(pi/8) */

#define Q15_SQRT3_OVER_2 ((int16_t)28378) /* sqrt(3)/2 */

#define Q15_INV_SQRT5 ((int16_t)14654)
#define Q15_INV_SQRT6 ((int16_t)13377) /* 1 / sqrt(6) */
#define Q15_INV_SQRT30 ((int16_t)5982) /* 1 / sqrt(30) */
#define Q15_INV_SQRT12 ((int16_t)9459) /* 1 / sqrt(12) */
#define Q15_INV_SQRT15 ((int16_t)8460) /* 1 / sqrt(15) */
#define Q15_INV_SQRT18 ((int16_t)7723) /* 1 / sqrt(18) */
#define Q15_INV_SQRT27 ((int16_t)6306) /* 1 / sqrt(27) */
#define Q15_INV_SQRT32 ((int16_t)5793) /* 1 / sqrt(32) */
#define Q15_INV_SQRT360 ((int16_t)1727) /* 1 / sqrt(360) */
#define Q15_INV_SQRT24 ((int16_t)6689) /* 1 / sqrt(24) */
#define Q15_INV_SQRT600 ((int16_t)1338) /* 1 / sqrt(600) */
#define Q15_INV_SQRT720 ((int16_t)1221) /* 1 / sqrt(720) */

#define Q15_COS_2PI_5 ((int16_t)10126) /* cos(2pi/5)  */
#define Q15_COS_4PI_5 ((int16_t)-26510) /* cos(4pi/5)  */
#define Q15_SIN_2PI_5 ((int16_t)31163) /* sin(2pi/5)  */
#define Q15_SIN_4PI_5 ((int16_t)19260) /* sin(4pi/5)  */

#define SR_MAX_LOG2 25
#define X86_DFT_RECURSION_MAX_DEPTH 32

#define STACK_MAX_N 1024

static inline __m128i load4_complex_strided_c16(const c16_t *src, int stride, int base)
{
  const c16_t a = src[(base + 0) * stride];
  const c16_t b = src[(base + 1) * stride];
  const c16_t c = src[(base + 2) * stride];
  const c16_t d = src[(base + 3) * stride];

  return _mm_setr_epi16(a.r, a.i, b.r, b.i, c.r, c.i, d.r, d.i);
}

static inline __m128i pack1_complex_lane0_c16(const c16_t a)
{
  return _mm_setr_epi16(a.r, a.i, 0, 0, 0, 0, 0, 0);
}

static inline int16_t sat_i16(long v)
{
  if (v > INT16_MAX)
    return INT16_MAX;

  if (v < INT16_MIN)
    return INT16_MIN;

  return (int16_t)v;
}

static inline int log2_int(unsigned int N)
{
  return __builtin_ctz(N);
}
static void *aligned_malloc64(size_t size)
{
  void *ptr = NULL;

  if (posix_memalign(&ptr, 64, size) != 0) {
    return NULL;
  }

  return ptr;
}

typedef struct {
  c16_t *ptr;
  size_t elems;
} x86_dft_tls_buffer_t;

static __thread x86_dft_tls_buffer_t g_x86_dft_q15_work;
static __thread x86_dft_tls_buffer_t g_x86_dft_split_work;
static __thread x86_dft_tls_buffer_t g_x86_dft_recursive_work[X86_DFT_RECURSION_MAX_DEPTH];
static __thread unsigned int g_x86_dft_recursive_depth;

static c16_t *x86_dft_tls_buffer_get(x86_dft_tls_buffer_t *buf, size_t need)
{
  if (need <= buf->elems)
    return buf->ptr;

  c16_t *p = aligned_malloc64(need * sizeof(*p));
  if (!p)
    return NULL;

  free(buf->ptr);
  buf->ptr = p;
  buf->elems = need;
  return p;
}

static inline c16_t *x86_dft_tls_work(size_t need)
{
  return x86_dft_tls_buffer_get(&g_x86_dft_q15_work, need);
}

static inline c16_t *x86_dft_tls_split_work(size_t need)
{
  return x86_dft_tls_buffer_get(&g_x86_dft_split_work, need);
}

static inline c16_t *x86_dft_recursive_work_acquire(size_t need)
{
  AssertFatal(g_x86_dft_recursive_depth < X86_DFT_RECURSION_MAX_DEPTH,
              "x86 DFT recursion depth exceeded: %u\n",
              g_x86_dft_recursive_depth);
  x86_dft_tls_buffer_t *buf = &g_x86_dft_recursive_work[g_x86_dft_recursive_depth++];
  c16_t *p = x86_dft_tls_buffer_get(buf, need);
  if (!p)
    g_x86_dft_recursive_depth--;
  return p;
}

static inline void x86_dft_recursive_work_release(void)
{
  AssertFatal(g_x86_dft_recursive_depth > 0, "x86 DFT recursion scratch underflow\n");
  g_x86_dft_recursive_depth--;
}

static inline int is_power_of_two_int(int x)
{
  return x > 0 && ((x & (x - 1)) == 0);
}

static inline simde__m256i c16_mul_q15_simd256(simde__m256i x, simde__m256i w_re_negim, simde__m256i w_im_re)
{
  simde__m256i round = simde_mm256_set1_epi32(1 << 14);

  simde__m256i re32 = simde_mm256_madd_epi16(x, w_re_negim);
  simde__m256i im32 = simde_mm256_madd_epi16(x, w_im_re);

  re32 = simde_mm256_srai_epi32(simde_mm256_add_epi32(re32, round), 15);
  im32 = simde_mm256_srai_epi32(simde_mm256_add_epi32(im32, round), 15);

  simde__m256i packed = simde_mm256_packs_epi32(re32, im32);

  const simde__m256i mask = simde_mm256_set_epi8(15,
                                                 14,
                                                 7,
                                                 6,
                                                 13,
                                                 12,
                                                 5,
                                                 4,
                                                 11,
                                                 10,
                                                 3,
                                                 2,
                                                 9,
                                                 8,
                                                 1,
                                                 0,
                                                 15,
                                                 14,
                                                 7,
                                                 6,
                                                 13,
                                                 12,
                                                 5,
                                                 4,
                                                 11,
                                                 10,
                                                 3,
                                                 2,
                                                 9,
                                                 8,
                                                 1,
                                                 0);

  return simde_mm256_shuffle_epi8(packed, mask);
}

static inline int16_t q15_from_float(float x)
{
  return sat_i16((int32_t)lrintf(32767.0f * x));
}

static inline __m128i swap_complex_pairs_i16_128(__m128i a)
{
  const __m128i shuf = _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

  return _mm_shuffle_epi8(a, shuf);
}

static inline __m128i complex_mul4_prepack_q15_128(__m128i a, __m128i w_re_re, __m128i w_im_signed)
{
  const __m128i a_swapped = swap_complex_pairs_i16_128(a);

  const __m128i prod_re = _mm_mulhrs_epi16(a, w_re_re);
  const __m128i prod_im = _mm_mulhrs_epi16(a_swapped, w_im_signed);

  return _mm_adds_epi16(prod_re, prod_im);
}

static inline __m128i complex_mul4_bcast_q15_128(__m128i a, int16_t wr, int16_t wi)
{
  const __m128i w_re_re = _mm_set1_epi16(wr);

  const __m128i w_im_signed = _mm_setr_epi16(-wi, wi, -wi, wi, -wi, wi, -wi, wi);

  return complex_mul4_prepack_q15_128(a, w_re_re, w_im_signed);
}

static inline __m128i mul_q15_128(__m128i z)
{
  /*
   * j * (r + ji) = -i + jr
   *
   * [r i] -> [-i r]
   */
  const __m128i swapped = swap_complex_pairs_i16_128(z);

  const __m128i sign = _mm_setr_epi16(-1, 1, -1, 1, -1, 1, -1, 1);

  return _mm_sign_epi16(swapped, sign);
}

static inline __m128i mul_minus_q15_128(__m128i z)
{
  /*
   * -j * (r + ji) = i - jr
   *
   * [r i] -> [i -r]
   */
  const __m128i swapped = swap_complex_pairs_i16_128(z);

  const __m128i sign = _mm_setr_epi16(1, -1, 1, -1, 1, -1, 1, -1);

  return _mm_sign_epi16(swapped, sign);
}

static inline __m128i q15_mul_i16_128(__m128i x, int16_t q15)
{
  return _mm_mulhrs_epi16(x, _mm_set1_epi16(q15));
}

typedef enum { DFT_DIR_FORWARD = -1, DFT_DIR_INVERSE = 1 } dft_dir_t;

static inline __m128i mul_minus_j_dir_i16_128(__m128i z, dft_dir_t dir)
{
  return (dir == DFT_DIR_FORWARD) ? mul_minus_q15_128(z) : mul_q15_128(z);
}

static inline __m128i mul_plus_j_dir_i16_128(__m128i z, dft_dir_t dir)
{
  return (dir == DFT_DIR_FORWARD) ? mul_q15_128(z) : mul_minus_q15_128(z);
}

static inline __m128i twiddle_im_dir_128(__m128i w_im_signed, dft_dir_t dir)
{
  return (dir == DFT_DIR_FORWARD) ? w_im_signed : _mm_sub_epi16(_mm_setzero_si128(), w_im_signed);
}

static inline int16_t twiddle_im_scalar_dir_i16(int16_t wi_forward, dft_dir_t dir)
{
  return (dir == DFT_DIR_FORWARD) ? wi_forward : sat_i16(-(long)wi_forward);
}

static inline __m256i swap_complex_pairs_i16_256(__m256i a)
{
  const __m256i shuf = _mm256_setr_epi8(2,
                                        3,
                                        0,
                                        1,
                                        6,
                                        7,
                                        4,
                                        5,
                                        10,
                                        11,
                                        8,
                                        9,
                                        14,
                                        15,
                                        12,
                                        13,

                                        2,
                                        3,
                                        0,
                                        1,
                                        6,
                                        7,
                                        4,
                                        5,
                                        10,
                                        11,
                                        8,
                                        9,
                                        14,
                                        15,
                                        12,
                                        13);

  return _mm256_shuffle_epi8(a, shuf);
}

static inline __m256i mul_j_i16_256(__m256i z)
{
  const __m256i swapped = swap_complex_pairs_i16_256(z);

  const __m256i sign = _mm256_setr_epi16(-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1);

  return _mm256_sign_epi16(swapped, sign);
}

static inline __m256i mul_minus_j_i16_256(__m256i z)
{
  const __m256i swapped = swap_complex_pairs_i16_256(z);

  const __m256i sign = _mm256_setr_epi16(+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1);

  return _mm256_sign_epi16(swapped, sign);
}

static inline __m256i mul_minus_j_dir_i16_256(__m256i z, dft_dir_t dir)
{
  return (dir == DFT_DIR_FORWARD) ? mul_minus_j_i16_256(z) : mul_j_i16_256(z);
}

static inline __m256i mul_plus_j_dir_i16_256(__m256i z, dft_dir_t dir)
{
  return (dir == DFT_DIR_FORWARD) ? mul_j_i16_256(z) : mul_minus_j_i16_256(z);
}

static inline void
transpose8_complex_i16_256(__m256i *r0, __m256i *r1, __m256i *r2, __m256i *r3, __m256i *r4, __m256i *r5, __m256i *r6, __m256i *r7)
{
  const __m256i a = *r0;
  const __m256i b = *r1;
  const __m256i c = *r2;
  const __m256i d = *r3;
  const __m256i e = *r4;
  const __m256i f = *r5;
  const __m256i g = *r6;
  const __m256i h = *r7;

  const __m256i t0 = _mm256_unpacklo_epi32(a, b);
  const __m256i t1 = _mm256_unpackhi_epi32(a, b);
  const __m256i t2 = _mm256_unpacklo_epi32(c, d);
  const __m256i t3 = _mm256_unpackhi_epi32(c, d);
  const __m256i t4 = _mm256_unpacklo_epi32(e, f);
  const __m256i t5 = _mm256_unpackhi_epi32(e, f);
  const __m256i t6 = _mm256_unpacklo_epi32(g, h);
  const __m256i t7 = _mm256_unpackhi_epi32(g, h);

  const __m256i s0 = _mm256_unpacklo_epi64(t0, t2);
  const __m256i s1 = _mm256_unpackhi_epi64(t0, t2);
  const __m256i s2 = _mm256_unpacklo_epi64(t1, t3);
  const __m256i s3 = _mm256_unpackhi_epi64(t1, t3);

  const __m256i s4 = _mm256_unpacklo_epi64(t4, t6);
  const __m256i s5 = _mm256_unpackhi_epi64(t4, t6);
  const __m256i s6 = _mm256_unpacklo_epi64(t5, t7);
  const __m256i s7 = _mm256_unpackhi_epi64(t5, t7);

  *r0 = _mm256_permute2x128_si256(s0, s4, 0x20);
  *r1 = _mm256_permute2x128_si256(s1, s5, 0x20);
  *r2 = _mm256_permute2x128_si256(s2, s6, 0x20);
  *r3 = _mm256_permute2x128_si256(s3, s7, 0x20);

  *r4 = _mm256_permute2x128_si256(s0, s4, 0x31);
  *r5 = _mm256_permute2x128_si256(s1, s5, 0x31);
  *r6 = _mm256_permute2x128_si256(s2, s6, 0x31);
  *r7 = _mm256_permute2x128_si256(s3, s7, 0x31);
}

static inline __m256i complex_mul8_prepack_q15_256(__m256i a, __m256i w_re_re, __m256i w_im_signed)
{
  const __m256i a_swapped = swap_complex_pairs_i16_256(a);

  const __m256i prod_re = _mm256_mulhrs_epi16(a, w_re_re);
  const __m256i prod_im = _mm256_mulhrs_epi16(a_swapped, w_im_signed);

  return _mm256_adds_epi16(prod_re, prod_im);
}

static void dft_mixed_radix_c16_scaled_strided(const c16_t *src, int stride, c16_t *dst, int N, dft_dir_t dir);
static void dft_mixed_radix_c16_scaled(const c16_t *src, c16_t *dst, int N, dft_dir_t dir);
static inline void dft4_void(const c16_t *src, c16_t *dst, dft_dir_t dir);
static inline void dft8_avx(const c16_t *src, c16_t *dst, dft_dir_t dir);
static inline void dft8_strided_q15_128(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir);
static inline void dft16_q15_128(const c16_t *src, c16_t *dst, dft_dir_t dir);
static inline void dft32_q15_128(const c16_t *src, c16_t *dst, dft_dir_t dir);
static inline void dft32_q15_256_contiguous(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft96_radix12_pfa_leaf8(const c16_t*,c16_t*,dft_dir_t);
static void dft120_radix15_pfa_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft144_radix18_leaf8_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir);
static void dft192_radix12_pfa_leaf16(const c16_t*,c16_t*,dft_dir_t);
static void dft216_radix27_leaf8_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft240_radix15_pfa_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static __attribute__((always_inline)) inline void dft6_pfa8_q15_256(__m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i y[6], dft_dir_t dir);
static void dft384_radix24_pfa_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft432_radix27_leaf16_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft480_radix30_pfa_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);


static void dft864_radix27_leaf32_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft1152_radix18_leaf64_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir);
static void dft1536_radix24_pfa_leaf64(const c16_t*,c16_t*,dft_dir_t);
static void dft1728_radix27_leaf64_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft1920_radix30_pfa_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft2160_radix30_leaf72x8_direct(const c16_t *src, c16_t *dst, dft_dir_t dir);
static inline void dft16_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir);
static inline void dft32_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir);
static void dft256_radix16_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft512_radix8_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static void dft1024_radix16_selected(const c16_t *src, c16_t *dst, dft_dir_t dir);
static size_t dft_power2_mixed_large_work_len(int N);
static void dft_power2_mixed_large_core(const c16_t *src, c16_t *dst, int N, dft_dir_t dir, c16_t *work);
static void radix3_pow2_selected(const c16_t *src, c16_t *dst, int N, dft_dir_t dir);
static void selected_q15_twiddles_init(void);

typedef struct {
  int r3_q15_blocks;
  unsigned char r3_q15_ready;
  __m128i *r3_q15_storage;
  __m128i *r3_q15_w1_re;
  __m128i *r3_q15_w1_im;
  __m128i *r3_q15_w2_re;
  __m128i *r3_q15_w2_im;
  __m128i *r3_q15_w1_re_inv;
  __m128i *r3_q15_w1_im_inv;
  __m128i *r3_q15_w2_re_inv;
  __m128i *r3_q15_w2_im_inv;
} r3_twiddle_t;

typedef struct {
  int r5_q15_blocks;
  unsigned char r5_q15_ready;
  __m128i *r5_q15_storage;
  __m128i *r5_q15_w1_re;
  __m128i *r5_q15_w1_im;
  __m128i *r5_q15_w2_re;
  __m128i *r5_q15_w2_im;
  __m128i *r5_q15_w3_re;
  __m128i *r5_q15_w3_im;
  __m128i *r5_q15_w4_re;
  __m128i *r5_q15_w4_im;
  __m128i *r5_q15_w1_re_inv;
  __m128i *r5_q15_w1_im_inv;
  __m128i *r5_q15_w2_re_inv;
  __m128i *r5_q15_w2_im_inv;
  __m128i *r5_q15_w3_re_inv;
  __m128i *r5_q15_w3_im_inv;
  __m128i *r5_q15_w4_re_inv;
  __m128i *r5_q15_w4_im_inv;
} r5_twiddle_t;

typedef struct {
  int q15_blocks;
  unsigned char q15_ready;
  __m256i *q15_re;
  __m256i *q15_im;
  __m256i *q15_re_inv;
  __m256i *q15_im_inv;
} power2_twiddle_t;

/* DFT64/128 are the only users of these fixed AVX2 tables.  Keep them once,
 * instead of embedding ~2 KiB of unused storage in every generic N entry. */
static __m256i g_dft64_re[2][8] __attribute__((aligned(64)));
static __m256i g_dft64_im[2][8] __attribute__((aligned(64)));
static __m256i g_dft128_re[2][8] __attribute__((aligned(64)));
static __m256i g_dft128_im[2][8] __attribute__((aligned(64)));

/* Static metadata exists only for coefficient sets used by the x86 kernels. */
#define R3_TWIDDLE_SLOT(N_) static r3_twiddle_t g_r3_twiddle_##N_ __attribute__((section(".bss.oai_dft_twmeta")))
R3_TWIDDLE_SLOT(24);
R3_TWIDDLE_SLOT(36);
R3_TWIDDLE_SLOT(48);
R3_TWIDDLE_SLOT(72);
R3_TWIDDLE_SLOT(96);
R3_TWIDDLE_SLOT(108);
R3_TWIDDLE_SLOT(120);
R3_TWIDDLE_SLOT(144);
R3_TWIDDLE_SLOT(192);
R3_TWIDDLE_SLOT(216);
R3_TWIDDLE_SLOT(240);
R3_TWIDDLE_SLOT(288);
R3_TWIDDLE_SLOT(324);
R3_TWIDDLE_SLOT(384);
R3_TWIDDLE_SLOT(432);
R3_TWIDDLE_SLOT(480);
R3_TWIDDLE_SLOT(576);
R3_TWIDDLE_SLOT(648);
R3_TWIDDLE_SLOT(768);
R3_TWIDDLE_SLOT(864);
R3_TWIDDLE_SLOT(960);
R3_TWIDDLE_SLOT(972);
R3_TWIDDLE_SLOT(1152);
R3_TWIDDLE_SLOT(1200);
R3_TWIDDLE_SLOT(1296);
R3_TWIDDLE_SLOT(1536);
R3_TWIDDLE_SLOT(1728);
R3_TWIDDLE_SLOT(1920);
R3_TWIDDLE_SLOT(1944);
R3_TWIDDLE_SLOT(2304);
R3_TWIDDLE_SLOT(2400);
R3_TWIDDLE_SLOT(2592);
R3_TWIDDLE_SLOT(2916);
R3_TWIDDLE_SLOT(3072);
R3_TWIDDLE_SLOT(6144);
R3_TWIDDLE_SLOT(12288);
R3_TWIDDLE_SLOT(18432);
R3_TWIDDLE_SLOT(24576);
R3_TWIDDLE_SLOT(36864);
R3_TWIDDLE_SLOT(49152);
R3_TWIDDLE_SLOT(98304);
R3_TWIDDLE_SLOT(1572864);
#undef R3_TWIDDLE_SLOT

#define R5_TWIDDLE_SLOT(N_) static r5_twiddle_t g_r5_twiddle_##N_ __attribute__((section(".bss.oai_dft_twmeta")))
R5_TWIDDLE_SLOT(40);
R5_TWIDDLE_SLOT(60);
R5_TWIDDLE_SLOT(80);
R5_TWIDDLE_SLOT(120);
R5_TWIDDLE_SLOT(160);
R5_TWIDDLE_SLOT(180);
R5_TWIDDLE_SLOT(300);
R5_TWIDDLE_SLOT(320);
R5_TWIDDLE_SLOT(360);
R5_TWIDDLE_SLOT(400);
R5_TWIDDLE_SLOT(540);
R5_TWIDDLE_SLOT(600);
R5_TWIDDLE_SLOT(640);
R5_TWIDDLE_SLOT(720);
R5_TWIDDLE_SLOT(800);
R5_TWIDDLE_SLOT(900);
R5_TWIDDLE_SLOT(1080);
R5_TWIDDLE_SLOT(1440);
R5_TWIDDLE_SLOT(1500);
R5_TWIDDLE_SLOT(1620);
R5_TWIDDLE_SLOT(1800);
R5_TWIDDLE_SLOT(2160);
R5_TWIDDLE_SLOT(2700);
R5_TWIDDLE_SLOT(2880);
R5_TWIDDLE_SLOT(3000);
R5_TWIDDLE_SLOT(3240);
#undef R5_TWIDDLE_SLOT

#define POWER2_TWIDDLE_SLOT(N_) static power2_twiddle_t g_power2_twiddle_##N_ __attribute__((section(".bss.oai_dft_twmeta")))
POWER2_TWIDDLE_SLOT(2048);
POWER2_TWIDDLE_SLOT(4096);
POWER2_TWIDDLE_SLOT(8192);
POWER2_TWIDDLE_SLOT(32768);
POWER2_TWIDDLE_SLOT(65536);
#undef POWER2_TWIDDLE_SLOT

static inline r3_twiddle_t *r3_twiddle_slot(int N)
{
  switch (N) {
    case 24: return &g_r3_twiddle_24;
    case 36: return &g_r3_twiddle_36;
    case 48: return &g_r3_twiddle_48;
    case 72: return &g_r3_twiddle_72;
    case 96: return &g_r3_twiddle_96;
    case 108: return &g_r3_twiddle_108;
    case 120: return &g_r3_twiddle_120;
    case 144: return &g_r3_twiddle_144;
    case 192: return &g_r3_twiddle_192;
    case 216: return &g_r3_twiddle_216;
    case 240: return &g_r3_twiddle_240;
    case 288: return &g_r3_twiddle_288;
    case 324: return &g_r3_twiddle_324;
    case 384: return &g_r3_twiddle_384;
    case 432: return &g_r3_twiddle_432;
    case 480: return &g_r3_twiddle_480;
    case 576: return &g_r3_twiddle_576;
    case 648: return &g_r3_twiddle_648;
    case 768: return &g_r3_twiddle_768;
    case 864: return &g_r3_twiddle_864;
    case 960: return &g_r3_twiddle_960;
    case 972: return &g_r3_twiddle_972;
    case 1152: return &g_r3_twiddle_1152;
    case 1200: return &g_r3_twiddle_1200;
    case 1296: return &g_r3_twiddle_1296;
    case 1536: return &g_r3_twiddle_1536;
    case 1728: return &g_r3_twiddle_1728;
    case 1920: return &g_r3_twiddle_1920;
    case 1944: return &g_r3_twiddle_1944;
    case 2304: return &g_r3_twiddle_2304;
    case 2400: return &g_r3_twiddle_2400;
    case 2592: return &g_r3_twiddle_2592;
    case 2916: return &g_r3_twiddle_2916;
    case 3072: return &g_r3_twiddle_3072;
    case 6144: return &g_r3_twiddle_6144;
    case 12288: return &g_r3_twiddle_12288;
    case 18432: return &g_r3_twiddle_18432;
    case 24576: return &g_r3_twiddle_24576;
    case 36864: return &g_r3_twiddle_36864;
    case 49152: return &g_r3_twiddle_49152;
    case 98304: return &g_r3_twiddle_98304;
    case 1572864: return &g_r3_twiddle_1572864;
    default: return NULL;
  }
}

static inline r5_twiddle_t *r5_twiddle_slot(int N)
{
  switch (N) {
    case 40: return &g_r5_twiddle_40;
    case 60: return &g_r5_twiddle_60;
    case 80: return &g_r5_twiddle_80;
    case 120: return &g_r5_twiddle_120;
    case 160: return &g_r5_twiddle_160;
    case 180: return &g_r5_twiddle_180;
    case 300: return &g_r5_twiddle_300;
    case 320: return &g_r5_twiddle_320;
    case 360: return &g_r5_twiddle_360;
    case 400: return &g_r5_twiddle_400;
    case 540: return &g_r5_twiddle_540;
    case 600: return &g_r5_twiddle_600;
    case 640: return &g_r5_twiddle_640;
    case 720: return &g_r5_twiddle_720;
    case 800: return &g_r5_twiddle_800;
    case 900: return &g_r5_twiddle_900;
    case 1080: return &g_r5_twiddle_1080;
    case 1440: return &g_r5_twiddle_1440;
    case 1500: return &g_r5_twiddle_1500;
    case 1620: return &g_r5_twiddle_1620;
    case 1800: return &g_r5_twiddle_1800;
    case 2160: return &g_r5_twiddle_2160;
    case 2700: return &g_r5_twiddle_2700;
    case 2880: return &g_r5_twiddle_2880;
    case 3000: return &g_r5_twiddle_3000;
    case 3240: return &g_r5_twiddle_3240;
    default: return NULL;
  }
}

static inline power2_twiddle_t *power2_twiddle_slot(int N)
{
  switch (N) {
    case 2048: return &g_power2_twiddle_2048;
    case 4096: return &g_power2_twiddle_4096;
    case 8192: return &g_power2_twiddle_8192;
    case 32768: return &g_power2_twiddle_32768;
    case 65536: return &g_power2_twiddle_65536;
    default: return NULL;
  }
}


static inline __m128i pack4_twiddle_q15_re_re_scaled(int k0, int mul, int N, float scale)
{
  int16_t v[8];
  for (int j = 0; j < 4; j++) {
    const int k = (mul * (k0 + j)) % N;
    const float theta = 2.0f * (float)M_PI * (float)k / (float)N;
    const int16_t wr = q15_from_float(cosf(theta) * scale);
    v[2 * j + 0] = wr;
    v[2 * j + 1] = wr;
  }
  return _mm_loadu_si128((const __m128i *)v);
}

static inline __m128i pack4_twiddle_q15_im_signed_scaled(int k0, int mul, int N, float scale, dft_dir_t dir)
{
  int16_t v[8];
  for (int j = 0; j < 4; j++) {
    const int k = (mul * (k0 + j)) % N;
    const float theta = 2.0f * (float)M_PI * (float)k / (float)N;
    const int16_t wi = q15_from_float((float)dir * sinf(theta) * scale);
    v[2 * j + 0] = -wi;
    v[2 * j + 1] = wi;
  }
  return _mm_loadu_si128((const __m128i *)v);
}

static inline __m256i pack8_twiddle_q15_re_re_scaled(int k0, int mul, int N, float scale)
{
  int16_t v[16] __attribute__((aligned(32)));
  for (int j = 0; j < 8; j++) {
    const int k = (mul * (k0 + j)) % N;
    const float theta = 2.0f * (float)M_PI * (float)k / (float)N;
    const int16_t wr = q15_from_float(cosf(theta) * scale);
    v[2 * j] = wr;
    v[2 * j + 1] = wr;
  }
  return _mm256_load_si256((const __m256i *)v);
}

static inline __m256i pack8_twiddle_q15_im_signed_scaled(int k0, int mul, int N, float scale, dft_dir_t dir)
{
  int16_t v[16] __attribute__((aligned(32)));
  for (int j = 0; j < 8; j++) {
    const int k = (mul * (k0 + j)) % N;
    const float theta = 2.0f * (float)M_PI * (float)k / (float)N;
    const int16_t wi = q15_from_float((float)dir * sinf(theta) * scale);
    v[2 * j] = -wi;
    v[2 * j + 1] = wi;
  }
  return _mm256_load_si256((const __m256i *)v);
}

static inline __m256i pack8_twiddle_q15_re_re256i(int k0, int mul, int N)
{
  int16_t v[16] __attribute__((aligned(32)));
  for (int j = 0; j < 8; j++) {
    const int k = (mul * (k0 + j)) % N;
    const float theta = 2.0f * (float)M_PI * (float)k / (float)N;
    const int16_t raw = q15_from_float(cosf(theta));
    const int16_t wr = (N == 64) ? raw / 8 : (N % 4 == 0 && N != 128) ? raw / 2 : raw / sqrtf(2.0f);
    v[2 * j + 0] = wr;
    v[2 * j + 1] = wr;
  }
  return _mm256_load_si256((const __m256i *)v);
}

static inline __m256i pack8_twiddle_q15_im_signed256i(int k0, int mul, int N, dft_dir_t dir)
{
  int16_t v[16] __attribute__((aligned(32)));
  for (int j = 0; j < 8; j++) {
    const int k = (mul * (k0 + j)) % N;
    const float theta = 2.0f * (float)M_PI * (float)k / (float)N;
    const int16_t raw = q15_from_float((float)dir * sinf(theta));
    const int16_t wi = (N == 64) ? raw / 8 : (N % 4 == 0 && N != 128) ? raw / 2 : raw / sqrtf(2.0f);
    v[2 * j + 0] = -wi;
    v[2 * j + 1] = wi;
  }
  return _mm256_load_si256((const __m256i *)v);
}

static int fixed_64_128_twiddles_init(int N)
{
  if (N != 64 && N != 128)
    return 1;

  for (int m = 0; m < 8; m++) {
    if (N == 64) {
      g_dft64_re[0][m] = pack8_twiddle_q15_re_re256i(0, m, N);
      g_dft64_im[0][m] = pack8_twiddle_q15_im_signed256i(0, m, N, DFT_DIR_FORWARD);
      g_dft64_re[1][m] = g_dft64_re[0][m];
      g_dft64_im[1][m] = pack8_twiddle_q15_im_signed256i(0, m, N, DFT_DIR_INVERSE);
    } else {
      g_dft128_re[0][m] = pack8_twiddle_q15_re_re256i(8 * m, 1, N);
      g_dft128_im[0][m] = pack8_twiddle_q15_im_signed256i(8 * m, 1, N, DFT_DIR_FORWARD);
      g_dft128_re[1][m] = g_dft128_re[0][m];
      g_dft128_im[1][m] = pack8_twiddle_q15_im_signed256i(8 * m, 1, N, DFT_DIR_INVERSE);
    }
  }
  return 1;
}
static int r3_twiddle_create_q15_simd(r3_twiddle_t *table, int N)
{

  if (N <= 0) {
    return 0;
  }

  if (N % 3 != 0) {
    return 1;
  }

  const int size = N / 3;
  const int blocks = (size + 3) / 4;

  table->r3_q15_blocks = blocks;
  table->r3_q15_storage = aligned_malloc64((size_t)8 * (size_t)blocks * sizeof(__m128i));
  if (!table->r3_q15_storage)
    return 0;

  __m128i *p = table->r3_q15_storage;
  table->r3_q15_w1_re = p; p += blocks;
  table->r3_q15_w1_im = p; p += blocks;
  table->r3_q15_w2_re = p; p += blocks;
  table->r3_q15_w2_im = p; p += blocks;
  table->r3_q15_w1_re_inv = p; p += blocks;
  table->r3_q15_w1_im_inv = p; p += blocks;
  table->r3_q15_w2_re_inv = p; p += blocks;
  table->r3_q15_w2_im_inv = p;

  for (int b = 0; b < blocks; b++) {
    const int k0 = 4 * b;
    const float scale3 = 1.0f / sqrtf(3.0f);

    table->r3_q15_w1_re[b] = pack4_twiddle_q15_re_re_scaled(k0, 1, N, scale3);
    table->r3_q15_w1_im[b] = pack4_twiddle_q15_im_signed_scaled(k0, 1, N, scale3, DFT_DIR_FORWARD);

    table->r3_q15_w2_re[b] = pack4_twiddle_q15_re_re_scaled(k0, 2, N, scale3);
    table->r3_q15_w2_im[b] = pack4_twiddle_q15_im_signed_scaled(k0, 2, N, scale3, DFT_DIR_FORWARD);

    table->r3_q15_w1_re_inv[b] = pack4_twiddle_q15_re_re_scaled(k0, 1, N, scale3);
    table->r3_q15_w1_im_inv[b] = pack4_twiddle_q15_im_signed_scaled(k0, 1, N, scale3, DFT_DIR_INVERSE);

    table->r3_q15_w2_re_inv[b] = pack4_twiddle_q15_re_re_scaled(k0, 2, N, scale3);
    table->r3_q15_w2_im_inv[b] = pack4_twiddle_q15_im_signed_scaled(k0, 2, N, scale3, DFT_DIR_INVERSE);
  }

  return 1;
}

static int r5_twiddle_create_q15_simd(r5_twiddle_t *table, int N)
{

  if (N <= 0)
    return 0;

  if (N % 5 != 0)
    return 1;

  const int size = N / 5;
  const int blocks = (size + 3) / 4;

  table->r5_q15_blocks = blocks;
  table->r5_q15_storage = aligned_malloc64((size_t)16 * (size_t)blocks * sizeof(__m128i));
  if (!table->r5_q15_storage)
    return 0;

  __m128i *p = table->r5_q15_storage;
  table->r5_q15_w1_re = p; p += blocks;
  table->r5_q15_w1_im = p; p += blocks;
  table->r5_q15_w2_re = p; p += blocks;
  table->r5_q15_w2_im = p; p += blocks;
  table->r5_q15_w3_re = p; p += blocks;
  table->r5_q15_w3_im = p; p += blocks;
  table->r5_q15_w4_re = p; p += blocks;
  table->r5_q15_w4_im = p; p += blocks;
  table->r5_q15_w1_re_inv = p; p += blocks;
  table->r5_q15_w1_im_inv = p; p += blocks;
  table->r5_q15_w2_re_inv = p; p += blocks;
  table->r5_q15_w2_im_inv = p; p += blocks;
  table->r5_q15_w3_re_inv = p; p += blocks;
  table->r5_q15_w3_im_inv = p; p += blocks;
  table->r5_q15_w4_re_inv = p; p += blocks;
  table->r5_q15_w4_im_inv = p;

  for (int b = 0; b < blocks; b++) {
    const int k0 = 4 * b;
    const float scale5 = 1.0f / sqrtf(5.0f);

    table->r5_q15_w1_re[b] = pack4_twiddle_q15_re_re_scaled(k0, 1, N, scale5);
    table->r5_q15_w1_im[b] = pack4_twiddle_q15_im_signed_scaled(k0, 1, N, scale5, DFT_DIR_FORWARD);

    table->r5_q15_w2_re[b] = pack4_twiddle_q15_re_re_scaled(k0, 2, N, scale5);
    table->r5_q15_w2_im[b] = pack4_twiddle_q15_im_signed_scaled(k0, 2, N, scale5, DFT_DIR_FORWARD);

    table->r5_q15_w3_re[b] = pack4_twiddle_q15_re_re_scaled(k0, 3, N, scale5);
    table->r5_q15_w3_im[b] = pack4_twiddle_q15_im_signed_scaled(k0, 3, N, scale5, DFT_DIR_FORWARD);

    table->r5_q15_w4_re[b] = pack4_twiddle_q15_re_re_scaled(k0, 4, N, scale5);
    table->r5_q15_w4_im[b] = pack4_twiddle_q15_im_signed_scaled(k0, 4, N, scale5, DFT_DIR_FORWARD);

    table->r5_q15_w1_re_inv[b] = pack4_twiddle_q15_re_re_scaled(k0, 1, N, scale5);
    table->r5_q15_w1_im_inv[b] = pack4_twiddle_q15_im_signed_scaled(k0, 1, N, scale5, DFT_DIR_INVERSE);

    table->r5_q15_w2_re_inv[b] = pack4_twiddle_q15_re_re_scaled(k0, 2, N, scale5);
    table->r5_q15_w2_im_inv[b] = pack4_twiddle_q15_im_signed_scaled(k0, 2, N, scale5, DFT_DIR_INVERSE);

    table->r5_q15_w3_re_inv[b] = pack4_twiddle_q15_re_re_scaled(k0, 3, N, scale5);
    table->r5_q15_w3_im_inv[b] = pack4_twiddle_q15_im_signed_scaled(k0, 3, N, scale5, DFT_DIR_INVERSE);

    table->r5_q15_w4_re_inv[b] = pack4_twiddle_q15_re_re_scaled(k0, 4, N, scale5);
    table->r5_q15_w4_im_inv[b] = pack4_twiddle_q15_im_signed_scaled(k0, 4, N, scale5, DFT_DIR_INVERSE);
  }

  return 1;
}

static pthread_mutex_t twiddle_component_mutex = PTHREAD_MUTEX_INITIALIZER;

static int r3_twiddle_ensure_q15(r3_twiddle_t *table, int N)
{
  if (__builtin_expect(__atomic_load_n(&table->r3_q15_ready, __ATOMIC_ACQUIRE), 1))
    return 1;

  pthread_mutex_lock(&twiddle_component_mutex);
  if (!__atomic_load_n(&table->r3_q15_ready, __ATOMIC_RELAXED)) {
    if (!r3_twiddle_create_q15_simd(table, N)) {
      pthread_mutex_unlock(&twiddle_component_mutex);
      return 0;
    }
    __atomic_store_n(&table->r3_q15_ready, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&twiddle_component_mutex);
  return 1;
}

static int r5_twiddle_ensure_q15(r5_twiddle_t *table, int N)
{
  if (__builtin_expect(__atomic_load_n(&table->r5_q15_ready, __ATOMIC_ACQUIRE), 1))
    return 1;

  pthread_mutex_lock(&twiddle_component_mutex);
  if (!__atomic_load_n(&table->r5_q15_ready, __ATOMIC_RELAXED)) {
    if (!r5_twiddle_create_q15_simd(table, N)) {
      pthread_mutex_unlock(&twiddle_component_mutex);
      return 0;
    }
    __atomic_store_n(&table->r5_q15_ready, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&twiddle_component_mutex);
  return 1;
}

static const r3_twiddle_t *r3_twiddle_get(int N)
{
  r3_twiddle_t *table = r3_twiddle_slot(N);
  if (!table || !r3_twiddle_ensure_q15(table, N))
    return NULL;
  return table;
}

static const r5_twiddle_t *r5_twiddle_get(int N)
{
  r5_twiddle_t *table = r5_twiddle_slot(N);
  if (!table || !r5_twiddle_ensure_q15(table, N))
    return NULL;
  return table;
}

/* Exact-size readiness state avoids repeated lazy-component checks on selected paths. */
typedef struct {
  unsigned char ready;
} radix9_plan_t;

typedef struct {
  unsigned char ready;
} radix15_plan_t;

typedef struct {
  unsigned char ready;
} radix25_plan_t;

typedef struct {
  unsigned char ready;
} radix3_selected_plan_t;

static pthread_mutex_t selected_plan_mutex = PTHREAD_MUTEX_INITIALIZER;

#define R9_PLAN_ENTRY(N_) { 0 }
static radix9_plan_t g_radix9_plans[] = {
  R9_PLAN_ENTRY(72), R9_PLAN_ENTRY(108), R9_PLAN_ENTRY(144), R9_PLAN_ENTRY(288),
  R9_PLAN_ENTRY(576), R9_PLAN_ENTRY(648), R9_PLAN_ENTRY(972), R9_PLAN_ENTRY(1152),
  R9_PLAN_ENTRY(1296), R9_PLAN_ENTRY(2304), R9_PLAN_ENTRY(2592),
};
#undef R9_PLAN_ENTRY

static inline radix9_plan_t *radix9_plan_slot(int N)
{
  switch (N) {
    case 72: return &g_radix9_plans[0];
    case 108: return &g_radix9_plans[1];
    case 144: return &g_radix9_plans[2];
    case 288: return &g_radix9_plans[3];
    case 576: return &g_radix9_plans[4];
    case 648: return &g_radix9_plans[5];
    case 972: return &g_radix9_plans[6];
    case 1152: return &g_radix9_plans[7];
    case 1296: return &g_radix9_plans[8];
    case 2304: return &g_radix9_plans[9];
    case 2592: return &g_radix9_plans[10];
    default: return NULL;
  }
}

static __attribute__((noinline, cold)) const radix9_plan_t *radix9_plan_get_slow(radix9_plan_t *p, int N)
{
  pthread_mutex_lock(&selected_plan_mutex);
  if (!__atomic_load_n(&p->ready, __ATOMIC_RELAXED)) {
    const r3_twiddle_t *twB = r3_twiddle_get(N);
    const r3_twiddle_t *twA = r3_twiddle_get(N / 3);
    AssertFatal(twB && twA, "Missing radix-9 plan N=%d\n", N);
    __atomic_store_n(&p->ready, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&selected_plan_mutex);
  return p;
}

static __attribute__((always_inline)) inline const radix9_plan_t *radix9_plan_get(int N)
{
  radix9_plan_t *p = radix9_plan_slot(N);
  if (!p)
    return NULL;
  if (__builtin_expect(__atomic_load_n(&p->ready, __ATOMIC_ACQUIRE), 1))
    return p;
  return radix9_plan_get_slow(p, N);
}

#define R15_PLAN_ENTRY(N_) { 0 }
static radix15_plan_t g_radix15_plans[] = {
  R15_PLAN_ENTRY(120), R15_PLAN_ENTRY(240), R15_PLAN_ENTRY(480),
  R15_PLAN_ENTRY(960), R15_PLAN_ENTRY(1920),
};
#undef R15_PLAN_ENTRY

static inline radix15_plan_t *radix15_plan_slot(int N)
{
  switch (N) {
    case 120: return &g_radix15_plans[0];
    case 240: return &g_radix15_plans[1];
    case 480: return &g_radix15_plans[2];
    case 960: return &g_radix15_plans[3];
    case 1920: return &g_radix15_plans[4];
    default: return NULL;
  }
}

static __attribute__((noinline, cold)) const radix15_plan_t *radix15_plan_get_slow(radix15_plan_t *p, int N)
{
  pthread_mutex_lock(&selected_plan_mutex);
  if (!__atomic_load_n(&p->ready, __ATOMIC_RELAXED)) {
    const r3_twiddle_t *twB = r3_twiddle_get(N);
    const r5_twiddle_t *twA = r5_twiddle_get(N / 3);
    AssertFatal(twB && twA, "Missing radix-15 plan N=%d\n", N);
    __atomic_store_n(&p->ready, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&selected_plan_mutex);
  return p;
}

static __attribute__((always_inline)) inline const radix15_plan_t *radix15_plan_get(int N)
{
  radix15_plan_t *p = radix15_plan_slot(N);
  if (!p)
    return NULL;
  if (__builtin_expect(__atomic_load_n(&p->ready, __ATOMIC_ACQUIRE), 1))
    return p;
  return radix15_plan_get_slow(p, N);
}

#define R25_PLAN_ENTRY(N_) { 0 }
static radix25_plan_t g_radix25_plans[] = {
  R25_PLAN_ENTRY(300), R25_PLAN_ENTRY(400), R25_PLAN_ENTRY(600), R25_PLAN_ENTRY(800),
};
#undef R25_PLAN_ENTRY

static inline radix25_plan_t *radix25_plan_slot(int N)
{
  switch (N) {
    case 300: return &g_radix25_plans[0];
    case 400: return &g_radix25_plans[1];
    case 600: return &g_radix25_plans[2];
    case 800: return &g_radix25_plans[3];
    default: return NULL;
  }
}

static __attribute__((noinline, cold)) const radix25_plan_t *radix25_plan_get_slow(radix25_plan_t *p, int N)
{
  pthread_mutex_lock(&selected_plan_mutex);
  if (!__atomic_load_n(&p->ready, __ATOMIC_RELAXED)) {
    const int M = N / 25;
    const r5_twiddle_t *tw_inner = r5_twiddle_get(5 * M);
    const r5_twiddle_t *tw_outer = r5_twiddle_get(N);
    AssertFatal(tw_inner && tw_outer, "Missing radix-25 plan N=%d\n", N);
    __atomic_store_n(&p->ready, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&selected_plan_mutex);
  return p;
}

static __attribute__((always_inline)) inline const radix25_plan_t *radix25_plan_get(int N)
{
  radix25_plan_t *p = radix25_plan_slot(N);
  if (!p)
    return NULL;
  if (__builtin_expect(__atomic_load_n(&p->ready, __ATOMIC_ACQUIRE), 1))
    return p;
  return radix25_plan_get_slow(p, N);
}

#define R3SEL_PLAN_ENTRY(N_) { 0 }
static radix3_selected_plan_t g_radix3_selected_plans[] = {
  R3SEL_PLAN_ENTRY(48), R3SEL_PLAN_ENTRY(96), R3SEL_PLAN_ENTRY(192),
  R3SEL_PLAN_ENTRY(384), R3SEL_PLAN_ENTRY(768), R3SEL_PLAN_ENTRY(1536),
  R3SEL_PLAN_ENTRY(3072), R3SEL_PLAN_ENTRY(6144), R3SEL_PLAN_ENTRY(12288),
  R3SEL_PLAN_ENTRY(24576), R3SEL_PLAN_ENTRY(49152), R3SEL_PLAN_ENTRY(98304),
  R3SEL_PLAN_ENTRY(1200), R3SEL_PLAN_ENTRY(2400),
};
#undef R3SEL_PLAN_ENTRY

static inline radix3_selected_plan_t *radix3_selected_plan_slot(int N)
{
  switch (N) {
    case 48: return &g_radix3_selected_plans[0];
    case 96: return &g_radix3_selected_plans[1];
    case 192: return &g_radix3_selected_plans[2];
    case 384: return &g_radix3_selected_plans[3];
    case 768: return &g_radix3_selected_plans[4];
    case 1536: return &g_radix3_selected_plans[5];
    case 3072: return &g_radix3_selected_plans[6];
    case 6144: return &g_radix3_selected_plans[7];
    case 12288: return &g_radix3_selected_plans[8];
    case 24576: return &g_radix3_selected_plans[9];
    case 49152: return &g_radix3_selected_plans[10];
    case 98304: return &g_radix3_selected_plans[11];
    case 1200: return &g_radix3_selected_plans[12];
    case 2400: return &g_radix3_selected_plans[13];
    default: return NULL;
  }
}

static __attribute__((noinline, cold)) const radix3_selected_plan_t *radix3_selected_plan_get_slow(radix3_selected_plan_t *p, int N)
{
  pthread_mutex_lock(&selected_plan_mutex);
  if (!__atomic_load_n(&p->ready, __ATOMIC_RELAXED)) {
    const r3_twiddle_t *tw = r3_twiddle_get(N);
    AssertFatal(tw, "Missing selected radix-3 plan N=%d\n", N);
    __atomic_store_n(&p->ready, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&selected_plan_mutex);
  return p;
}

static __attribute__((always_inline)) inline const radix3_selected_plan_t *radix3_selected_plan_get(int N)
{
  radix3_selected_plan_t *p = radix3_selected_plan_slot(N);
  if (!p)
    return NULL;
  if (__builtin_expect(__atomic_load_n(&p->ready, __ATOMIC_ACQUIRE), 1))
    return p;
  return radix3_selected_plan_get_slow(p, N);
}

static inline int selected_radix15_radix30_size(int N)
{
  switch (N) {
    case 120:
    case 240:
    case 480:
    case 960:
    case 1920:
      return 1;
    default:
      return 0;
  }
}

static pthread_mutex_t mixed_power2_twiddle_mutex = PTHREAD_MUTEX_INITIALIZER;

static int power2_twiddle_ensure_q15(power2_twiddle_t *table, int N, int radix)
{
  if (radix != 8 && radix != 16)
    return 0;

  if (__builtin_expect(__atomic_load_n(&table->q15_ready, __ATOMIC_ACQUIRE), 1))
    return 1;

  pthread_mutex_lock(&mixed_power2_twiddle_mutex);
  if (__atomic_load_n(&table->q15_ready, __ATOMIC_RELAXED)) {
    pthread_mutex_unlock(&mixed_power2_twiddle_mutex);
    return 1;
  }

  const int size = N / radix;
  if (size < 8 || (size & 7)) {
    pthread_mutex_unlock(&mixed_power2_twiddle_mutex);
    return 0;
  }

  const int blocks = size / 8;
  const size_t count = (size_t)radix * (size_t)blocks;
  table->q15_re = aligned_malloc64(count * sizeof(__m256i));
  table->q15_im = aligned_malloc64(count * sizeof(__m256i));
  table->q15_re_inv = aligned_malloc64(count * sizeof(__m256i));
  table->q15_im_inv = aligned_malloc64(count * sizeof(__m256i));
  if (!table->q15_re || !table->q15_im || !table->q15_re_inv || !table->q15_im_inv) {
    free(table->q15_re);
    free(table->q15_im);
    free(table->q15_re_inv);
    free(table->q15_im_inv);
    table->q15_re = NULL;
    table->q15_im = NULL;
    table->q15_re_inv = NULL;
    table->q15_im_inv = NULL;
    pthread_mutex_unlock(&mixed_power2_twiddle_mutex);
    return 0;
  }

  const float scale = 1.0f / sqrtf(8.0f);
  for (int r = 0; r < radix; r++) {
    for (int b = 0; b < blocks; b++) {
      const int idx = r * blocks + b;
      const int k0 = 8 * b;
      table->q15_re[idx] = pack8_twiddle_q15_re_re_scaled(k0, r, N, scale);
      table->q15_im[idx] = pack8_twiddle_q15_im_signed_scaled(k0, r, N, scale, DFT_DIR_FORWARD);
      table->q15_re_inv[idx] = pack8_twiddle_q15_re_re_scaled(k0, r, N, scale);
      table->q15_im_inv[idx] = pack8_twiddle_q15_im_signed_scaled(k0, r, N, scale, DFT_DIR_INVERSE);
    }
  }

  table->q15_blocks = blocks;
  __atomic_store_n(&table->q15_ready, 1, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&mixed_power2_twiddle_mutex);
  return 1;
}

typedef struct {
  int N;
  int blocks;
  simde__m256i *storage;
  simde__m256i *W1_RE_NEGIM;
  simde__m256i *W1_IM_RE;
  simde__m256i *W3_RE_NEGIM;
  simde__m256i *W3_IM_RE;
} sr_twiddle_simd_t;

static sr_twiddle_simd_t sr_twiddles_fwd[SR_MAX_LOG2 + 1];
static sr_twiddle_simd_t sr_twiddles_bwd[SR_MAX_LOG2 + 1];
static unsigned char sr_twiddles_fwd_ready[SR_MAX_LOG2 + 1];
static unsigned char sr_twiddles_bwd_ready[SR_MAX_LOG2 + 1];

static int init_sr_twiddle_simd(sr_twiddle_simd_t *tw, int N, dft_dir_t dir)
{
  int quarter = N / 4;
  int blocks = quarter / 8;

  tw->N = N;
  tw->blocks = blocks;

  tw->storage = aligned_alloc(32, (size_t)4 * (size_t)blocks * sizeof(simde__m256i));
  if (!tw->storage)
    return 0;
  tw->W1_RE_NEGIM = tw->storage;
  tw->W1_IM_RE = tw->storage + blocks;
  tw->W3_RE_NEGIM = tw->storage + 2 * blocks;
  tw->W3_IM_RE = tw->storage + 3 * blocks;

  for (int b = 0; b < blocks; b++) {
    int16_t w1_re_negim[16] __attribute__((aligned(64)));
    int16_t w1_im_re[16] __attribute__((aligned(64)));
    int16_t w3_re_negim[16] __attribute__((aligned(64)));
    int16_t w3_im_re[16] __attribute__((aligned(64)));

    for (int j = 0; j < 8; j++) {
      int k = 8 * b + j;

      float theta1 = (float)dir * 2.0f * (float)M_PI * k / (float)N;
      float theta3 = (float)dir * 6.0f * (float)M_PI * k / (float)N;

      int16_t w1r = sat_i16(lrintf(32767.0f * cosf(theta1))) / 2;
      int16_t w1i = sat_i16(lrintf(32767.0f * sinf(theta1))) / 2;

      int16_t w3r = sat_i16(lrintf(32767.0f * cosf(theta3))) / 2;
      int16_t w3i = sat_i16(lrintf(32767.0f * sinf(theta3))) / 2;

      w1_re_negim[2 * j] = w1r;
      w1_re_negim[2 * j + 1] = sat_i16(-(long)w1i);
      w1_im_re[2 * j] = w1i;
      w1_im_re[2 * j + 1] = w1r;

      w3_re_negim[2 * j] = w3r;
      w3_re_negim[2 * j + 1] = sat_i16(-(long)w3i);
      w3_im_re[2 * j] = w3i;
      w3_im_re[2 * j + 1] = w3r;
    }

    tw->W1_RE_NEGIM[b] = simde_mm256_load_si256((simde__m256i *)w1_re_negim);
    tw->W1_IM_RE[b] = simde_mm256_load_si256((simde__m256i *)w1_im_re);
    tw->W3_RE_NEGIM[b] = simde_mm256_load_si256((simde__m256i *)w3_re_negim);
    tw->W3_IM_RE[b] = simde_mm256_load_si256((simde__m256i *)w3_im_re);
  }
  return 1;
}

static sr_twiddle_simd_t *sr_twiddle_table_create(int N, dft_dir_t dir)
{
  const int idx = log2_int((unsigned int)N);

  sr_twiddle_simd_t *tw = (dir == DFT_DIR_FORWARD) ? &sr_twiddles_fwd[idx] : &sr_twiddles_bwd[idx];

  memset(tw, 0, sizeof(*tw));

  if (!init_sr_twiddle_simd(tw, N, dir)) {
    return NULL;
  }

  return tw;
}

const sr_twiddle_simd_t *sr_twiddle_table_get(int N, dft_dir_t dir)
{
  if (N <= 0 || !is_power_of_two_int(N)) {
    LOG_E(PHY, "Invalid split-radix N=%d; expected a power of two\n", N);
    return NULL;
  }

  const int idx = log2_int((unsigned int)N);
  AssertFatal(idx <= SR_MAX_LOG2,
              "Split-radix log2(N)=%d exceeds SR_MAX_LOG2=%d\n",
              idx,
              SR_MAX_LOG2);

  sr_twiddle_simd_t *tw = (dir == DFT_DIR_FORWARD) ? &sr_twiddles_fwd[idx] : &sr_twiddles_bwd[idx];
  unsigned char *ready = (dir == DFT_DIR_FORWARD) ? &sr_twiddles_fwd_ready[idx] : &sr_twiddles_bwd_ready[idx];

  if (__builtin_expect(__atomic_load_n(ready, __ATOMIC_ACQUIRE), 1))
    return tw;

  pthread_mutex_lock(&sr_twiddle_mutex);

  if (!__atomic_load_n(ready, __ATOMIC_RELAXED)) {
    if (!sr_twiddle_table_create(N, dir)) {
      pthread_mutex_unlock(&sr_twiddle_mutex);
      return NULL;
    }
    __atomic_store_n(ready, 1, __ATOMIC_RELEASE);
  }

  pthread_mutex_unlock(&sr_twiddle_mutex);
  return tw;
}

static int sr_twiddle_prepare_power2(int N, dft_dir_t dir)
{
  if (N <= 1024)
    return 1;
  for (int n = 2048; n <= N; n <<= 1) {
    if (!sr_twiddle_table_get(n, dir))
      return 0;
    if (n > (INT32_MAX >> 1))
      break;
  }
  return 1;
}

#define DFT_C16_SR_MAX_N 65536

static void dft_c16_init_impl(void)
{
  AssertFatal(fixed_64_128_twiddles_init(64), "Failed to initialize DFT64 twiddles\n");
  AssertFatal(fixed_64_128_twiddles_init(128), "Failed to initialize DFT128 twiddles\n");
}

__attribute__((constructor)) static void dft_c16_library_init(void)
{
  dft_c16_init_impl();
}

static inline __m128i dft64_dc_from_h0(__m256i h0)
{
  const __m256i real_mask = _mm256_setr_epi16(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);

  const __m256i imag_mask = _mm256_setr_epi16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);

  __m256i real32 = _mm256_madd_epi16(h0, real_mask);
  __m256i imag32 = _mm256_madd_epi16(h0, imag_mask);

  /* Each half contains [sum_real, sum_imag, ...] after the horizontal additions. */
  __m256i sum = _mm256_hadd_epi32(real32, imag32);
  sum = _mm256_hadd_epi32(sum, sum);

  __m128i dc32 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));

  /* Rounded division by 8. */
  const __m128i sign = _mm_srai_epi32(dc32, 31);

  dc32 = _mm_add_epi32(dc32, _mm_add_epi32(_mm_set1_epi32(4), sign));

  dc32 = _mm_srai_epi32(dc32, 3);

  return _mm_packs_epi32(dc32, _mm_setzero_si128());
}

static __attribute__((always_inline)) inline void dft8x8_q15_256_dir(const __m256i x0,
                                      const __m256i x1,
                                      const __m256i x2,
                                      const __m256i x3,
                                      const __m256i x4,
                                      const __m256i x5,
                                      const __m256i x6,
                                      const __m256i x7,
                                      __m256i *Y0,
                                      __m256i *Y1,
                                      __m256i *Y2,
                                      __m256i *Y3,
                                      __m256i *Y4,
                                      __m256i *Y5,
                                      __m256i *Y6,
                                      __m256i *Y7,
                                      dft_dir_t dir)
{
  const __m256i c = _mm256_set1_epi16(Q15_INV_SQRT2);

  const __m256i s04 = _mm256_adds_epi16(x0, x4);
  const __m256i d04 = _mm256_subs_epi16(x0, x4);

  const __m256i s15 = _mm256_adds_epi16(x1, x5);
  const __m256i d15 = _mm256_subs_epi16(x1, x5);

  const __m256i s26 = _mm256_adds_epi16(x2, x6);
  const __m256i d26 = _mm256_subs_epi16(x2, x6);

  const __m256i s37 = _mm256_adds_epi16(x3, x7);
  const __m256i d37 = _mm256_subs_epi16(x3, x7);

  const __m256i s02 = _mm256_adds_epi16(s04, s26);
  const __m256i d02 = _mm256_subs_epi16(s04, s26);

  const __m256i s13 = _mm256_adds_epi16(s15, s37);
  const __m256i d13 = _mm256_subs_epi16(s15, s37);

  *Y0 = _mm256_adds_epi16(s02, s13);
  *Y4 = _mm256_subs_epi16(s02, s13);

  *Y2 = _mm256_adds_epi16(d02, mul_minus_j_dir_i16_256(d13, dir));
  *Y6 = _mm256_adds_epi16(d02, mul_plus_j_dir_i16_256(d13, dir));

  const __m256i p = _mm256_adds_epi16(d15, d37);
  const __m256i q = _mm256_subs_epi16(d15, d37);

  const __m256i d26_mj = mul_minus_j_dir_i16_256(d26, dir);
  const __m256i d26_pj = mul_plus_j_dir_i16_256(d26, dir);

  const __m256i base_mj = _mm256_adds_epi16(d04, d26_mj);
  const __m256i base_pj = _mm256_adds_epi16(d04, d26_pj);

  const __m256i t1_arg = _mm256_adds_epi16(q, mul_minus_j_dir_i16_256(p, dir));
  const __m256i t3_arg = _mm256_adds_epi16(q, mul_plus_j_dir_i16_256(p, dir));

  const __m256i t1 = _mm256_mulhrs_epi16(c, t1_arg);
  const __m256i t3 = _mm256_mulhrs_epi16(c, t3_arg);

  *Y1 = _mm256_adds_epi16(base_mj, t1);
  *Y5 = _mm256_subs_epi16(base_mj, t1);

  *Y7 = _mm256_adds_epi16(base_pj, t3);
  *Y3 = _mm256_subs_epi16(base_pj, t3);
}

static inline void dft64_avx(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  const __m256i x0 = _mm256_loadu_si256((const __m256i *)(src + 0));
  const __m256i x1 = _mm256_loadu_si256((const __m256i *)(src + 8));
  const __m256i x2 = _mm256_loadu_si256((const __m256i *)(src + 16));
  const __m256i x3 = _mm256_loadu_si256((const __m256i *)(src + 24));
  const __m256i x4 = _mm256_loadu_si256((const __m256i *)(src + 32));
  const __m256i x5 = _mm256_loadu_si256((const __m256i *)(src + 40));
  const __m256i x6 = _mm256_loadu_si256((const __m256i *)(src + 48));
  const __m256i x7 = _mm256_loadu_si256((const __m256i *)(src + 56));

  __m256i H0, H1, H2, H3;
  __m256i H4, H5, H6, H7;

  dft8x8_q15_256_dir(x0, x1, x2, x3, x4, x5, x6, x7, &H0, &H1, &H2, &H3, &H4, &H5, &H6, &H7, dir);
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i *C64_RE = g_dft64_re[ds];
  const __m256i *C64_IM = g_dft64_im[ds];
  const __m128i dc = dft64_dc_from_h0(H0);
  H0 = _mm256_srai_epi16(H0, 3);

  H1 = complex_mul8_prepack_q15_256(H1, C64_RE[1], C64_IM[1]);
  H2 = complex_mul8_prepack_q15_256(H2, C64_RE[2], C64_IM[2]);
  H3 = complex_mul8_prepack_q15_256(H3, C64_RE[3], C64_IM[3]);
  H4 = complex_mul8_prepack_q15_256(H4, C64_RE[4], C64_IM[4]);
  H5 = complex_mul8_prepack_q15_256(H5, C64_RE[5], C64_IM[5]);
  H6 = complex_mul8_prepack_q15_256(H6, C64_RE[6], C64_IM[6]);
  H7 = complex_mul8_prepack_q15_256(H7, C64_RE[7], C64_IM[7]);

  transpose8_complex_i16_256(&H0, &H1, &H2, &H3, &H4, &H5, &H6, &H7);

  __m256i Y0, Y1, Y2, Y3;
  __m256i Y4, Y5, Y6, Y7;

  dft8x8_q15_256_dir(H0, H1, H2, H3, H4, H5, H6, H7, &Y0, &Y1, &Y2, &Y3, &Y4, &Y5, &Y6, &Y7, dir);
  Y0 = _mm256_blend_epi32(Y0, _mm256_castsi128_si256(dc), 0x01);
  _mm256_storeu_si256((__m256i *)(dst + 0), Y0);
  _mm256_storeu_si256((__m256i *)(dst + 8), Y1);
  _mm256_storeu_si256((__m256i *)(dst + 16), Y2);
  _mm256_storeu_si256((__m256i *)(dst + 24), Y3);
  _mm256_storeu_si256((__m256i *)(dst + 32), Y4);
  _mm256_storeu_si256((__m256i *)(dst + 40), Y5);
  _mm256_storeu_si256((__m256i *)(dst + 48), Y6);
  _mm256_storeu_si256((__m256i *)(dst + 56), Y7);
}

static inline void dft64_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir)
{
  if (stride == 1) {
    dft64_avx((c16_t *)src, dst, dir);
    return;
  }
  c16_t tmp[64] __attribute__((aligned(64)));
  for (int i = 0; i < 64; i++) {
    tmp[i] = src[i * stride];
  }

  dft64_avx(tmp, dst, dir);
}

static inline void dft128_stage0_blk_q15_256_dir(const c16_t *src,
                                                 c16_t *a,
                                                 c16_t *b,
                                                 int blk,
                                                 dft_dir_t dir,
                                                 const __m256i *tw_RE,
                                                 const __m256i *tw_IM)
{
  (void)dir;
  const __m256i x0 = _mm256_loadu_si256((const __m256i *)(src + 8 * blk));

  const __m256i x1 = _mm256_loadu_si256((const __m256i *)(src + 64 + 8 * blk));

  __m256i sum = _mm256_adds_epi16(x0, x1);
  __m256i diff = _mm256_subs_epi16(x0, x1);

  const __m256i s = _mm256_set1_epi16(Q15_INV_SQRT2);
  sum = _mm256_mulhrs_epi16(sum, s);

  diff = complex_mul8_prepack_q15_256(diff, tw_RE[blk], tw_IM[blk]);

  _mm256_store_si256((__m256i *)(a + 8 * blk), sum);

  _mm256_store_si256((__m256i *)(b + 8 * blk), diff);
}

static inline void interleave64_complex_q15_256(const c16_t *A, const c16_t *B, c16_t *dst)
{
  for (int blk = 0; blk < 8; blk++) {
    const __m256i va = _mm256_load_si256((const __m256i *)(A + 8 * blk));

    const __m256i vb = _mm256_load_si256((const __m256i *)(B + 8 * blk));

    /*
     * va = [A0 A1 A2 A3 | A4 A5 A6 A7]
     * vb = [B0 B1 B2 B3 | B4 B5 B6 B7]
     *
     * Each A0/B0 is one c16_t = 32 bits.
     */
    const __m256i lo = _mm256_unpacklo_epi32(va, vb);
    const __m256i hi = _mm256_unpackhi_epi32(va, vb);

    /*
     * out0 = [A0 B0 A1 B1 A2 B2 A3 B3]
     * out1 = [A4 B4 A5 B5 A6 B6 A7 B7]
     */
    const __m256i out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
    const __m256i out1 = _mm256_permute2x128_si256(lo, hi, 0x31);

    _mm256_storeu_si256((__m256i *)(dst + 16 * blk), out0);

    _mm256_storeu_si256((__m256i *)(dst + 16 * blk + 8), out1);
  }
}

static inline void dft128_dir(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  c16_t a[64] __attribute__((aligned(32)));
  c16_t b[64] __attribute__((aligned(32)));

  c16_t A[64] __attribute__((aligned(32)));
  c16_t B[64] __attribute__((aligned(32)));
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i *W128_RE = g_dft128_re[ds];
  const __m256i *W128_IM = g_dft128_im[ds];

  for (int blk = 0; blk < 8; blk++) {
    dft128_stage0_blk_q15_256_dir(src, a, b, blk, dir, W128_RE, W128_IM);
  }

  dft64_avx(a, A, dir);
  dft64_avx(b, B, dir);

  interleave64_complex_q15_256(A, B, dst);
}

static inline void dft128_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir)
{
  if (stride == 1) {
    dft128_dir((c16_t *)src, dst, dir);
    return;
  }

  c16_t tmp[128] __attribute__((aligned(64)));

  for (int i = 0; i < 128; i++) {
    tmp[i] = src[i * stride];
  }

  dft128_dir(tmp, dst, dir);
}

static inline size_t split_radix_work_len_c16(int N)
{
  size_t need = 0;

  while (N > 128) {
    need += 2u * (size_t)N;
    N >>= 1;
  }

  return need;
}

static inline void sr_combine_simd(c16_t *E, c16_t *O1, c16_t *O3, c16_t *y, int N, const sr_twiddle_simd_t *tw, dft_dir_t dir)
{
  int half = N / 2;
  int quarter = N / 4;

  const simde__m256i swap_mask = simde_mm256_setr_epi8(2,
                                                       3,
                                                       0,
                                                       1,
                                                       6,
                                                       7,
                                                       4,
                                                       5,
                                                       10,
                                                       11,
                                                       8,
                                                       9,
                                                       14,
                                                       15,
                                                       12,
                                                       13,
                                                       2,
                                                       3,
                                                       0,
                                                       1,
                                                       6,
                                                       7,
                                                       4,
                                                       5,
                                                       10,
                                                       11,
                                                       8,
                                                       9,
                                                       14,
                                                       15,
                                                       12,
                                                       13);

  simde__m256i sign_mask;

  if (dir == DFT_DIR_FORWARD) {
    sign_mask = simde_mm256_setr_epi16(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);
  } else {
    sign_mask = simde_mm256_setr_epi16(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);
  }

  const simde__m256i sqrt2_inv = simde_mm256_set1_epi16(Q15_INV_SQRT2);

  for (int b = 0; b < tw->blocks; b++) {
    int k = 8 * b;

    simde__m256i O1v = simde_mm256_load_si256((simde__m256i *)&O1[k]);
    simde__m256i O3v = simde_mm256_load_si256((simde__m256i *)&O3[k]);

    simde__m256i t1 = c16_mul_q15_simd256(O1v, tw->W1_RE_NEGIM[b], tw->W1_IM_RE[b]);

    simde__m256i t2 = c16_mul_q15_simd256(O3v, tw->W3_RE_NEGIM[b], tw->W3_IM_RE[b]);

    simde__m256i a = simde_mm256_add_epi16(t1, t2);
    simde__m256i d = simde_mm256_sub_epi16(t1, t2);

    simde__m256i bval = simde_mm256_shuffle_epi8(d, swap_mask);
    bval = simde_mm256_sign_epi16(bval, sign_mask);

    simde__m256i E0 = simde_mm256_load_si256((simde__m256i *)&E[k]);
    simde__m256i E1 = simde_mm256_load_si256((simde__m256i *)&E[k + quarter]);

    E0 = simde_mm256_mulhrs_epi16(E0, sqrt2_inv);
    E1 = simde_mm256_mulhrs_epi16(E1, sqrt2_inv);

    simde__m256i Y0 = simde_mm256_add_epi16(E0, a);
    simde__m256i Y2 = simde_mm256_sub_epi16(E0, a);
    simde__m256i Y1 = simde_mm256_add_epi16(E1, bval);
    simde__m256i Y3 = simde_mm256_sub_epi16(E1, bval);

    simde_mm256_store_si256((simde__m256i *)&y[k], Y0);
    simde_mm256_store_si256((simde__m256i *)&y[k + quarter], Y1);
    simde_mm256_store_si256((simde__m256i *)&y[k + half], Y2);
    simde_mm256_store_si256((simde__m256i *)&y[k + 3 * quarter], Y3);
  }
}

static inline void pack_split_radix_input_avx2_fused(const c16_t *__restrict x, c16_t *__restrict sub_in, int N)
{
  _Static_assert(sizeof(c16_t) == 4, "c16_t must be 32-bit");

  const int half = N >> 1;
  const int quarter = N >> 2;

  c16_t *__restrict E_in = sub_in;
  c16_t *__restrict O1_in = sub_in + half;
  c16_t *__restrict O3_in = sub_in + half + quarter;

  const simde__m256i idx = simde_mm256_setr_epi32(0, 2, 4, 6, 1, 5, 3, 7);

  int in = 0;
  int e = 0;
  int o = 0;

  for (; in + 32 <= N; in += 32, e += 16, o += 8) {
    simde__m256i v0 = simde_mm256_loadu_si256((const simde__m256i *)&x[in + 0]);

    simde__m256i v1 = simde_mm256_loadu_si256((const simde__m256i *)&x[in + 8]);

    simde__m256i v2 = simde_mm256_loadu_si256((const simde__m256i *)&x[in + 16]);

    simde__m256i v3 = simde_mm256_loadu_si256((const simde__m256i *)&x[in + 24]);

    /*
     * p0 = [x0,  x2,  x4,  x6,  x1,  x5,  x3,  x7]
     * p1 = [x8,  x10, x12, x14, x9,  x13, x11, x15]
     * p2 = [x16, x18, x20, x22, x17, x21, x19, x23]
     * p3 = [x24, x26, x28, x30, x25, x29, x27, x31]
     */
    simde__m256i p0 = simde_mm256_permutevar8x32_epi32(v0, idx);
    simde__m256i p1 = simde_mm256_permutevar8x32_epi32(v1, idx);
    simde__m256i p2 = simde_mm256_permutevar8x32_epi32(v2, idx);
    simde__m256i p3 = simde_mm256_permutevar8x32_epi32(v3, idx);

    /*
     * E output:
     *
     * low128(p0) = x0,  x2,  x4,  x6
     * low128(p1) = x8,  x10, x12, x14
     * low128(p2) = x16, x18, x20, x22
     * low128(p3) = x24, x26, x28, x30
     */
    simde_mm_store_si128((simde__m128i *)&E_in[e + 0], simde_mm256_castsi256_si128(p0));

    simde_mm_store_si128((simde__m128i *)&E_in[e + 4], simde_mm256_castsi256_si128(p1));

    simde_mm_store_si128((simde__m128i *)&E_in[e + 8], simde_mm256_castsi256_si128(p2));

    simde_mm_store_si128((simde__m128i *)&E_in[e + 12], simde_mm256_castsi256_si128(p3));

    /*
     * high128(p0) = x1,  x5,  x3,  x7
     * high128(p1) = x9,  x13, x11, x15
     * high128(p2) = x17, x21, x19, x23
     * high128(p3) = x25, x29, x27, x31
     */
    simde__m128i h0 = simde_mm256_extracti128_si256(p0, 1);
    simde__m128i h1 = simde_mm256_extracti128_si256(p1, 1);
    simde__m128i h2 = simde_mm256_extracti128_si256(p2, 1);
    simde__m128i h3 = simde_mm256_extracti128_si256(p3, 1);

    /*
     * O1:
     * unpacklo_epi64(h0,h1) = x1,  x5,  x9,  x13
     * unpacklo_epi64(h2,h3) = x17, x21, x25, x29
     *
     * O3:
     * unpackhi_epi64(h0,h1) = x3,  x7,  x11, x15
     * unpackhi_epi64(h2,h3) = x19, x23, x27, x31
     */
    simde__m128i o1_0 = simde_mm_unpacklo_epi64(h0, h1);
    simde__m128i o3_0 = simde_mm_unpackhi_epi64(h0, h1);

    simde__m128i o1_1 = simde_mm_unpacklo_epi64(h2, h3);
    simde__m128i o3_1 = simde_mm_unpackhi_epi64(h2, h3);

    simde_mm_store_si128((simde__m128i *)&O1_in[o + 0], o1_0);
    simde_mm_store_si128((simde__m128i *)&O1_in[o + 4], o1_1);

    simde_mm_store_si128((simde__m128i *)&O3_in[o + 0], o3_0);
    simde_mm_store_si128((simde__m128i *)&O3_in[o + 4], o3_1);
  }
}

static void dft_split_radix_pure_simd_core(c16_t *__restrict x, c16_t *__restrict y, c16_t *__restrict work, int N, dft_dir_t dir)
{
  if (N == 4) {
    dft4_void(x, y, dir);
    return;
  }
  if (N == 8) {
    dft8_avx(x, y, dir);
    return;
  }
  if (N == 16) {
    dft16_q15_128(x, y, dir);
    return;
  }
  if (N == 32) {
    dft32_q15_128(x, y, dir);
    return;
  }
  if (N == 64) {
    dft64_avx(x, y, dir);
    return;
  }
  if (N == 128) {
    dft128_dir(x, y, dir);
    return;
  }
  if (N == 256) {
    dft256_radix16_selected(x, y, dir);
    return;
  }
  if (N == 512) {
    dft512_radix8_selected(x, y, dir);
    return;
  }
  if (N == 1024) {
    dft1024_radix16_selected(x, y, dir);
    return;
  }
  AssertFatal(N >= 4 && is_power_of_two_int(N), "Invalid split-radix N=%d\n", N);

  const int half = N >> 1;
  const int quarter = N >> 2;
  c16_t *sub_in = work;
  c16_t *sub_out = work + N;
  c16_t *E = sub_out;
  c16_t *O1 = sub_out + half;
  c16_t *O3 = sub_out + half + quarter;

  pack_split_radix_input_avx2_fused(x, sub_in, N);
  dft_split_radix_pure_simd_core(sub_in, E, work + 2 * N, half, dir);
  dft_split_radix_pure_simd_core(sub_in + half, O1, work + 2 * N, quarter, dir);
  dft_split_radix_pure_simd_core(sub_in + half + quarter, O3, work + 2 * N, quarter, dir);

  const int idx = log2_int((unsigned int)N);
  const sr_twiddle_simd_t *table = dir == DFT_DIR_FORWARD ? &sr_twiddles_fwd[idx] : &sr_twiddles_bwd[idx];
  sr_combine_simd(E, O1, O3, y, N, table, dir);
}

static void dft_split_radix_pure_simd(c16_t *x, c16_t *y, int N, dft_dir_t dir)
{
  if (!sr_twiddle_prepare_power2(N, dir))
    return;
  const size_t work_len = split_radix_work_len_c16(N);

  if (work_len == 0) {
    dft_split_radix_pure_simd_core(x, y, NULL, N, dir);
    return;
  }

  c16_t *work = x86_dft_tls_split_work(work_len);
  if (!work) {
    LOG_E(PHY, "Split-radix workspace allocation failed N=%d work_len=%zu\n", N, work_len);
    return;
  }

  dft_split_radix_pure_simd_core(x, y, work, N, dir);
}

static void dft_split_radix_pure_simd_core_strided(const c16_t *__restrict x,
                                                   int stride,
                                                   c16_t *__restrict y,
                                                   c16_t *__restrict work,
                                                   int N,
                                                   dft_dir_t dir)
{
  if (stride == 1) {
    dft_split_radix_pure_simd_core((c16_t *)x, y, work, N, dir);
    return;
  }
  if (N == 4) {
    c16_t tmp[4] __attribute__((aligned(16)));
    for (int i = 0; i < 4; i++) tmp[i] = x[i * stride];
    dft4_void(tmp, y, dir);
    return;
  }
  if (N == 8) {
    dft8_strided_q15_128(x, stride, y, dir);
    return;
  }
  if (N == 16) {
    dft16_q15_128_strided(x, stride, y, dir);
    return;
  }
  if (N == 32) {
    dft32_q15_128_strided(x, stride, y, dir);
    return;
  }
  if (N == 64) {
    dft64_q15_128_strided(x, stride, y, dir);
    return;
  }
  if (N == 128) {
    dft128_q15_128_strided(x, stride, y, dir);
    return;
  }

  const int half = N >> 1;
  const int quarter = N >> 2;
  c16_t *E = work;
  c16_t *O1 = work + half;
  c16_t *O3 = work + half + quarter;
  c16_t *child_work = work + N;

  dft_split_radix_pure_simd_core_strided(x, stride * 2, E, child_work, half, dir);
  dft_split_radix_pure_simd_core_strided(x + stride, stride * 4, O1, child_work, quarter, dir);
  dft_split_radix_pure_simd_core_strided(x + 3 * stride, stride * 4, O3, child_work, quarter, dir);

  const int idx = log2_int((unsigned int)N);
  const sr_twiddle_simd_t *table = dir == DFT_DIR_FORWARD ? &sr_twiddles_fwd[idx] : &sr_twiddles_bwd[idx];
  sr_combine_simd(E, O1, O3, y, N, table, dir);
}

static void dft_split_radix_pure_simd_strided(const c16_t *x, int stride, c16_t *y, int N, dft_dir_t dir)
{
  if (!sr_twiddle_prepare_power2(N, dir))
    return;
  const size_t work_len = split_radix_work_len_c16(N);

  if (work_len == 0) {
    dft_split_radix_pure_simd_core_strided(x, stride, y, NULL, N, dir);
    return;
  }

  c16_t *work = x86_dft_tls_split_work(work_len);
  if (!work) {
    LOG_E(PHY, "Split-radix workspace allocation failed N=%d work_len=%zu\n", N, work_len);
    return;
  }

  dft_split_radix_pure_simd_core_strided(x, stride, y, work, N, dir);
}

static inline __m128i dft4_avx(__m128i x, dft_dir_t dir)
{
  /*
   * lo = [x0 x1 x0 x1]
   * hi = [x2 x3 x2 x3]
   */
  x = _mm_srai_epi16(x, 1);
  const __m128i lo = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 1, 0));
  const __m128i hi = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 2, 3, 2));

  const __m128i s = _mm_adds_epi16(lo, hi);
  const __m128i d = _mm_subs_epi16(lo, hi);

  const __m128i s_sw = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
  const __m128i d_sw = _mm_shuffle_epi32(d, _MM_SHUFFLE(2, 3, 0, 1));

  const __m128i y0v = _mm_adds_epi16(s, s_sw);
  const __m128i y2v = _mm_subs_epi16(s, s_sw);

  const __m128i y1v = _mm_adds_epi16(d, mul_minus_j_dir_i16_128(d_sw, dir));
  const __m128i y3v = _mm_adds_epi16(d, mul_plus_j_dir_i16_128(d_sw, dir));

  /*
   * y0v lane0 = Y0
   * y1v lane0 = Y1
   * y2v lane0 = Y2
   * y3v lane0 = Y3
   */
  const __m128i y01 = _mm_unpacklo_epi32(y0v, y1v); // [Y0 Y1 ... ...]
  const __m128i y23 = _mm_unpacklo_epi32(y2v, y3v); // [Y2 Y3 ... ...]

  return _mm_unpacklo_epi64(y01, y23); // [Y0 Y1 Y2 Y3]
}

static inline void dft4_void(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  const __m128i x = _mm_loadu_si128((const __m128i *)src);

  const __m128i y = dft4_avx(x, dir);

  _mm_storeu_si128((__m128i *)dst, y);
}

static inline void dft8_avx(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  /*
   * v0 = [x0 x1 x2 x3]
   * v1 = [x4 x5 x6 x7]
   */
  const __m128i v0 = _mm_loadu_si128((const __m128i *)(src + 0));

  const __m128i v1 = _mm_loadu_si128((const __m128i *)(src + 4));

  /*
   * E_in = [x0 x2 x4 x6]
   * O_in = [x1 x3 x5 x7]
   */
  const __m128i v0_even = _mm_shuffle_epi32(v0, _MM_SHUFFLE(2, 0, 2, 0));

  const __m128i v1_even = _mm_shuffle_epi32(v1, _MM_SHUFFLE(2, 0, 2, 0));

  const __m128i v0_odd = _mm_shuffle_epi32(v0, _MM_SHUFFLE(3, 1, 3, 1));

  const __m128i v1_odd = _mm_shuffle_epi32(v1, _MM_SHUFFLE(3, 1, 3, 1));

  const __m128i E_in = _mm_unpacklo_epi64(v0_even, v1_even); // [x0 x2 x4 x6]

  const __m128i O_in = _mm_unpacklo_epi64(v0_odd, v1_odd); // [x1 x3 x5 x7]

  /*
   * E = DFT4(x0, x2, x4, x6)
   * O = DFT4(x1, x3, x5, x7)
   */
  const __m128i E = q15_mul_i16_128(dft4_avx(E_in, dir), Q15_INV_SQRT2);
  const __m128i O = dft4_avx(O_in, dir);

  /*
   * Twiddles W8 forward :
   *
   * W8^0 =  1 + j0
   * W8^1 =  c - jc
   * W8^2 =  0 - j1
   * W8^3 = -c - jc
   *
   * c = 1/sqrt(2) = 23170
   *
   * w_re_re     = [wr0 wr0 wr1 wr1 wr2 wr2 wr3 wr3]
   * w_im_signed = [-wi0 wi0 -wi1 wi1 -wi2 wi2 -wi3 wi3]
   */
  const __m128i W8_RE_RE = _mm_setr_epi16(23170, 23170, Q15_HALF, Q15_HALF, 0, 0, -Q15_HALF, -Q15_HALF);

  const __m128i W8_IM_SIGNED_FWD = _mm_setr_epi16(0, 0, Q15_HALF, -Q15_HALF, 23170, -23170, Q15_HALF, -Q15_HALF);

  const __m128i W8_IM_SIGNED = twiddle_im_dir_128(W8_IM_SIGNED_FWD, dir);

  /*
   * T[k] = W8^k * O[k], k = 0..3
   */
  const __m128i T = complex_mul4_prepack_q15_128(O, W8_RE_RE, W8_IM_SIGNED);

  /*
   * Combine radix-2 :
   *
   * Y[k]     = E[k] + T[k]
   * Y[k + 4] = E[k] - T[k]
   */
  const __m128i Y03 = _mm_adds_epi16(E, T); // [Y0 Y1 Y2 Y3]
  const __m128i Y47 = _mm_subs_epi16(E, T); // [Y4 Y5 Y6 Y7]

  _mm_storeu_si128((__m128i *)(dst + 0), Y03);
  _mm_storeu_si128((__m128i *)(dst + 4), Y47);
}

static inline void dft8_strided_q15_128(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir)
{
  c16_t tmp[8] __attribute__((aligned(16)));

  for (int i = 0; i < 8; i++) {
    tmp[i] = src[i * stride];
  }

  dft8_avx(tmp, dst, dir);
}

/*
 * Twiddles for DFT16 radix-4 combine.
 *
 * Format:
 *   RE_RE     = [ re0, re0, re1, re1, re2, re2, re3, re3 ]
 *   IM_SIGNED = [-im0, im0,-im1, im1,-im2, im2,-im3, im3]
 *
 * Forward:
 *   W16^k = cos(2*pi*k/16) - j sin(2*pi*k/16)
 */

/* W16^(1*j), j = 0..3 */
static const int16_t W16_1_RE_RE[8] __attribute__((
    aligned(16))) = {Q15_ONE, Q15_ONE, Q15_COS_PI_8, Q15_COS_PI_8, Q15_INV_SQRT2, Q15_INV_SQRT2, Q15_SIN_PI_8, Q15_SIN_PI_8};

static const int16_t W16_1_IM_SIGNED[8]
    __attribute__((aligned(16))) = {0, 0, Q15_SIN_PI_8, -Q15_SIN_PI_8, Q15_INV_SQRT2, -Q15_INV_SQRT2, Q15_COS_PI_8, -Q15_COS_PI_8};

/* W16^(2*j), j = 0..3 */
static const int16_t W16_2_RE_RE[8]
    __attribute__((aligned(16))) = {Q15_ONE, Q15_ONE, Q15_INV_SQRT2, Q15_INV_SQRT2, 0, 0, -Q15_INV_SQRT2, -Q15_INV_SQRT2};

static const int16_t W16_2_IM_SIGNED[8]
    __attribute__((aligned(16))) = {0, 0, Q15_INV_SQRT2, -Q15_INV_SQRT2, Q15_ONE, -Q15_ONE, Q15_INV_SQRT2, -Q15_INV_SQRT2};

/* W16^(3*j), j = 0..3 */
static const int16_t W16_3_RE_RE[8] __attribute__((
    aligned(16))) = {Q15_ONE, Q15_ONE, Q15_SIN_PI_8, Q15_SIN_PI_8, -Q15_INV_SQRT2, -Q15_INV_SQRT2, -Q15_COS_PI_8, -Q15_COS_PI_8};

static const int16_t W16_3_IM_SIGNED[8]
    __attribute__((aligned(16))) = {0, 0, Q15_COS_PI_8, -Q15_COS_PI_8, Q15_INV_SQRT2, -Q15_INV_SQRT2, -Q15_SIN_PI_8, Q15_SIN_PI_8};

static inline void dft4x4_q15_128(const __m128i x0,
                                  const __m128i x1,
                                  const __m128i x2,
                                  const __m128i x3,
                                  __m128i *Y0,
                                  __m128i *Y1,
                                  __m128i *Y2,
                                  __m128i *Y3,
                                  dft_dir_t dir)
{
  const __m128i x0s = _mm_srai_epi16(x0, 1);
  const __m128i x1s = _mm_srai_epi16(x1, 1);
  const __m128i x2s = _mm_srai_epi16(x2, 1);
  const __m128i x3s = _mm_srai_epi16(x3, 1);

  const __m128i s02 = _mm_adds_epi16(x0s, x2s);
  const __m128i d02 = _mm_subs_epi16(x0s, x2s);

  const __m128i s13 = _mm_adds_epi16(x1s, x3s);
  const __m128i d13 = _mm_subs_epi16(x1s, x3s);

  *Y0 = _mm_adds_epi16(s02, s13);
  *Y2 = _mm_subs_epi16(s02, s13);

  /*
   * Forward DFT4:
   * Y1 = d02 - j*d13
   * Y3 = d02 + j*d13
   */
  *Y1 = _mm_adds_epi16(d02, mul_minus_j_dir_i16_128(d13, dir));
  *Y3 = _mm_adds_epi16(d02, mul_plus_j_dir_i16_128(d13, dir));
}
static inline void transpose4_complex_i16_128(__m128i *Y0, __m128i *Y1, __m128i *Y2, __m128i *Y3)
{
  const __m128i a = *Y0; // [a0 a1 a2 a3]
  const __m128i b = *Y1; // [b0 b1 b2 b3]
  const __m128i c = *Y2; // [c0 c1 c2 c3]
  const __m128i d = *Y3; // [d0 d1 d2 d3]

  const __m128i ab_lo = _mm_unpacklo_epi32(a, b); // [a0 b0 a1 b1]
  const __m128i ab_hi = _mm_unpackhi_epi32(a, b); // [a2 b2 a3 b3]

  const __m128i cd_lo = _mm_unpacklo_epi32(c, d); // [c0 d0 c1 d1]
  const __m128i cd_hi = _mm_unpackhi_epi32(c, d); // [c2 d2 c3 d3]

  *Y0 = _mm_unpacklo_epi64(ab_lo, cd_lo); // [a0 b0 c0 d0]
  *Y1 = _mm_unpackhi_epi64(ab_lo, cd_lo); // [a1 b1 c1 d1]
  *Y2 = _mm_unpacklo_epi64(ab_hi, cd_hi); // [a2 b2 c2 d2]
  *Y3 = _mm_unpackhi_epi64(ab_hi, cd_hi); // [a3 b3 c3 d3]
}

static inline void combine16_q15_128(const __m128i H[4], c16_t *dst, dft_dir_t dir)
{
  /*
   * H[0] = [H0[0] H0[1] H0[2] H0[3]]
   * H[1] = [H1[0] H1[1] H1[2] H1[3]]
   * H[2] = [H2[0] H2[1] H2[2] H2[3]]
   * H[3] = [H3[0] H3[1] H3[2] H3[3]]
   */
  const __m128i W1_RE_RE = _mm_load_si128((const __m128i *)W16_1_RE_RE);

  const __m128i W1_IM_SIGNED = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W16_1_IM_SIGNED), dir);

  const __m128i W2_RE_RE = _mm_load_si128((const __m128i *)W16_2_RE_RE);

  const __m128i W2_IM_SIGNED = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W16_2_IM_SIGNED), dir);

  const __m128i W3_RE_RE = _mm_load_si128((const __m128i *)W16_3_RE_RE);

  const __m128i W3_IM_SIGNED = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W16_3_IM_SIGNED), dir);
  const __m128i A0 = H[0];

  const __m128i A1 = complex_mul4_prepack_q15_128(H[1], W1_RE_RE, W1_IM_SIGNED);

  const __m128i A2 = complex_mul4_prepack_q15_128(H[2], W2_RE_RE, W2_IM_SIGNED);

  const __m128i A3 = complex_mul4_prepack_q15_128(H[3], W3_RE_RE, W3_IM_SIGNED);

  __m128i t0 = A0;
  __m128i t1 = A1;
  __m128i t2 = A2;
  __m128i t3 = A3;

  transpose4_complex_i16_128(&t0, &t1, &t2, &t3);

  __m128i Y0, Y1, Y2, Y3;

  dft4x4_q15_128(t0, t1, t2, t3, &Y0, &Y1, &Y2, &Y3, dir);

  _mm_storeu_si128((__m128i *)(dst + 0), Y0);
  _mm_storeu_si128((__m128i *)(dst + 4), Y1);
  _mm_storeu_si128((__m128i *)(dst + 8), Y2);
  _mm_storeu_si128((__m128i *)(dst + 12), Y3);
}

static inline void dft16_q15_128(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  __m128i H[4] __attribute__((aligned(16)));

  const __m128i x0 = _mm_loadu_si128((const __m128i *)(src + 0));

  const __m128i x1 = _mm_loadu_si128((const __m128i *)(src + 4));

  const __m128i x2 = _mm_loadu_si128((const __m128i *)(src + 8));

  const __m128i x3 = _mm_loadu_si128((const __m128i *)(src + 12));

  dft4x4_q15_128(x0, x1, x2, x3, &H[0], &H[1], &H[2], &H[3], dir);

  combine16_q15_128(H, dst, dir);
}

/*
 * Format:
 *   RE_RE     = [ re0, re0, re1, re1, re2, re2, re3, re3 ]
 *   IM_SIGNED = [-im0, im0,-im1, im1,-im2, im2,-im3, im3]
 *
 * Forward:
 *   W12^k = cos(2*pi*k/12) - j sin(2*pi*k/12)
 *
 */

/* W12^k, k = 0..3 */
static const int16_t W12_R3_W1_RE_RE[8] __attribute__((aligned(16))) = {Q15_INV_SQRT3,
                                                                        Q15_INV_SQRT3, /* 1 / sqrt3 */
                                                                        Q15_HALF,
                                                                        Q15_HALF, /* cos(pi/6) / sqrt3 = 1/2 */
                                                                        Q15_INV_2SQRT3,
                                                                        Q15_INV_2SQRT3, /* cos(pi/3) / sqrt3 */
                                                                        0,
                                                                        0};

static const int16_t W12_R3_W1_IM_SIGNED[8]
    __attribute__((aligned(16))) = {0, 0, Q15_INV_2SQRT3, -Q15_INV_2SQRT3, Q15_HALF, -Q15_HALF, Q15_INV_SQRT3, -Q15_INV_SQRT3};

/* W12^(2k), k = 0..3 */
static const int16_t W12_R3_W2_RE_RE[8] __attribute__((aligned(16))) = {Q15_INV_SQRT3,
                                                                        Q15_INV_SQRT3,
                                                                        Q15_INV_2SQRT3,
                                                                        Q15_INV_2SQRT3,
                                                                        -Q15_INV_2SQRT3,
                                                                        -Q15_INV_2SQRT3,
                                                                        -Q15_INV_SQRT3,
                                                                        -Q15_INV_SQRT3};

static const int16_t W12_R3_W2_IM_SIGNED[8] __attribute__((aligned(16))) = {0, 0, Q15_HALF, -Q15_HALF, Q15_HALF, -Q15_HALF, 0, 0};

static inline __m128i pack3_complex_plus_zero_c16(const c16_t a, const c16_t b, const c16_t c)
{
  return _mm_setr_epi16(a.r, a.i, b.r, b.i, c.r, c.i, 0, 0);
}

#if defined(__GNUC__)
#define OAI_DFT_SMALL_HOT __attribute__((hot, aligned(32), section(".text.hot.oai_dft_small")))
#else
#define OAI_DFT_SMALL_HOT
#endif

OAI_DFT_SMALL_HOT void dft16(int16_t *x, int16_t *y, uint8_t scale_flag)
{
  const c16_t *src = (const c16_t *)x;
  c16_t *dst = (c16_t *)y;

  (void)scale_flag;

  dft16_q15_128(src, dst, DFT_DIR_FORWARD);
}

static inline void dft12_q15_128(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  /*
   * lane 0 : src[0], src[3], src[6],  src[9]
   * lane 1 : src[1], src[4], src[7],  src[10]
   * lane 2 : src[2], src[5], src[8],  src[11]
   * lane 3 : dummy
   */
  const __m128i x0 = pack3_complex_plus_zero_c16(src[0], src[1], src[2]);
  const __m128i x1 = pack3_complex_plus_zero_c16(src[3], src[4], src[5]);
  const __m128i x2 = pack3_complex_plus_zero_c16(src[6], src[7], src[8]);
  const __m128i x3 = pack3_complex_plus_zero_c16(src[9], src[10], src[11]);

  __m128i H0, H1, H2, H3;

  /*
   * dft4x4_q15_128 applies the 1/2 unitary scaling.
   * H0 = [F0[0], F1[0], F2[0], dummy]
   * H1 = [F0[1], F1[1], F2[1], dummy]
   * H2 = [F0[2], F1[2], F2[2], dummy]
   * H3 = [F0[3], F1[3], F2[3], dummy]
   */
  dft4x4_q15_128(x0, x1, x2, x3, &H0, &H1, &H2, &H3, dir);

  /*
   * After transposition:
   * H0 = A  = [F0[0], F0[1], F0[2], F0[3]]
   * H1 = X1 = [F1[0], F1[1], F1[2], F1[3]]
   * H2 = X2 = [F2[0], F2[1], F2[2], F2[3]]
   * H3 = dummy
   */
  transpose4_complex_i16_128(&H0, &H1, &H2, &H3);

  const __m128i A = H0;
  const __m128i X1 = H1;
  const __m128i X2 = H2;

  const __m128i W1_RE = _mm_load_si128((const __m128i *)W12_R3_W1_RE_RE);

  const __m128i W1_IM = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W12_R3_W1_IM_SIGNED), dir);

  const __m128i W2_RE = _mm_load_si128((const __m128i *)W12_R3_W2_RE_RE);

  const __m128i W2_IM = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W12_R3_W2_IM_SIGNED), dir);

  /* The combined scale is 1/2 * 1/sqrt(3) = 1/sqrt(12). */
  const __m128i As = q15_mul_i16_128(A, Q15_INV_SQRT3);

  /*
   * B[k] = W12^k    * X1[k] / sqrt(3)
   * C[k] = W12^(2k) * X2[k] / sqrt(3)
   */
  const __m128i B = complex_mul4_prepack_q15_128(X1, W1_RE, W1_IM);

  const __m128i C = complex_mul4_prepack_q15_128(X2, W2_RE, W2_IM);

  const __m128i S = _mm_adds_epi16(B, C);
  const __m128i D = _mm_subs_epi16(B, C);

  /*
   * Y0 = A + B + C
   */
  const __m128i Y0 = _mm_adds_epi16(As, S);

  /*
   * base = A - 1/2 * (B + C)
   */
  const __m128i halfS = q15_mul_i16_128(S, Q15_HALF);

  const __m128i base = _mm_subs_epi16(As, halfS);

  /*
   * c3D = sqrt(3)/2 * (B - C)
   */
  const __m128i c3D = q15_mul_i16_128(D, Q15_SQRT3_OVER_2);

  /*
   * Forward radix-3:
   *
   * Y1 = base - j*c3D
   * Y2 = base + j*c3D
   */
  const __m128i Y1 = _mm_adds_epi16(base, mul_minus_j_dir_i16_128(c3D, dir));

  const __m128i Y2 = _mm_adds_epi16(base, mul_plus_j_dir_i16_128(c3D, dir));

  /*
   * size = 4
   *
   * dst[0..3]   = Y0
   * dst[4..7]   = Y1
   * dst[8..11]  = Y2
   */
  _mm_storeu_si128((__m128i *)(dst + 0), Y0);
  _mm_storeu_si128((__m128i *)(dst + 4), Y1);
  _mm_storeu_si128((__m128i *)(dst + 8), Y2);
}

OAI_DFT_SMALL_HOT void dft12(int16_t *x, int16_t *y, uint8_t scale_flag)
{
  const c16_t *src = (const c16_t *)x;
  c16_t *dst = (c16_t *)y;

  (void)scale_flag;

  dft12_q15_128(src, dst, DFT_DIR_FORWARD);
}

static inline void dft12_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir)
{
  const __m128i x0 = pack3_complex_plus_zero_c16(src[0 * stride], src[1 * stride], src[2 * stride]);

  const __m128i x1 = pack3_complex_plus_zero_c16(src[3 * stride], src[4 * stride], src[5 * stride]);

  const __m128i x2 = pack3_complex_plus_zero_c16(src[6 * stride], src[7 * stride], src[8 * stride]);

  const __m128i x3 = pack3_complex_plus_zero_c16(src[9 * stride], src[10 * stride], src[11 * stride]);

  __m128i H0, H1, H2, H3;

  dft4x4_q15_128(x0, x1, x2, x3, &H0, &H1, &H2, &H3, dir);

  transpose4_complex_i16_128(&H0, &H1, &H2, &H3);

  const __m128i A = H0;
  const __m128i X1 = H1;
  const __m128i X2 = H2;

  const __m128i W1_RE = _mm_load_si128((const __m128i *)W12_R3_W1_RE_RE);

  const __m128i W1_IM = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W12_R3_W1_IM_SIGNED), dir);

  const __m128i W2_RE = _mm_load_si128((const __m128i *)W12_R3_W2_RE_RE);

  const __m128i W2_IM = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W12_R3_W2_IM_SIGNED), dir);

  const __m128i As = q15_mul_i16_128(A, Q15_INV_SQRT3);

  const __m128i B = complex_mul4_prepack_q15_128(X1, W1_RE, W1_IM);

  const __m128i C = complex_mul4_prepack_q15_128(X2, W2_RE, W2_IM);

  const __m128i S = _mm_adds_epi16(B, C);
  const __m128i D = _mm_subs_epi16(B, C);

  const __m128i Y0 = _mm_adds_epi16(As, S);

  const __m128i halfS = q15_mul_i16_128(S, Q15_HALF);

  const __m128i base = _mm_subs_epi16(As, halfS);

  const __m128i c3D = q15_mul_i16_128(D, Q15_SQRT3_OVER_2);

  const __m128i Y1 = _mm_adds_epi16(base, mul_minus_j_dir_i16_128(c3D, dir));

  const __m128i Y2 = _mm_adds_epi16(base, mul_plus_j_dir_i16_128(c3D, dir));

  _mm_storeu_si128((__m128i *)(dst + 0), Y0);
  _mm_storeu_si128((__m128i *)(dst + 4), Y1);
  _mm_storeu_si128((__m128i *)(dst + 8), Y2);
}

/* W32^(1*j), j = 0..7 */

static const int16_t W32_1_RE_RE_LO[8] __attribute__((aligned(16))) = {DFT32_Q15_ONE,
                                                                       DFT32_Q15_ONE,
                                                                       DFT32_Q15_COS_PI_16,
                                                                       DFT32_Q15_COS_PI_16,
                                                                       Q15_COS_PI_8,
                                                                       Q15_COS_PI_8,
                                                                       DFT32_Q15_COS_3PI_16,
                                                                       DFT32_Q15_COS_3PI_16};

static const int16_t W32_1_IM_SIGNED_LO[8] __attribute__((aligned(16))) =
    {0, 0, DFT32_Q15_SIN_PI_16, -DFT32_Q15_SIN_PI_16, Q15_SIN_PI_8, -Q15_SIN_PI_8, DFT32_Q15_SIN_3PI_16, -DFT32_Q15_SIN_3PI_16};

static const int16_t W32_1_RE_RE_HI[8] __attribute__((aligned(16))) = {Q15_INV_SQRT2,
                                                                       Q15_INV_SQRT2,
                                                                       DFT32_Q15_SIN_3PI_16,
                                                                       DFT32_Q15_SIN_3PI_16,
                                                                       Q15_SIN_PI_8,
                                                                       Q15_SIN_PI_8,
                                                                       DFT32_Q15_SIN_PI_16,
                                                                       DFT32_Q15_SIN_PI_16};

static const int16_t W32_1_IM_SIGNED_HI[8] __attribute__((aligned(16))) = {Q15_INV_SQRT2,
                                                                           -Q15_INV_SQRT2,
                                                                           DFT32_Q15_COS_3PI_16,
                                                                           -DFT32_Q15_COS_3PI_16,
                                                                           Q15_COS_PI_8,
                                                                           -Q15_COS_PI_8,
                                                                           DFT32_Q15_COS_PI_16,
                                                                           -DFT32_Q15_COS_PI_16};

/* W32^(2*j), j = 0..7 */

static const int16_t W32_2_RE_RE_LO[8] __attribute__((aligned(
    16))) = {DFT32_Q15_ONE, DFT32_Q15_ONE, Q15_COS_PI_8, Q15_COS_PI_8, Q15_INV_SQRT2, Q15_INV_SQRT2, Q15_SIN_PI_8, Q15_SIN_PI_8};

static const int16_t W32_2_IM_SIGNED_LO[8]
    __attribute__((aligned(16))) = {0, 0, Q15_SIN_PI_8, -Q15_SIN_PI_8, Q15_INV_SQRT2, -Q15_INV_SQRT2, Q15_COS_PI_8, -Q15_COS_PI_8};

static const int16_t W32_2_RE_RE_HI[8] __attribute__((
    aligned(16))) = {0, 0, -Q15_SIN_PI_8, -Q15_SIN_PI_8, -Q15_INV_SQRT2, -Q15_INV_SQRT2, -Q15_COS_PI_8, -Q15_COS_PI_8};

static const int16_t W32_2_IM_SIGNED_HI[8] __attribute__((aligned(16))) =
    {DFT32_Q15_ONE, -DFT32_Q15_ONE, Q15_COS_PI_8, -Q15_COS_PI_8, Q15_INV_SQRT2, -Q15_INV_SQRT2, Q15_SIN_PI_8, -Q15_SIN_PI_8};

/* W32^(3*j), j = 0..7 */

static const int16_t W32_3_RE_RE_LO[8] __attribute__((aligned(16))) = {DFT32_Q15_ONE,
                                                                       DFT32_Q15_ONE,
                                                                       DFT32_Q15_COS_3PI_16,
                                                                       DFT32_Q15_COS_3PI_16,
                                                                       Q15_SIN_PI_8,
                                                                       Q15_SIN_PI_8,
                                                                       -DFT32_Q15_SIN_PI_16,
                                                                       -DFT32_Q15_SIN_PI_16};

static const int16_t W32_3_IM_SIGNED_LO[8] __attribute__((aligned(16))) =
    {0, 0, DFT32_Q15_SIN_3PI_16, -DFT32_Q15_SIN_3PI_16, Q15_COS_PI_8, -Q15_COS_PI_8, DFT32_Q15_COS_PI_16, -DFT32_Q15_COS_PI_16};

static const int16_t W32_3_RE_RE_HI[8] __attribute__((aligned(16))) = {-Q15_INV_SQRT2,
                                                                       -Q15_INV_SQRT2,
                                                                       -DFT32_Q15_COS_PI_16,
                                                                       -DFT32_Q15_COS_PI_16,
                                                                       -Q15_COS_PI_8,
                                                                       -Q15_COS_PI_8,
                                                                       -DFT32_Q15_SIN_3PI_16,
                                                                       -DFT32_Q15_SIN_3PI_16};

static const int16_t W32_3_IM_SIGNED_HI[8] __attribute__((aligned(16))) = {Q15_INV_SQRT2,
                                                                           -Q15_INV_SQRT2,
                                                                           DFT32_Q15_SIN_PI_16,
                                                                           -DFT32_Q15_SIN_PI_16,
                                                                           -Q15_SIN_PI_8,
                                                                           Q15_SIN_PI_8,
                                                                           -DFT32_Q15_COS_3PI_16,
                                                                           DFT32_Q15_COS_3PI_16};

static inline void dft8x4_q15_128(const __m128i x0,
                                  const __m128i x1,
                                  const __m128i x2,
                                  const __m128i x3,
                                  const __m128i x4,
                                  const __m128i x5,
                                  const __m128i x6,
                                  const __m128i x7,
                                  __m128i *Y0,
                                  __m128i *Y1,
                                  __m128i *Y2,
                                  __m128i *Y3,
                                  __m128i *Y4,
                                  __m128i *Y5,
                                  __m128i *Y6,
                                  __m128i *Y7,
                                  dft_dir_t dir)
{
  __m128i E0, E1, E2, E3;
  __m128i O0, O1, O2, O3;

  dft4x4_q15_128(x0, x2, x4, x6, &E0, &E1, &E2, &E3, dir);
  dft4x4_q15_128(x1, x3, x5, x7, &O0, &O1, &O2, &O3, dir);

  const __m128i E0s = q15_mul_i16_128(E0, Q15_INV_SQRT2);
  const __m128i E1s = q15_mul_i16_128(E1, Q15_INV_SQRT2);
  const __m128i E2s = q15_mul_i16_128(E2, Q15_INV_SQRT2);
  const __m128i E3s = q15_mul_i16_128(E3, Q15_INV_SQRT2);

  /*
   * W8 scaled by 1/sqrt(2):
   *
   * W8^0 / sqrt(2) =  1/sqrt(2)
   * W8^1 / sqrt(2) =  1/2 - j1/2
   * W8^2 / sqrt(2) =  0 - j1/sqrt(2)
   * W8^3 / sqrt(2) = -1/2 - j1/2
   */
  const __m128i T0 = q15_mul_i16_128(O0, Q15_INV_SQRT2);

  const __m128i T1 = complex_mul4_bcast_q15_128(O1, Q15_HALF, twiddle_im_scalar_dir_i16(-Q15_HALF, dir));

  const __m128i T2 = complex_mul4_bcast_q15_128(O2, 0, twiddle_im_scalar_dir_i16(-Q15_INV_SQRT2, dir));

  const __m128i T3 = complex_mul4_bcast_q15_128(O3, -Q15_HALF, twiddle_im_scalar_dir_i16(-Q15_HALF, dir));

  *Y0 = _mm_adds_epi16(E0s, T0);
  *Y4 = _mm_subs_epi16(E0s, T0);

  *Y1 = _mm_adds_epi16(E1s, T1);
  *Y5 = _mm_subs_epi16(E1s, T1);

  *Y2 = _mm_adds_epi16(E2s, T2);
  *Y6 = _mm_subs_epi16(E2s, T2);

  *Y3 = _mm_adds_epi16(E3s, T3);
  *Y7 = _mm_subs_epi16(E3s, T3);
}

#define DFT32_AVX2_TW __attribute__((section(".rodata.oai_dft32_avx2"), aligned(32)))

static const int16_t W32_1_RE_RE_256[16] DFT32_AVX2_TW = {
    DFT32_Q15_ONE, DFT32_Q15_ONE,
    DFT32_Q15_COS_PI_16, DFT32_Q15_COS_PI_16,
    Q15_COS_PI_8, Q15_COS_PI_8,
    DFT32_Q15_COS_3PI_16, DFT32_Q15_COS_3PI_16,
    Q15_INV_SQRT2, Q15_INV_SQRT2,
    DFT32_Q15_SIN_3PI_16, DFT32_Q15_SIN_3PI_16,
    Q15_SIN_PI_8, Q15_SIN_PI_8,
    DFT32_Q15_SIN_PI_16, DFT32_Q15_SIN_PI_16};

static const int16_t W32_1_IM_SIGNED_256[16] DFT32_AVX2_TW = {
    0, 0,
    DFT32_Q15_SIN_PI_16, -DFT32_Q15_SIN_PI_16,
    Q15_SIN_PI_8, -Q15_SIN_PI_8,
    DFT32_Q15_SIN_3PI_16, -DFT32_Q15_SIN_3PI_16,
    Q15_INV_SQRT2, -Q15_INV_SQRT2,
    DFT32_Q15_COS_3PI_16, -DFT32_Q15_COS_3PI_16,
    Q15_COS_PI_8, -Q15_COS_PI_8,
    DFT32_Q15_COS_PI_16, -DFT32_Q15_COS_PI_16};

static const int16_t W32_2_RE_RE_256[16] DFT32_AVX2_TW = {
    DFT32_Q15_ONE, DFT32_Q15_ONE,
    Q15_COS_PI_8, Q15_COS_PI_8,
    Q15_INV_SQRT2, Q15_INV_SQRT2,
    Q15_SIN_PI_8, Q15_SIN_PI_8,
    0, 0,
    -Q15_SIN_PI_8, -Q15_SIN_PI_8,
    -Q15_INV_SQRT2, -Q15_INV_SQRT2,
    -Q15_COS_PI_8, -Q15_COS_PI_8};

static const int16_t W32_2_IM_SIGNED_256[16] DFT32_AVX2_TW = {
    0, 0,
    Q15_SIN_PI_8, -Q15_SIN_PI_8,
    Q15_INV_SQRT2, -Q15_INV_SQRT2,
    Q15_COS_PI_8, -Q15_COS_PI_8,
    DFT32_Q15_ONE, -DFT32_Q15_ONE,
    Q15_COS_PI_8, -Q15_COS_PI_8,
    Q15_INV_SQRT2, -Q15_INV_SQRT2,
    Q15_SIN_PI_8, -Q15_SIN_PI_8};

static const int16_t W32_3_RE_RE_256[16] DFT32_AVX2_TW = {
    DFT32_Q15_ONE, DFT32_Q15_ONE,
    DFT32_Q15_COS_3PI_16, DFT32_Q15_COS_3PI_16,
    Q15_SIN_PI_8, Q15_SIN_PI_8,
    -DFT32_Q15_SIN_PI_16, -DFT32_Q15_SIN_PI_16,
    -Q15_INV_SQRT2, -Q15_INV_SQRT2,
    -DFT32_Q15_COS_PI_16, -DFT32_Q15_COS_PI_16,
    -Q15_COS_PI_8, -Q15_COS_PI_8,
    -DFT32_Q15_SIN_3PI_16, -DFT32_Q15_SIN_3PI_16};

static const int16_t W32_3_IM_SIGNED_256[16] DFT32_AVX2_TW = {
    0, 0,
    DFT32_Q15_SIN_3PI_16, -DFT32_Q15_SIN_3PI_16,
    Q15_COS_PI_8, -Q15_COS_PI_8,
    DFT32_Q15_COS_PI_16, -DFT32_Q15_COS_PI_16,
    Q15_INV_SQRT2, -Q15_INV_SQRT2,
    DFT32_Q15_SIN_PI_16, -DFT32_Q15_SIN_PI_16,
    -Q15_SIN_PI_8, Q15_SIN_PI_8,
    -DFT32_Q15_COS_3PI_16, DFT32_Q15_COS_3PI_16};

#undef DFT32_AVX2_TW

static inline __m256i twiddle_im_dir_256(__m256i w_im_signed, dft_dir_t dir)
{
  return (dir == DFT_DIR_FORWARD) ? w_im_signed : _mm256_sub_epi16(_mm256_setzero_si256(), w_im_signed);
}

/* Two independent 4x4 transposes, one in each 128-bit lane. Complex Q15
 * samples are treated as 32-bit lanes. */
static inline void transpose4_complex_i16_256x2_shuffle(__m256i *Y0, __m256i *Y1, __m256i *Y2, __m256i *Y3)
{
  const __m256 a = _mm256_castsi256_ps(*Y0);
  const __m256 b = _mm256_castsi256_ps(*Y1);
  const __m256 c = _mm256_castsi256_ps(*Y2);
  const __m256 d = _mm256_castsi256_ps(*Y3);

  const __m256 ab_lo = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 ab_hi = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 3, 2));
  const __m256 cd_lo = _mm256_shuffle_ps(c, d, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 cd_hi = _mm256_shuffle_ps(c, d, _MM_SHUFFLE(3, 2, 3, 2));

  *Y0 = _mm256_castps_si256(_mm256_shuffle_ps(ab_lo, cd_lo, _MM_SHUFFLE(2, 0, 2, 0)));
  *Y1 = _mm256_castps_si256(_mm256_shuffle_ps(ab_lo, cd_lo, _MM_SHUFFLE(3, 1, 3, 1)));
  *Y2 = _mm256_castps_si256(_mm256_shuffle_ps(ab_hi, cd_hi, _MM_SHUFFLE(2, 0, 2, 0)));
  *Y3 = _mm256_castps_si256(_mm256_shuffle_ps(ab_hi, cd_hi, _MM_SHUFFLE(3, 1, 3, 1)));
}

static inline void combine32_q15_128(const __m128i H_lo[4], const __m128i H_hi[4], c16_t *dst, dft_dir_t dir)
{
  const __m128i W1_RE_LO = _mm_load_si128((const __m128i *)W32_1_RE_RE_LO);
  const __m128i W1_IM_LO = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W32_1_IM_SIGNED_LO), dir);

  const __m128i W1_RE_HI = _mm_load_si128((const __m128i *)W32_1_RE_RE_HI);
  const __m128i W1_IM_HI = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W32_1_IM_SIGNED_HI), dir);

  const __m128i W2_RE_LO = _mm_load_si128((const __m128i *)W32_2_RE_RE_LO);
  const __m128i W2_IM_LO = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W32_2_IM_SIGNED_LO), dir);

  const __m128i W2_RE_HI = _mm_load_si128((const __m128i *)W32_2_RE_RE_HI);
  const __m128i W2_IM_HI = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W32_2_IM_SIGNED_HI), dir);

  const __m128i W3_RE_LO = _mm_load_si128((const __m128i *)W32_3_RE_RE_LO);
  const __m128i W3_IM_LO = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W32_3_IM_SIGNED_LO), dir);

  const __m128i W3_RE_HI = _mm_load_si128((const __m128i *)W32_3_RE_RE_HI);
  const __m128i W3_IM_HI = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W32_3_IM_SIGNED_HI), dir);

  const __m128i A0_lo = H_lo[0];
  const __m128i A0_hi = H_hi[0];

  const __m128i A1_lo = complex_mul4_prepack_q15_128(H_lo[1], W1_RE_LO, W1_IM_LO);
  const __m128i A1_hi = complex_mul4_prepack_q15_128(H_hi[1], W1_RE_HI, W1_IM_HI);

  const __m128i A2_lo = complex_mul4_prepack_q15_128(H_lo[2], W2_RE_LO, W2_IM_LO);
  const __m128i A2_hi = complex_mul4_prepack_q15_128(H_hi[2], W2_RE_HI, W2_IM_HI);

  const __m128i A3_lo = complex_mul4_prepack_q15_128(H_lo[3], W3_RE_LO, W3_IM_LO);
  const __m128i A3_hi = complex_mul4_prepack_q15_128(H_hi[3], W3_RE_HI, W3_IM_HI);

  __m128i lo0 = A0_lo;
  __m128i lo1 = A1_lo;
  __m128i lo2 = A2_lo;
  __m128i lo3 = A3_lo;

  __m128i hi0 = A0_hi;
  __m128i hi1 = A1_hi;
  __m128i hi2 = A2_hi;
  __m128i hi3 = A3_hi;

  transpose4_complex_i16_128(&lo0, &lo1, &lo2, &lo3);
  transpose4_complex_i16_128(&hi0, &hi1, &hi2, &hi3);

  __m128i Y0, Y1, Y2, Y3;
  __m128i Y4, Y5, Y6, Y7;

  dft8x4_q15_128(lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3, &Y0, &Y1, &Y2, &Y3, &Y4, &Y5, &Y6, &Y7, dir);

  _mm_storeu_si128((__m128i *)(dst + 0), Y0);
  _mm_storeu_si128((__m128i *)(dst + 4), Y1);
  _mm_storeu_si128((__m128i *)(dst + 8), Y2);
  _mm_storeu_si128((__m128i *)(dst + 12), Y3);

  _mm_storeu_si128((__m128i *)(dst + 16), Y4);
  _mm_storeu_si128((__m128i *)(dst + 20), Y5);
  _mm_storeu_si128((__m128i *)(dst + 24), Y6);
  _mm_storeu_si128((__m128i *)(dst + 28), Y7);
}

static inline void dft32_q15_128(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  dft32_q15_256_contiguous(src, dst, dir);
}

void dft32(int16_t *x, int16_t *y, uint8_t scale_flag)
{
  const c16_t *src = (const c16_t *)x;
  c16_t *dst = (c16_t *)y;

  (void)scale_flag;

  dft32_q15_256_contiguous(src, dst, DFT_DIR_FORWARD);
}

static inline void dft32_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir)
{
  __m128i H_lo[4] __attribute__((aligned(16)));
  __m128i H_hi[4] __attribute__((aligned(16)));

  const __m128i x0_lo = load4_complex_strided_c16(src, stride, 0);
  const __m128i x0_hi = load4_complex_strided_c16(src, stride, 4);

  const __m128i x1_lo = load4_complex_strided_c16(src, stride, 8);
  const __m128i x1_hi = load4_complex_strided_c16(src, stride, 12);

  const __m128i x2_lo = load4_complex_strided_c16(src, stride, 16);
  const __m128i x2_hi = load4_complex_strided_c16(src, stride, 20);

  const __m128i x3_lo = load4_complex_strided_c16(src, stride, 24);
  const __m128i x3_hi = load4_complex_strided_c16(src, stride, 28);

  dft4x4_q15_128(x0_lo, x1_lo, x2_lo, x3_lo, &H_lo[0], &H_lo[1], &H_lo[2], &H_lo[3], dir);

  dft4x4_q15_128(x0_hi, x1_hi, x2_hi, x3_hi, &H_hi[0], &H_hi[1], &H_hi[2], &H_hi[3], dir);

  combine32_q15_128(H_lo, H_hi, dst, dir);
}

/*
 * W24^k, k = 0..3, scaled by 1/sqrt(3)
 */
static const int16_t W24_R3_W1_RE_RE_LO[8] __attribute__((aligned(16))) = {
    18919,
    18919, /* cos(0)      / sqrt3 */
    18274,
    18274, /* cos(pi/12)  / sqrt3 */
    16384,
    16384, /* cos(pi/6)   / sqrt3 */
    13377,
    13377 /* cos(pi/4)   / sqrt3 */
};

static const int16_t W24_R3_W1_IM_SIGNED_LO[8] __attribute__((aligned(16))) = {0, 0, 4896, -4896, 9459, -9459, 13377, -13377};

/*
 * W24^k, k = 4..7, scaled by 1/sqrt(3)
 */
static const int16_t W24_R3_W1_RE_RE_HI[8] __attribute__((aligned(16))) = {
    9459,
    9459, /* cos(pi/3)    / sqrt3 */
    4896,
    4896, /* cos(5pi/12)  / sqrt3 */
    0,
    0, /* cos(pi/2)    / sqrt3 */
    -4896,
    -4896 /* cos(7pi/12)  / sqrt3 */
};

static const int16_t W24_R3_W1_IM_SIGNED_HI[8]
    __attribute__((aligned(16))) = {16384, -16384, 18274, -18274, 18919, -18919, 18274, -18274};

/*
 * W24^(2k), k = 0..3, scaled by 1/sqrt(3)
 */
static const int16_t W24_R3_W2_RE_RE_LO[8] __attribute__((aligned(16))) = {
    18919,
    18919, /* k=0 : W24^0 */
    16384,
    16384, /* k=1 : W24^2  */
    9459,
    9459, /* k=2 : W24^4  */
    0,
    0 /* k=3 : W24^6  */
};

static const int16_t W24_R3_W2_IM_SIGNED_LO[8] __attribute__((aligned(16))) = {0, 0, 9459, -9459, 16384, -16384, 18919, -18919};

/*
 * W24^(2k), k = 4..7, scaled by 1/sqrt(3)
 */
static const int16_t W24_R3_W2_RE_RE_HI[8] __attribute__((aligned(16))) = {
    -9459,
    -9459, /* k=4 : W24^8  */
    -16384,
    -16384, /* k=5 : W24^10 */
    -18919,
    -18919, /* k=6 : W24^12 */
    -16384,
    -16384 /* k=7 : W24^14 */
};

static const int16_t W24_R3_W2_IM_SIGNED_HI[8] __attribute__((aligned(16))) = {16384, -16384, 9459, -9459, 0, 0, -9459, 9459};

static inline void radix3_combine4_q15_128_scaled(__m128i A,
                                                  __m128i X1,
                                                  __m128i X2,
                                                  __m128i w1_re,
                                                  __m128i w1_im,
                                                  __m128i w2_re,
                                                  __m128i w2_im,
                                                  __m128i *Y0,
                                                  __m128i *Y1,
                                                  __m128i *Y2,
                                                  dft_dir_t dir)
{
  const __m128i As = q15_mul_i16_128(A, Q15_INV_SQRT3);

  /*
   * B[k] = W24^k    * X1[k] / sqrt(3)
   * C[k] = W24^(2k) * X2[k] / sqrt(3)
   */
  const __m128i B = complex_mul4_prepack_q15_128(X1, w1_re, w1_im);

  const __m128i C = complex_mul4_prepack_q15_128(X2, w2_re, w2_im);

  const __m128i S = _mm_adds_epi16(B, C);
  const __m128i D = _mm_subs_epi16(B, C);

  /*
   * Y0 = A + B + C
   */
  *Y0 = _mm_adds_epi16(As, S);

  /*
   * base = A - 1/2 * (B + C)
   */
  const __m128i halfS = q15_mul_i16_128(S, Q15_HALF);

  const __m128i base = _mm_subs_epi16(As, halfS);

  /*
   * c3D = sqrt(3)/2 * (B - C)
   */
  const __m128i c3D = q15_mul_i16_128(D, Q15_SQRT3_OVER_2);

  /*
   * Forward radix-3:
   *
   * Y1 = base - j*c3D
   * Y2 = base + j*c3D
   */
  *Y1 = _mm_adds_epi16(base, mul_minus_j_dir_i16_128(c3D, dir));
  *Y2 = _mm_adds_epi16(base, mul_plus_j_dir_i16_128(c3D, dir));
}

static inline void dft24_q15_128(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  /*
   * Radix-3 split.
   *
   * Lane 0 : src[0], src[3], src[6],  ..., src[21]
   * Lane 1 : src[1], src[4], src[7],  ..., src[22]
   * Lane 2 : src[2], src[5], src[8],  ..., src[23]
   * Lane 3 : dummy
   */

  const __m128i x0 = pack3_complex_plus_zero_c16(src[0], src[1], src[2]);
  const __m128i x1 = pack3_complex_plus_zero_c16(src[3], src[4], src[5]);
  const __m128i x2 = pack3_complex_plus_zero_c16(src[6], src[7], src[8]);
  const __m128i x3 = pack3_complex_plus_zero_c16(src[9], src[10], src[11]);

  const __m128i x4 = pack3_complex_plus_zero_c16(src[12], src[13], src[14]);
  const __m128i x5 = pack3_complex_plus_zero_c16(src[15], src[16], src[17]);
  const __m128i x6 = pack3_complex_plus_zero_c16(src[18], src[19], src[20]);
  const __m128i x7 = pack3_complex_plus_zero_c16(src[21], src[22], src[23]);

  __m128i H0, H1, H2, H3;
  __m128i H4, H5, H6, H7;

  /*
   * H0 = [F0[0], F1[0], F2[0], dummy]
   * H1 = [F0[1], F1[1], F2[1], dummy]
   * ...
   * H7 = [F0[7], F1[7], F2[7], dummy]

   */
  dft8x4_q15_128(x0, x1, x2, x3, x4, x5, x6, x7, &H0, &H1, &H2, &H3, &H4, &H5, &H6, &H7, dir);

  /*
   * Transpose k=0..3 :
   *
   * A_lo  = [F0[0], F0[1], F0[2], F0[3]]
   * X1_lo = [F1[0], F1[1], F1[2], F1[3]]
   * X2_lo = [F2[0], F2[1], F2[2], F2[3]]
   */
  transpose4_complex_i16_128(&H0, &H1, &H2, &H3);

  const __m128i A_lo = H0;
  const __m128i X1_lo = H1;
  const __m128i X2_lo = H2;

  /*
   * Transpose k=4..7 :
   *
   * A_hi  = [F0[4], F0[5], F0[6], F0[7]]
   * X1_hi = [F1[4], F1[5], F1[6], F1[7]]
   * X2_hi = [F2[4], F2[5], F2[6], F2[7]]
   */
  transpose4_complex_i16_128(&H4, &H5, &H6, &H7);

  const __m128i A_hi = H4;
  const __m128i X1_hi = H5;
  const __m128i X2_hi = H6;

  const __m128i W1_RE_LO = _mm_load_si128((const __m128i *)W24_R3_W1_RE_RE_LO);
  const __m128i W1_IM_LO = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W24_R3_W1_IM_SIGNED_LO), dir);

  const __m128i W2_RE_LO = _mm_load_si128((const __m128i *)W24_R3_W2_RE_RE_LO);
  const __m128i W2_IM_LO = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W24_R3_W2_IM_SIGNED_LO), dir);

  const __m128i W1_RE_HI = _mm_load_si128((const __m128i *)W24_R3_W1_RE_RE_HI);
  const __m128i W1_IM_HI = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W24_R3_W1_IM_SIGNED_HI), dir);

  const __m128i W2_RE_HI = _mm_load_si128((const __m128i *)W24_R3_W2_RE_RE_HI);
  const __m128i W2_IM_HI = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W24_R3_W2_IM_SIGNED_HI), dir);

  __m128i Y0_lo, Y1_lo, Y2_lo;
  __m128i Y0_hi, Y1_hi, Y2_hi;

  radix3_combine4_q15_128_scaled(A_lo, X1_lo, X2_lo, W1_RE_LO, W1_IM_LO, W2_RE_LO, W2_IM_LO, &Y0_lo, &Y1_lo, &Y2_lo, dir);

  radix3_combine4_q15_128_scaled(A_hi, X1_hi, X2_hi, W1_RE_HI, W1_IM_HI, W2_RE_HI, W2_IM_HI, &Y0_hi, &Y1_hi, &Y2_hi, dir);

  /*
   * size = 8
   *
   * dst[0..7]    = Y0
   * dst[8..15]   = Y1
   * dst[16..23]  = Y2
   */
  _mm_storeu_si128((__m128i *)(dst + 0), Y0_lo);
  _mm_storeu_si128((__m128i *)(dst + 4), Y0_hi);

  _mm_storeu_si128((__m128i *)(dst + 8), Y1_lo);
  _mm_storeu_si128((__m128i *)(dst + 12), Y1_hi);

  _mm_storeu_si128((__m128i *)(dst + 16), Y2_lo);
  _mm_storeu_si128((__m128i *)(dst + 20), Y2_hi);
}

void dft24(int16_t *x, int16_t *y, uint8_t scale_flag)
{
  (void)scale_flag;
  radix3_pow2_selected((const c16_t *)x, (c16_t *)y, 24, DFT_DIR_FORWARD);
}

static inline void dft24_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir)
{
  c16_t tmp[24] __attribute__((aligned(64)));

  for (int i = 0; i < 24; i++) {
    tmp[i] = src[i * stride];
  }

  dft24_q15_128(tmp, dst, dir);
}

/*
 * W20^(1*k), k=0..3, scaled by 1/sqrt(5)
 */
static const int16_t W20_1_RE_RE[8] __attribute__((aligned(16))) = {14654, 14654, 13937, 13937, 11855, 11855, 8613, 8613};

static const int16_t W20_1_IM_SIGNED[8] __attribute__((aligned(16))) = {0, 0, 4528, -4528, 8613, -8613, 11855, -11855};

/*
 * W20^(2*k), k=0..3, scaled by 1/sqrt(5)
 */
static const int16_t W20_2_RE_RE[8] __attribute__((aligned(16))) = {14654, 14654, 11855, 11855, 4528, 4528, -4528, -4528};

static const int16_t W20_2_IM_SIGNED[8] __attribute__((aligned(16))) = {0, 0, 8613, -8613, 13937, -13937, 13937, -13937};

/*
 * W20^(3*k), k=0..3, scaled by 1/sqrt(5)
 */
static const int16_t W20_3_RE_RE[8] __attribute__((aligned(16))) = {14654, 14654, 8613, 8613, -4528, -4528, -13937, -13937};

static const int16_t W20_3_IM_SIGNED[8] __attribute__((aligned(16))) = {0, 0, 11855, -11855, 13937, -13937, 4528, -4528};

/*
 * W20^(4*k), k=0..3, scaled by 1/sqrt(5)
 */
static const int16_t W20_4_RE_RE[8] __attribute__((aligned(16))) = {14654, 14654, 4528, 4528, -11855, -11855, -11855, -11855};

static const int16_t W20_4_IM_SIGNED[8] __attribute__((aligned(16))) = {0, 0, 13937, -13937, 8613, -8613, -8613, 8613};

static inline void radix5_combine4_q15_128_dft20(__m128i A,
                                                 __m128i X1,
                                                 __m128i X2,
                                                 __m128i X3,
                                                 __m128i X4,
                                                 __m128i w1_re,
                                                 __m128i w1_im,
                                                 __m128i w2_re,
                                                 __m128i w2_im,
                                                 __m128i w3_re,
                                                 __m128i w3_im,
                                                 __m128i w4_re,
                                                 __m128i w4_im,
                                                 __m128i *Y0,
                                                 __m128i *Y1,
                                                 __m128i *Y2,
                                                 __m128i *Y3,
                                                 __m128i *Y4,
                                                 dft_dir_t dir)
{
  /*
   * Twiddles W20 are already scaled by 1/sqrt(5).
   */
  const __m128i B = complex_mul4_prepack_q15_128(X1, w1_re, w1_im);

  const __m128i C = complex_mul4_prepack_q15_128(X2, w2_re, w2_im);

  const __m128i D = complex_mul4_prepack_q15_128(X3, w3_re, w3_im);

  const __m128i E = complex_mul4_prepack_q15_128(X4, w4_re, w4_im);

  const __m128i BE = _mm_adds_epi16(B, E);
  const __m128i BEminus = _mm_subs_epi16(B, E);

  const __m128i CD = _mm_adds_epi16(C, D);
  const __m128i CDminus = _mm_subs_epi16(C, D);

  /*
   * A also needs radix-5 scaling.
   */
  const __m128i As = q15_mul_i16_128(A, Q15_INV_SQRT5);

  /*
   * Y0 = A + B + C + D + E
   */
  *Y0 = _mm_adds_epi16(As, _mm_adds_epi16(BE, CD));

  /*
   * Y1 / Y4
   */
  const __m128i base1 = _mm_adds_epi16(As, _mm_adds_epi16(q15_mul_i16_128(BE, Q15_COS_2PI_5), q15_mul_i16_128(CD, Q15_COS_4PI_5)));

  const __m128i imag1 = _mm_adds_epi16(q15_mul_i16_128(BEminus, Q15_SIN_2PI_5), q15_mul_i16_128(CDminus, Q15_SIN_4PI_5));

  *Y1 = _mm_adds_epi16(base1, mul_minus_j_dir_i16_128(imag1, dir));
  *Y4 = _mm_adds_epi16(base1, mul_plus_j_dir_i16_128(imag1, dir));

  /*
   * Y2 / Y3
   */
  const __m128i base2 = _mm_adds_epi16(As, _mm_adds_epi16(q15_mul_i16_128(BE, Q15_COS_4PI_5), q15_mul_i16_128(CD, Q15_COS_2PI_5)));

  const __m128i imag2 = _mm_subs_epi16(q15_mul_i16_128(BEminus, Q15_SIN_4PI_5), q15_mul_i16_128(CDminus, Q15_SIN_2PI_5));

  *Y2 = _mm_adds_epi16(base2, mul_minus_j_dir_i16_128(imag2, dir));
  *Y3 = _mm_adds_epi16(base2, mul_plus_j_dir_i16_128(imag2, dir));
}
static inline void dft20_q15_128(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  /*
   * Radix-5 split, size = 4.
   *
   * Branches:
   *
   * r=0 : src[0],  src[5],  src[10], src[15]
   * r=1 : src[1],  src[6],  src[11], src[16]
   * r=2 : src[2],  src[7],  src[12], src[17]
   * r=3 : src[3],  src[8],  src[13], src[18]
   * r=4 : src[4],  src[9],  src[14], src[19]
   *
   * First dft4x4 computes r=0..3 in parallel.
   * Second dft4x4 computes r=4 in lane0 only.
   */

  const __m128i x0 = load4_complex_strided_c16(src, 1, 0);

  const __m128i x1 = load4_complex_strided_c16(src, 1, 5);

  const __m128i x2 = load4_complex_strided_c16(src, 1, 10);

  const __m128i x3 = load4_complex_strided_c16(src, 1, 15);

  __m128i H0, H1, H2, H3;

  /*
   * H0 = [F0[0], F1[0], F2[0], F3[0]]
   * H1 = [F0[1], F1[1], F2[1], F3[1]]
   * H2 = [F0[2], F1[2], F2[2], F3[2]]
   * H3 = [F0[3], F1[3], F2[3], F3[3]]
   */
  dft4x4_q15_128(x0, x1, x2, x3, &H0, &H1, &H2, &H3, dir);

  /*
   * Transpose to get:
   *
   * H0 = A  = [F0[0], F0[1], F0[2], F0[3]]
   * H1 = X1 = [F1[0], F1[1], F1[2], F1[3]]
   * H2 = X2 = [F2[0], F2[1], F2[2], F2[3]]
   * H3 = X3 = [F3[0], F3[1], F3[2], F3[3]]
   */
  transpose4_complex_i16_128(&H0, &H1, &H2, &H3);

  const __m128i A = H0;
  const __m128i X1 = H1;
  const __m128i X2 = H2;
  const __m128i X3 = H3;

  /*
   * Branch r=4.
   * Only lane0 is useful.
   */
  const __m128i z0 = pack1_complex_lane0_c16(src[4]);

  const __m128i z1 = pack1_complex_lane0_c16(src[9]);

  const __m128i z2 = pack1_complex_lane0_c16(src[14]);

  const __m128i z3 = pack1_complex_lane0_c16(src[19]);

  __m128i G0, G1, G2, G3;

  dft4x4_q15_128(z0, z1, z2, z3, &G0, &G1, &G2, &G3, dir);

  /*
   * X4 = [F4[0], F4[1], F4[2], F4[3]]
   */
  const __m128i g01 = _mm_unpacklo_epi32(G0, G1);
  const __m128i g23 = _mm_unpacklo_epi32(G2, G3);
  const __m128i X4 = _mm_unpacklo_epi64(g01, g23);

  const __m128i W1_RE = _mm_load_si128((const __m128i *)W20_1_RE_RE);
  const __m128i W1_IM = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W20_1_IM_SIGNED), dir);

  const __m128i W2_RE = _mm_load_si128((const __m128i *)W20_2_RE_RE);
  const __m128i W2_IM = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W20_2_IM_SIGNED), dir);

  const __m128i W3_RE = _mm_load_si128((const __m128i *)W20_3_RE_RE);
  const __m128i W3_IM = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W20_3_IM_SIGNED), dir);

  const __m128i W4_RE = _mm_load_si128((const __m128i *)W20_4_RE_RE);
  const __m128i W4_IM = twiddle_im_dir_128(_mm_load_si128((const __m128i *)W20_4_IM_SIGNED), dir);

  __m128i Y0, Y1, Y2, Y3, Y4;

  radix5_combine4_q15_128_dft20(A,
                                X1,
                                X2,
                                X3,
                                X4,
                                W1_RE,
                                W1_IM,
                                W2_RE,
                                W2_IM,
                                W3_RE,
                                W3_IM,
                                W4_RE,
                                W4_IM,
                                &Y0,
                                &Y1,
                                &Y2,
                                &Y3,
                                &Y4,
                                dir);

  /*
   * size = 4
   *
   * dst[0..3]    = Y0
   * dst[4..7]    = Y1
   * dst[8..11]   = Y2
   * dst[12..15]  = Y3
   * dst[16..19]  = Y4
   */
  _mm_storeu_si128((__m128i *)(dst + 0), Y0);
  _mm_storeu_si128((__m128i *)(dst + 4), Y1);
  _mm_storeu_si128((__m128i *)(dst + 8), Y2);
  _mm_storeu_si128((__m128i *)(dst + 12), Y3);
  _mm_storeu_si128((__m128i *)(dst + 16), Y4);
}

void dft20(int16_t *x, int16_t *y, uint8_t scale_flag)
{
  const c16_t *src = (const c16_t *)x;
  c16_t *dst = (c16_t *)y;

  (void)scale_flag;

  dft20_q15_128(src, dst, DFT_DIR_FORWARD);
}

static inline void dft20_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir)
{
  c16_t tmp[20] __attribute__((aligned(64)));

  for (int i = 0; i < 20; i++) {
    tmp[i] = src[i * stride];
  }

  dft20_q15_128(tmp, dst, dir);
}

static inline void dft16_q15_128_from_regs(const __m128i x0,
                                           const __m128i x1,
                                           const __m128i x2,
                                           const __m128i x3,
                                           c16_t *dst,
                                           dft_dir_t dir)
{
  __m128i H[4] __attribute__((aligned(16)));

  dft4x4_q15_128(x0, x1, x2, x3, &H[0], &H[1], &H[2], &H[3], dir);

  combine16_q15_128(H, dst, dir);
}

static inline void dft16_q15_128_strided(const c16_t *src, int stride, c16_t *dst, dft_dir_t dir)
{
  const __m128i x0 = load4_complex_strided_c16(src, stride, 0);

  const __m128i x1 = load4_complex_strided_c16(src, stride, 4);

  const __m128i x2 = load4_complex_strided_c16(src, stride, 8);

  const __m128i x3 = load4_complex_strided_c16(src, stride, 12);

  dft16_q15_128_from_regs(x0, x1, x2, x3, dst, dir);
}

static pthread_mutex_t selected_q15_twiddle_mutex = PTHREAD_MUTEX_INITIALIZER;
static int selected_q15_twiddles_ready;
static __m256i selected_w16_re[2][8] __attribute__((aligned(64)));
static __m256i selected_w16_im[2][8] __attribute__((aligned(64)));
/* Optional W16 coefficients with the R16 1/sqrt(2) factor folded in.
 * Paths using this table combine normalization and W16 multiplication in one
 * Q15 operation; the separate table preserves the default R16 rounding order. */
static __m256i selected_w16_s2_re[2][8] __attribute__((aligned(64)));
static __m256i selected_w16_s2_im[2][8] __attribute__((aligned(64)));
static __m256i selected_dft64_re[2][8][8] __attribute__((aligned(64)));
static __m256i selected_dft64_im[2][8][8] __attribute__((aligned(64)));
static __m256i selected_dft256_re[2][16][2] __attribute__((aligned(64)));
static __m256i selected_dft256_im[2][16][2] __attribute__((aligned(64)));
static __m256i selected_dft512_re[2][8][8] __attribute__((aligned(64)));
static __m256i selected_dft512_im[2][8][8] __attribute__((aligned(64)));
static __m256i selected_dft1024_re[2][16][8] __attribute__((aligned(64)));
static __m256i selected_dft1024_im[2][16][8] __attribute__((aligned(64)));

static inline __m256i selected_re_vec(int16_t wr)
{
  return _mm256_set1_epi16(wr);
}

static inline __m256i selected_im_vec(int16_t wi)
{
  return _mm256_setr_epi16(-wi, wi, -wi, wi, -wi, wi, -wi, wi,
                           -wi, wi, -wi, wi, -wi, wi, -wi, wi);
}

static void selected_q15_twiddles_init(void)
{
  if (__builtin_expect(__atomic_load_n(&selected_q15_twiddles_ready, __ATOMIC_ACQUIRE), 1))
    return;

  pthread_mutex_lock(&selected_q15_twiddle_mutex);
  if (__atomic_load_n(&selected_q15_twiddles_ready, __ATOMIC_RELAXED)) {
    pthread_mutex_unlock(&selected_q15_twiddle_mutex);
    return;
  }

  const float s8 = 1.0f / sqrtf(8.0f);
  const float s2_fold = 1.0f / sqrtf(2.0f);
  for (int ds = 0; ds < 2; ds++) {
    const dft_dir_t dir = ds == 0 ? DFT_DIR_FORWARD : DFT_DIR_INVERSE;

    for (int q = 0; q < 8; q++) {
      const float a = (float)dir * 2.0f * (float)M_PI * (float)q / 16.0f;
      const int16_t wr = q15_from_float(cosf(a));
      const int16_t wi = q15_from_float(sinf(a));
      selected_w16_re[ds][q] = selected_re_vec(wr);
      selected_w16_im[ds][q] = selected_im_vec(wi);
      selected_w16_s2_re[ds][q] = selected_re_vec(q15_from_float(cosf(a) * s2_fold));
      selected_w16_s2_im[ds][q] = selected_im_vec(q15_from_float(sinf(a) * s2_fold));
    }

    for (int r = 0; r < 8; r++) {
      for (int q = 0; q < 8; q++) {
        const float a = (float)dir * 2.0f * (float)M_PI * (float)(r * q) / 64.0f;
        const int16_t wr = q15_from_float(cosf(a)) / 8;
        const int16_t wi = q15_from_float(sinf(a)) / 8;
        selected_dft64_re[ds][r][q] = selected_re_vec(wr);
        selected_dft64_im[ds][r][q] = selected_im_vec(wi);
      }
    }

    for (int r = 0; r < 16; r++) {
      for (int b = 0; b < 2; b++) {
        int16_t re[16] __attribute__((aligned(32)));
        int16_t im[16] __attribute__((aligned(32)));
        for (int lane = 0; lane < 8; lane++) {
          const int n = 8 * b + lane;
          const float a = (float)dir * 2.0f * (float)M_PI * (float)(r * n) / 256.0f;
          const int16_t wr = q15_from_float(cosf(a) * s8);
          const int16_t wi = q15_from_float(sinf(a) * s8);
          re[2 * lane] = wr;
          re[2 * lane + 1] = wr;
          im[2 * lane] = -wi;
          im[2 * lane + 1] = wi;
        }
        selected_dft256_re[ds][r][b] = _mm256_load_si256((const __m256i *)re);
        selected_dft256_im[ds][r][b] = _mm256_load_si256((const __m256i *)im);
      }
    }

    for (int r = 0; r < 8; r++) {
      for (int b = 0; b < 8; b++) {
        int16_t re[16] __attribute__((aligned(32)));
        int16_t im[16] __attribute__((aligned(32)));
        for (int lane = 0; lane < 8; lane++) {
          const int n = 8 * b + lane;
          const float a = (float)dir * 2.0f * (float)M_PI * (float)(r * n) / 512.0f;
          const int16_t wr = q15_from_float(cosf(a) * s8);
          const int16_t wi = q15_from_float(sinf(a) * s8);
          re[2 * lane] = wr;
          re[2 * lane + 1] = wr;
          im[2 * lane] = -wi;
          im[2 * lane + 1] = wi;
        }
        selected_dft512_re[ds][r][b] = _mm256_load_si256((const __m256i *)re);
        selected_dft512_im[ds][r][b] = _mm256_load_si256((const __m256i *)im);
      }
    }

    for (int r = 0; r < 16; r++) {
      for (int b = 0; b < 8; b++) {
        int16_t re[16] __attribute__((aligned(32)));
        int16_t im[16] __attribute__((aligned(32)));
        for (int lane = 0; lane < 8; lane++) {
          const int n = 8 * b + lane;
          const float a = (float)dir * 2.0f * (float)M_PI * (float)(r * n) / 1024.0f;
          const int16_t wr = q15_from_float(cosf(a) * s8);
          const int16_t wi = q15_from_float(sinf(a) * s8);
          re[2 * lane] = wr;
          re[2 * lane + 1] = wr;
          im[2 * lane] = -wi;
          im[2 * lane + 1] = wi;
        }
        selected_dft1024_re[ds][r][b] = _mm256_load_si256((const __m256i *)re);
        selected_dft1024_im[ds][r][b] = _mm256_load_si256((const __m256i *)im);
      }
    }
  }

  __atomic_store_n(&selected_q15_twiddles_ready, 1, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&selected_q15_twiddle_mutex);
}

static inline void dft64x8_selected_store(const __m256i x[64], c16_t *dst, int output_radix, int output_offset, dft_dir_t dir)
{
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  __m256i t[8][8] __attribute__((aligned(64)));

  {
    __m256i h[8];
    dft8x8_q15_256_dir(x[0], x[8], x[16], x[24], x[32], x[40], x[48], x[56],
                       &h[0], &h[1], &h[2], &h[3], &h[4], &h[5], &h[6], &h[7], dir);
    t[0][0] = _mm256_srai_epi16(h[0], 3);
    for (int r = 1; r < 8; r++)
      t[r][0] = _mm256_mulhrs_epi16(h[r], selected_dft64_re[ds][r][0]);
  }

  for (int q = 1; q < 8; q++) {
    __m256i h[8];
    dft8x8_q15_256_dir(x[q], x[q + 8], x[q + 16], x[q + 24], x[q + 32], x[q + 40], x[q + 48], x[q + 56],
                       &h[0], &h[1], &h[2], &h[3], &h[4], &h[5], &h[6], &h[7], dir);
    t[0][q] = _mm256_srai_epi16(h[0], 3);
    for (int r = 1; r < 8; r++)
      t[r][q] = complex_mul8_prepack_q15_256(h[r], selected_dft64_re[ds][r][q], selected_dft64_im[ds][r][q]);
  }

  for (int r = 0; r < 8; r++) {
    __m256i y[8];
    dft8x8_q15_256_dir(t[r][0], t[r][1], t[r][2], t[r][3], t[r][4], t[r][5], t[r][6], t[r][7],
                       &y[0], &y[1], &y[2], &y[3], &y[4], &y[5], &y[6], &y[7], dir);
    for (int q = 0; q < 8; q++) {
      const int k = 8 * q + r;
      _mm256_storeu_si256((__m256i *)(dst + output_radix * k + output_offset), y[q]);
    }
  }
}

static inline __m256i dft64x8_dc_q15_256(const __m256i x[64])
{
  __m256i sum_lo = _mm256_setzero_si256();
  __m256i sum_hi = _mm256_setzero_si256();

  for (int q = 0; q < 8; q++) {
    const __m256i s04 = _mm256_adds_epi16(x[q + 0], x[q + 32]);
    const __m256i s15 = _mm256_adds_epi16(x[q + 8], x[q + 40]);
    const __m256i s26 = _mm256_adds_epi16(x[q + 16], x[q + 48]);
    const __m256i s37 = _mm256_adds_epi16(x[q + 24], x[q + 56]);
    const __m256i s02 = _mm256_adds_epi16(s04, s26);
    const __m256i s13 = _mm256_adds_epi16(s15, s37);
    const __m256i h0 = _mm256_adds_epi16(s02, s13);

    sum_lo = _mm256_add_epi32(sum_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(h0)));
    sum_hi = _mm256_add_epi32(sum_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(h0, 1)));
  }

  const __m256i round = _mm256_set1_epi32(4);
  sum_lo = _mm256_srai_epi32(_mm256_add_epi32(sum_lo,
                                               _mm256_add_epi32(round, _mm256_srai_epi32(sum_lo, 31))),
                              3);
  sum_hi = _mm256_srai_epi32(_mm256_add_epi32(sum_hi,
                                               _mm256_add_epi32(round, _mm256_srai_epi32(sum_hi, 31))),
                              3);

  const __m128i lo = _mm_packs_epi32(_mm256_castsi256_si128(sum_lo), _mm256_extracti128_si256(sum_lo, 1));
  const __m128i hi = _mm_packs_epi32(_mm256_castsi256_si128(sum_hi), _mm256_extracti128_si256(sum_hi, 1));
  return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static __attribute__((always_inline)) inline void dft64x8_parent_store_from_branches(const c16_t *b,int M,c16_t *dst,int output_stride,int first_branch,dft_dir_t dir)
{
  __m256i leaf[64] __attribute__((aligned(64))); selected_q15_twiddles_init();
  for(int n=0;n<64;n+=8){__m256i z[8];for(int br=0;br<8;br++)z[br]=_mm256_loadu_si256((const __m256i*)(b+(first_branch+br)*M+n));transpose8_complex_i16_256(&z[0],&z[1],&z[2],&z[3],&z[4],&z[5],&z[6],&z[7]);for(int lane=0;lane<8;lane++)leaf[n+lane]=z[lane];}
  const __m256i dc=dft64x8_dc_q15_256(leaf);dft64x8_selected_store(leaf,dst,output_stride,first_branch,dir);_mm256_storeu_si256((__m256i*)(dst+first_branch),dc);
}

/* Consume eight branch-major DFT64 leaves in parallel. Each branch is loaded
 * as contiguous complex samples, then the 8x8 transpose makes one AVX2 lane
 * correspond to one branch before the lane-parallel DFT64 kernel writes the
 * final interleaved radix-24 output. */
static __attribute__((always_inline)) inline void dft1536_radix24_dft64x8_store(const c16_t*b,int M,c16_t*dst,int R,int first,dft_dir_t dir){__m256i leaf[64] __attribute__((aligned(64)));selected_q15_twiddles_init();for(int n=0;n<64;n+=8){__m256i z[8];for(int br=0;br<8;br++)z[br]=_mm256_loadu_si256((const __m256i*)(b+(first+br)*M+n));transpose8_complex_i16_256(&z[0],&z[1],&z[2],&z[3],&z[4],&z[5],&z[6],&z[7]);for(int l=0;l<8;l++)leaf[n+l]=z[l];}const __m256i dc=dft64x8_dc_q15_256(leaf);dft64x8_selected_store(leaf,dst,R,first,dir);_mm256_storeu_si256((__m256i*)(dst+first),dc);}

static __attribute__((always_inline)) inline void dft64x8_branch_major_store(const c16_t*b,int M,c16_t*dst,int R,int first,dft_dir_t dir){__m256i leaf[64] __attribute__((aligned(64)));selected_q15_twiddles_init();for(int n=0;n<64;n+=8){__m256i z[8];for(int br=0;br<8;br++)z[br]=_mm256_loadu_si256((const __m256i*)(b+(first+br)*M+n));transpose8_complex_i16_256(&z[0],&z[1],&z[2],&z[3],&z[4],&z[5],&z[6],&z[7]);for(int l=0;l<8;l++)leaf[n+l]=z[l];}const __m256i dc=dft64x8_dc_q15_256(leaf);dft64x8_selected_store(leaf,dst,R,first,dir);_mm256_storeu_si256((__m256i*)(dst+first),dc);}


static __attribute__((always_inline)) inline void dft16x8_selected(const __m256i x[16], __m256i y[16], dft_dir_t dir)
{
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);
  __m256i a[8], b[8], e[8], o[8];

  a[0] = _mm256_mulhrs_epi16(_mm256_adds_epi16(x[0], x[8]), s2);
  b[0] = _mm256_mulhrs_epi16(_mm256_subs_epi16(x[0], x[8]), s2);
  for (int q = 1; q < 8; q++) {
    a[q] = _mm256_mulhrs_epi16(_mm256_adds_epi16(x[q], x[q + 8]), s2);
    b[q] = _mm256_mulhrs_epi16(_mm256_subs_epi16(x[q], x[q + 8]), s2);
    b[q] = complex_mul8_prepack_q15_256(b[q], selected_w16_re[ds][q], selected_w16_im[ds][q]);
  }

  dft8x8_q15_256_dir(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                     &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7], dir);
  dft8x8_q15_256_dir(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                     &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], dir);

  for (int k = 0; k < 8; k++) {
    y[2 * k] = e[k];
    y[2 * k + 1] = o[k];
  }
}


/* R16 helper with 1/sqrt(2) folded into the non-trivial W16 coefficients.
 * The q=0 branch keeps the explicit Q15 multiply, while q=1..7 apply the
 * same scale together with the complex W16 multiplication. This preserves
 * the arithmetic ordering selected by call sites that use this helper. */
static __attribute__((always_inline)) inline void
dft16x8_selected_w16folded(const __m256i x[16], __m256i y[16], dft_dir_t dir)
{
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);
  __m256i a[8], b[8], e[8], o[8];

  a[0] = _mm256_mulhrs_epi16(_mm256_adds_epi16(x[0], x[8]), s2);
  b[0] = _mm256_mulhrs_epi16(_mm256_subs_epi16(x[0], x[8]), s2);
  for (int q = 1; q < 8; q++) {
    a[q] = _mm256_mulhrs_epi16(_mm256_adds_epi16(x[q], x[q + 8]), s2);
    const __m256i d = _mm256_subs_epi16(x[q], x[q + 8]);
    b[q] = complex_mul8_prepack_q15_256(d, selected_w16_s2_re[ds][q], selected_w16_s2_im[ds][q]);
  }

  dft8x8_q15_256_dir(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                     &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7], dir);
  dft8x8_q15_256_dir(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                     &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], dir);

  for (int k = 0; k < 8; k++) {
    y[2 * k] = e[k];
    y[2 * k + 1] = o[k];
  }
}

static void dft256_radix16_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  selected_q15_twiddles_init();
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);
  const __m256i s8 = _mm256_set1_epi16(Q15_INV_SQRT8);
  __m256i child[2][16] __attribute__((aligned(64)));

  for (int bidx = 0; bidx < 2; bidx++) {
    const int n = 8 * bidx;
    __m256i a[8], b[8], e[8], o[8], r[16];

    const __m256i lo0 = _mm256_loadu_si256((const __m256i *)(src + n));
    const __m256i hi0 = _mm256_loadu_si256((const __m256i *)(src + 128 + n));
    a[0] = _mm256_mulhrs_epi16(_mm256_adds_epi16(lo0, hi0), s2);
    b[0] = _mm256_mulhrs_epi16(_mm256_subs_epi16(lo0, hi0), s2);

    for (int q = 1; q < 8; q++) {
      const __m256i lo = _mm256_loadu_si256((const __m256i *)(src + q * 16 + n));
      const __m256i hi = _mm256_loadu_si256((const __m256i *)(src + (q + 8) * 16 + n));
      a[q] = _mm256_mulhrs_epi16(_mm256_adds_epi16(lo, hi), s2);
      b[q] = _mm256_mulhrs_epi16(_mm256_subs_epi16(lo, hi), s2);
      b[q] = complex_mul8_prepack_q15_256(b[q], selected_w16_re[ds][q], selected_w16_im[ds][q]);
    }

    dft8x8_q15_256_dir(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                       &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7], dir);
    dft8x8_q15_256_dir(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                       &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], dir);

    r[0] = _mm256_mulhrs_epi16(e[0], s8);
    r[1] = complex_mul8_prepack_q15_256(o[0], selected_dft256_re[ds][1][bidx], selected_dft256_im[ds][1][bidx]);
    for (int k = 1; k < 8; k++) {
      const int re = 2 * k;
      const int ro = re + 1;
      r[re] = complex_mul8_prepack_q15_256(e[k], selected_dft256_re[ds][re][bidx], selected_dft256_im[ds][re][bidx]);
      r[ro] = complex_mul8_prepack_q15_256(o[k], selected_dft256_re[ds][ro][bidx], selected_dft256_im[ds][ro][bidx]);
    }

    transpose8_complex_i16_256(&r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
    transpose8_complex_i16_256(&r[8], &r[9], &r[10], &r[11], &r[12], &r[13], &r[14], &r[15]);
    for (int lane = 0; lane < 8; lane++) {
      child[0][n + lane] = r[lane];
      child[1][n + lane] = r[8 + lane];
    }
  }

  for (int group = 0; group < 2; group++) {
    __m256i y[16];
    dft16x8_selected(child[group], y, dir);
    for (int k = 0; k < 16; k++) {
      y[k] = _mm256_mulhrs_epi16(y[k], s8);
      _mm256_storeu_si256((__m256i *)(dst + 16 * k + 8 * group), y[k]);
    }
  }
}

static void dft256_radix16_selected_w16folded(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  selected_q15_twiddles_init();
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);
  const __m256i s8 = _mm256_set1_epi16(Q15_INV_SQRT8);
  __m256i child[2][16] __attribute__((aligned(64)));

  for (int bidx = 0; bidx < 2; bidx++) {
    const int n = 8 * bidx;
    __m256i a[8], b[8], e[8], o[8], r[16];

    const __m256i lo0 = _mm256_loadu_si256((const __m256i *)(src + n));
    const __m256i hi0 = _mm256_loadu_si256((const __m256i *)(src + 128 + n));
    a[0] = _mm256_mulhrs_epi16(_mm256_adds_epi16(lo0, hi0), s2);
    b[0] = _mm256_mulhrs_epi16(_mm256_subs_epi16(lo0, hi0), s2);

    for (int q = 1; q < 8; q++) {
      const __m256i lo = _mm256_loadu_si256((const __m256i *)(src + q * 16 + n));
      const __m256i hi = _mm256_loadu_si256((const __m256i *)(src + (q + 8) * 16 + n));
      a[q] = _mm256_mulhrs_epi16(_mm256_adds_epi16(lo, hi), s2);
      b[q] = _mm256_mulhrs_epi16(_mm256_subs_epi16(lo, hi), s2);
      b[q] = complex_mul8_prepack_q15_256(b[q], selected_w16_re[ds][q], selected_w16_im[ds][q]);
    }

    dft8x8_q15_256_dir(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                       &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7], dir);
    dft8x8_q15_256_dir(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                       &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], dir);

    r[0] = _mm256_mulhrs_epi16(e[0], s8);
    r[1] = complex_mul8_prepack_q15_256(o[0], selected_dft256_re[ds][1][bidx], selected_dft256_im[ds][1][bidx]);
    for (int k = 1; k < 8; k++) {
      const int re = 2 * k;
      const int ro = re + 1;
      r[re] = complex_mul8_prepack_q15_256(e[k], selected_dft256_re[ds][re][bidx], selected_dft256_im[ds][re][bidx]);
      r[ro] = complex_mul8_prepack_q15_256(o[k], selected_dft256_re[ds][ro][bidx], selected_dft256_im[ds][ro][bidx]);
    }

    transpose8_complex_i16_256(&r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
    transpose8_complex_i16_256(&r[8], &r[9], &r[10], &r[11], &r[12], &r[13], &r[14], &r[15]);
    for (int lane = 0; lane < 8; lane++) {
      child[0][n + lane] = r[lane];
      child[1][n + lane] = r[8 + lane];
    }
  }

  for (int group = 0; group < 2; group++) {
    __m256i y[16];
    dft16x8_selected_w16folded(child[group], y, dir);
    for (int k = 0; k < 16; k++) {
      y[k] = _mm256_mulhrs_epi16(y[k], s8);
      _mm256_storeu_si256((__m256i *)(dst + 16 * k + 8 * group), y[k]);
    }
  }
}

static void dft512_radix8_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  selected_q15_twiddles_init();
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  c16_t stage[512] __attribute__((aligned(64)));
  __m256i leaf[64] __attribute__((aligned(64)));

  for (int b = 0; b < 8; b++) {
    const int n = 8 * b;
    __m256i x[8], h[8];
    for (int r = 0; r < 8; r++)
      x[r] = _mm256_loadu_si256((const __m256i *)(src + r * 64 + n));
    dft8x8_q15_256_dir(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       &h[0], &h[1], &h[2], &h[3], &h[4], &h[5], &h[6], &h[7], dir);
    for (int r = 0; r < 8; r++) {
      h[r] = complex_mul8_prepack_q15_256(h[r], selected_dft512_re[ds][r][b], selected_dft512_im[ds][r][b]);
      _mm256_store_si256((__m256i *)(stage + r * 64 + n), h[r]);
    }
  }

  for (int n = 0; n < 64; n += 8) {
    __m256i z[8];
    for (int r = 0; r < 8; r++)
      z[r] = _mm256_load_si256((const __m256i *)(stage + r * 64 + n));
    transpose8_complex_i16_256(&z[0], &z[1], &z[2], &z[3], &z[4], &z[5], &z[6], &z[7]);
    for (int lane = 0; lane < 8; lane++)
      leaf[n + lane] = z[lane];
  }

  dft64x8_selected_store(leaf, dst, 8, 0, dir);
}

static void dft1024_radix16_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  selected_q15_twiddles_init();
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);
  __m256i leaf[2][64] __attribute__((aligned(64)));

  for (int bidx = 0; bidx < 8; bidx++) {
    const int n = 8 * bidx;
    __m256i a[8], b[8], e[8], o[8];

    for (int q = 0; q < 8; q++) {
      const __m256i lo = _mm256_loadu_si256((const __m256i *)(src + q * 64 + n));
      const __m256i hi = _mm256_loadu_si256((const __m256i *)(src + (q + 8) * 64 + n));
      a[q] = _mm256_mulhrs_epi16(_mm256_adds_epi16(lo, hi), s2);
      b[q] = _mm256_mulhrs_epi16(_mm256_subs_epi16(lo, hi), s2);
      if (q)
        b[q] = complex_mul8_prepack_q15_256(b[q], selected_w16_re[ds][q], selected_w16_im[ds][q]);
    }

    dft8x8_q15_256_dir(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                       &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7], dir);
    dft8x8_q15_256_dir(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                       &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], dir);

    __m256i z0 = complex_mul8_prepack_q15_256(e[0], selected_dft1024_re[ds][0][bidx], selected_dft1024_im[ds][0][bidx]);
    __m256i z1 = complex_mul8_prepack_q15_256(o[0], selected_dft1024_re[ds][1][bidx], selected_dft1024_im[ds][1][bidx]);
    __m256i z2 = complex_mul8_prepack_q15_256(e[1], selected_dft1024_re[ds][2][bidx], selected_dft1024_im[ds][2][bidx]);
    __m256i z3 = complex_mul8_prepack_q15_256(o[1], selected_dft1024_re[ds][3][bidx], selected_dft1024_im[ds][3][bidx]);
    __m256i z4 = complex_mul8_prepack_q15_256(e[2], selected_dft1024_re[ds][4][bidx], selected_dft1024_im[ds][4][bidx]);
    __m256i z5 = complex_mul8_prepack_q15_256(o[2], selected_dft1024_re[ds][5][bidx], selected_dft1024_im[ds][5][bidx]);
    __m256i z6 = complex_mul8_prepack_q15_256(e[3], selected_dft1024_re[ds][6][bidx], selected_dft1024_im[ds][6][bidx]);
    __m256i z7 = complex_mul8_prepack_q15_256(o[3], selected_dft1024_re[ds][7][bidx], selected_dft1024_im[ds][7][bidx]);
    transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    leaf[0][n + 0] = z0;
    leaf[0][n + 1] = z1;
    leaf[0][n + 2] = z2;
    leaf[0][n + 3] = z3;
    leaf[0][n + 4] = z4;
    leaf[0][n + 5] = z5;
    leaf[0][n + 6] = z6;
    leaf[0][n + 7] = z7;

    z0 = complex_mul8_prepack_q15_256(e[4], selected_dft1024_re[ds][8][bidx], selected_dft1024_im[ds][8][bidx]);
    z1 = complex_mul8_prepack_q15_256(o[4], selected_dft1024_re[ds][9][bidx], selected_dft1024_im[ds][9][bidx]);
    z2 = complex_mul8_prepack_q15_256(e[5], selected_dft1024_re[ds][10][bidx], selected_dft1024_im[ds][10][bidx]);
    z3 = complex_mul8_prepack_q15_256(o[5], selected_dft1024_re[ds][11][bidx], selected_dft1024_im[ds][11][bidx]);
    z4 = complex_mul8_prepack_q15_256(e[6], selected_dft1024_re[ds][12][bidx], selected_dft1024_im[ds][12][bidx]);
    z5 = complex_mul8_prepack_q15_256(o[6], selected_dft1024_re[ds][13][bidx], selected_dft1024_im[ds][13][bidx]);
    z6 = complex_mul8_prepack_q15_256(e[7], selected_dft1024_re[ds][14][bidx], selected_dft1024_im[ds][14][bidx]);
    z7 = complex_mul8_prepack_q15_256(o[7], selected_dft1024_re[ds][15][bidx], selected_dft1024_im[ds][15][bidx]);
    transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    leaf[1][n + 0] = z0;
    leaf[1][n + 1] = z1;
    leaf[1][n + 2] = z2;
    leaf[1][n + 3] = z3;
    leaf[1][n + 4] = z4;
    leaf[1][n + 5] = z5;
    leaf[1][n + 6] = z6;
    leaf[1][n + 7] = z7;
  }

  dft64x8_selected_store(leaf[0], dst, 16, 0, dir);
  dft64x8_selected_store(leaf[1], dst, 16, 8, dir);
}

static int dft_power2_radix8_stage_large(const c16_t *src, c16_t *stage, int N, dft_dir_t dir)
{
  power2_twiddle_t *tw = power2_twiddle_slot(N);
  if (!tw || !power2_twiddle_ensure_q15(tw, N, 8))
    return 0;

  const int M = N / 8;
  const int blocks = tw->q15_blocks;
  const __m256i *re = dir == DFT_DIR_FORWARD ? tw->q15_re : tw->q15_re_inv;
  const __m256i *im = dir == DFT_DIR_FORWARD ? tw->q15_im : tw->q15_im_inv;

  for (int b = 0; b < blocks; b++) {
    const int n = 8 * b;
    __m256i x[8], y[8];
    for (int r = 0; r < 8; r++)
      x[r] = _mm256_loadu_si256((const __m256i *)(src + r * M + n));

    dft8x8_q15_256_dir(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       &y[0], &y[1], &y[2], &y[3], &y[4], &y[5], &y[6], &y[7], dir);

    for (int r = 0; r < 8; r++) {
      const int idx = r * blocks + b;
      y[r] = complex_mul8_prepack_q15_256(y[r], re[idx], im[idx]);
      _mm256_store_si256((__m256i *)(stage + r * M + n), y[r]);
    }
  }
  return 1;
}

static int dft_power2_radix16_stage_large(const c16_t *src, c16_t *stage, int N, dft_dir_t dir)
{
  selected_q15_twiddles_init();
  power2_twiddle_t *tw = power2_twiddle_slot(N);
  if (!tw || !power2_twiddle_ensure_q15(tw, N, 16))
    return 0;

  const int M = N / 16;
  const int blocks = tw->q15_blocks;
  const __m256i *re = dir == DFT_DIR_FORWARD ? tw->q15_re : tw->q15_re_inv;
  const __m256i *im = dir == DFT_DIR_FORWARD ? tw->q15_im : tw->q15_im_inv;

  for (int b = 0; b < blocks; b++) {
    const int n = 8 * b;
    __m256i x[16], y[16];
    for (int r = 0; r < 16; r++)
      x[r] = _mm256_loadu_si256((const __m256i *)(src + r * M + n));

    dft16x8_selected(x, y, dir);

    for (int r = 0; r < 16; r++) {
      const int idx = r * blocks + b;
      y[r] = complex_mul8_prepack_q15_256(y[r], re[idx], im[idx]);
      _mm256_store_si256((__m256i *)(stage + r * M + n), y[r]);
    }
  }
  return 1;
}


static void dft2048_radix8x16_fused_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 2048, CHILD_N = 256, M = 16 };
  c16_t outer_stage[N] __attribute__((aligned(64)));
  __m256i child_in[16][16] __attribute__((aligned(64)));

  selected_q15_twiddles_init();
  if (!dft_power2_radix8_stage_large(src, outer_stage, N, dir))
    return;

  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);
  const __m256i s8 = _mm256_set1_epi16(Q15_INV_SQRT8);

  for (int bidx = 0; bidx < 2; bidx++) {
    const int n = 8 * bidx;
    __m256i R[8][16] __attribute__((aligned(64)));

    for (int outer_r = 0; outer_r < 8; outer_r++) {
      const c16_t *child_src = outer_stage + outer_r * CHILD_N;
      __m256i a[8], b[8], e[8], o[8];

      a[0] = _mm256_mulhrs_epi16(
          _mm256_adds_epi16(_mm256_loadu_si256((const __m256i *)(child_src + n)),
                            _mm256_loadu_si256((const __m256i *)(child_src + 128 + n))),
          s2);
      b[0] = _mm256_mulhrs_epi16(
          _mm256_subs_epi16(_mm256_loadu_si256((const __m256i *)(child_src + n)),
                            _mm256_loadu_si256((const __m256i *)(child_src + 128 + n))),
          s2);

      for (int q = 1; q < 8; q++) {
        const __m256i lo = _mm256_loadu_si256((const __m256i *)(child_src + q * M + n));
        const __m256i hi = _mm256_loadu_si256((const __m256i *)(child_src + (q + 8) * M + n));
        a[q] = _mm256_mulhrs_epi16(_mm256_adds_epi16(lo, hi), s2);
        b[q] = _mm256_mulhrs_epi16(_mm256_subs_epi16(lo, hi), s2);
        b[q] = complex_mul8_prepack_q15_256(b[q], selected_w16_re[ds][q], selected_w16_im[ds][q]);
      }

      dft8x8_q15_256_dir(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                         &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7], dir);
      dft8x8_q15_256_dir(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                         &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], dir);

      for (int k = 0; k < 8; k++) {
        const int re = 2 * k;
        const int ro = re + 1;
        R[outer_r][re] = complex_mul8_prepack_q15_256(
            e[k], selected_dft256_re[ds][re][bidx], selected_dft256_im[ds][re][bidx]);
        R[outer_r][ro] = complex_mul8_prepack_q15_256(
            o[k], selected_dft256_re[ds][ro][bidx], selected_dft256_im[ds][ro][bidx]);
      }
    }

    for (int r16 = 0; r16 < 16; r16++) {
      __m256i z0 = R[0][r16], z1 = R[1][r16], z2 = R[2][r16], z3 = R[3][r16];
      __m256i z4 = R[4][r16], z5 = R[5][r16], z6 = R[6][r16], z7 = R[7][r16];
      transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
      child_in[r16][n + 0] = z0;
      child_in[r16][n + 1] = z1;
      child_in[r16][n + 2] = z2;
      child_in[r16][n + 3] = z3;
      child_in[r16][n + 4] = z4;
      child_in[r16][n + 5] = z5;
      child_in[r16][n + 6] = z6;
      child_in[r16][n + 7] = z7;
    }
  }

  for (int r16 = 0; r16 < 16; r16++) {
    __m256i y[16];
    dft16x8_selected(child_in[r16], y, dir);
    for (int k = 0; k < 16; k++) {
      y[k] = _mm256_mulhrs_epi16(y[k], s8);
      _mm256_storeu_si256((__m256i *)(dst + 128 * k + 8 * r16), y[k]);
    }
  }
}

static void dft2048_radix8x16_fused_selected_w16folded(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 2048, CHILD_N = 256, M = 16 };
  c16_t outer_stage[N] __attribute__((aligned(64)));
  __m256i child_in[16][16] __attribute__((aligned(64)));

  selected_q15_twiddles_init();
  if (!dft_power2_radix8_stage_large(src, outer_stage, N, dir))
    return;

  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);
  const __m256i s8 = _mm256_set1_epi16(Q15_INV_SQRT8);

  for (int bidx = 0; bidx < 2; bidx++) {
    const int n = 8 * bidx;
    __m256i R[8][16] __attribute__((aligned(64)));

    for (int outer_r = 0; outer_r < 8; outer_r++) {
      const c16_t *child_src = outer_stage + outer_r * CHILD_N;
      __m256i a[8], b[8], e[8], o[8];

      a[0] = _mm256_mulhrs_epi16(
          _mm256_adds_epi16(_mm256_loadu_si256((const __m256i *)(child_src + n)),
                            _mm256_loadu_si256((const __m256i *)(child_src + 128 + n))),
          s2);
      b[0] = _mm256_mulhrs_epi16(
          _mm256_subs_epi16(_mm256_loadu_si256((const __m256i *)(child_src + n)),
                            _mm256_loadu_si256((const __m256i *)(child_src + 128 + n))),
          s2);

      for (int q = 1; q < 8; q++) {
        const __m256i lo = _mm256_loadu_si256((const __m256i *)(child_src + q * M + n));
        const __m256i hi = _mm256_loadu_si256((const __m256i *)(child_src + (q + 8) * M + n));
        a[q] = _mm256_mulhrs_epi16(_mm256_adds_epi16(lo, hi), s2);
        b[q] = _mm256_mulhrs_epi16(_mm256_subs_epi16(lo, hi), s2);
        b[q] = complex_mul8_prepack_q15_256(b[q], selected_w16_re[ds][q], selected_w16_im[ds][q]);
      }

      dft8x8_q15_256_dir(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                         &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7], dir);
      dft8x8_q15_256_dir(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                         &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], dir);

      for (int k = 0; k < 8; k++) {
        const int re = 2 * k;
        const int ro = re + 1;
        R[outer_r][re] = complex_mul8_prepack_q15_256(
            e[k], selected_dft256_re[ds][re][bidx], selected_dft256_im[ds][re][bidx]);
        R[outer_r][ro] = complex_mul8_prepack_q15_256(
            o[k], selected_dft256_re[ds][ro][bidx], selected_dft256_im[ds][ro][bidx]);
      }
    }

    for (int r16 = 0; r16 < 16; r16++) {
      __m256i z0 = R[0][r16], z1 = R[1][r16], z2 = R[2][r16], z3 = R[3][r16];
      __m256i z4 = R[4][r16], z5 = R[5][r16], z6 = R[6][r16], z7 = R[7][r16];
      transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
      child_in[r16][n + 0] = z0;
      child_in[r16][n + 1] = z1;
      child_in[r16][n + 2] = z2;
      child_in[r16][n + 3] = z3;
      child_in[r16][n + 4] = z4;
      child_in[r16][n + 5] = z5;
      child_in[r16][n + 6] = z6;
      child_in[r16][n + 7] = z7;
    }
  }

  for (int r16 = 0; r16 < 16; r16++) {
    __m256i y[16];
    dft16x8_selected_w16folded(child_in[r16], y, dir);
    for (int k = 0; k < 16; k++) {
      y[k] = _mm256_mulhrs_epi16(y[k], s8);
      _mm256_storeu_si256((__m256i *)(dst + 128 * k + 8 * r16), y[k]);
    }
  }
}

static void dft4096_radix8x8_fused_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 4096, CHILD_N = 512, M = 64 };
  c16_t outer_stage[N] __attribute__((aligned(64)));
  __m256i leaf_in[8][64] __attribute__((aligned(64)));

  selected_q15_twiddles_init();
  if (!dft_power2_radix8_stage_large(src, outer_stage, N, dir))
    return;

  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;

  for (int bidx = 0; bidx < 8; bidx++) {
    const int n = 8 * bidx;
    __m256i R[8][8] __attribute__((aligned(64)));

    for (int outer_r = 0; outer_r < 8; outer_r++) {
      const c16_t *child_src = outer_stage + outer_r * CHILD_N;
      __m256i x[8], h[8];
      for (int r = 0; r < 8; r++)
        x[r] = _mm256_loadu_si256((const __m256i *)(child_src + r * M + n));

      dft8x8_q15_256_dir(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                         &h[0], &h[1], &h[2], &h[3], &h[4], &h[5], &h[6], &h[7], dir);

      for (int r = 0; r < 8; r++)
        R[outer_r][r] = complex_mul8_prepack_q15_256(
            h[r], selected_dft512_re[ds][r][bidx], selected_dft512_im[ds][r][bidx]);
    }

    for (int r = 0; r < 8; r++) {
      __m256i z0 = R[0][r], z1 = R[1][r], z2 = R[2][r], z3 = R[3][r];
      __m256i z4 = R[4][r], z5 = R[5][r], z6 = R[6][r], z7 = R[7][r];
      transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
      leaf_in[r][n + 0] = z0;
      leaf_in[r][n + 1] = z1;
      leaf_in[r][n + 2] = z2;
      leaf_in[r][n + 3] = z3;
      leaf_in[r][n + 4] = z4;
      leaf_in[r][n + 5] = z5;
      leaf_in[r][n + 6] = z6;
      leaf_in[r][n + 7] = z7;
    }
  }

  for (int r = 0; r < 8; r++)
    dft64x8_selected_store(leaf_in[r], dst, 64, 8 * r, dir);
}

static void dft8192_radix8x16_fused_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 8192, CHILD_N = 1024, M = 64 };
  c16_t outer_stage[N] __attribute__((aligned(64)));
  __m256i leaf_in[16][64] __attribute__((aligned(64)));

  selected_q15_twiddles_init();
  if (!dft_power2_radix8_stage_large(src, outer_stage, N, dir))
    return;

  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);

  for (int bidx = 0; bidx < 8; bidx++) {
    const int n = 8 * bidx;
    __m256i R[8][16] __attribute__((aligned(64)));

    for (int outer_r = 0; outer_r < 8; outer_r++) {
      const c16_t *child_src = outer_stage + outer_r * CHILD_N;
      __m256i a[8], b[8], e[8], o[8];

      for (int q = 0; q < 8; q++) {
        const __m256i lo = _mm256_loadu_si256((const __m256i *)(child_src + q * M + n));
        const __m256i hi = _mm256_loadu_si256((const __m256i *)(child_src + (q + 8) * M + n));
        a[q] = _mm256_mulhrs_epi16(_mm256_adds_epi16(lo, hi), s2);
        b[q] = _mm256_mulhrs_epi16(_mm256_subs_epi16(lo, hi), s2);
        if (q)
          b[q] = complex_mul8_prepack_q15_256(b[q], selected_w16_re[ds][q], selected_w16_im[ds][q]);
      }

      dft8x8_q15_256_dir(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                         &e[0], &e[1], &e[2], &e[3], &e[4], &e[5], &e[6], &e[7], dir);
      dft8x8_q15_256_dir(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                         &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], dir);

      for (int k = 0; k < 8; k++) {
        const int re = 2 * k;
        const int ro = re + 1;
        R[outer_r][re] = complex_mul8_prepack_q15_256(
            e[k], selected_dft1024_re[ds][re][bidx], selected_dft1024_im[ds][re][bidx]);
        R[outer_r][ro] = complex_mul8_prepack_q15_256(
            o[k], selected_dft1024_re[ds][ro][bidx], selected_dft1024_im[ds][ro][bidx]);
      }
    }

    for (int r = 0; r < 16; r++) {
      __m256i z0 = R[0][r], z1 = R[1][r], z2 = R[2][r], z3 = R[3][r];
      __m256i z4 = R[4][r], z5 = R[5][r], z6 = R[6][r], z7 = R[7][r];
      transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
      leaf_in[r][n + 0] = z0;
      leaf_in[r][n + 1] = z1;
      leaf_in[r][n + 2] = z2;
      leaf_in[r][n + 3] = z3;
      leaf_in[r][n + 4] = z4;
      leaf_in[r][n + 5] = z5;
      leaf_in[r][n + 6] = z6;
      leaf_in[r][n + 7] = z7;
    }
  }

  for (int r = 0; r < 16; r++)
    dft64x8_selected_store(leaf_in[r], dst, 128, 8 * r, dir);
}

static inline void radix8_blocked_to_natural_q15_large(const c16_t *blocked, c16_t *dst, int M)
{
  for (int q = 0; q < M; q += 8) {
    __m256i y[8];
    for (int r = 0; r < 8; r++)
      y[r] = _mm256_load_si256((const __m256i *)(blocked + r * M + q));
    transpose8_complex_i16_256(&y[0], &y[1], &y[2], &y[3], &y[4], &y[5], &y[6], &y[7]);
    for (int lane = 0; lane < 8; lane++)
      _mm256_storeu_si256((__m256i *)(dst + 8 * (q + lane)), y[lane]);
  }
}

static inline void radix16_blocked_to_natural_q15_large(const c16_t *blocked, c16_t *dst, int M)
{
  for (int q = 0; q < M; q += 8) {
    __m256i lo[8], hi[8];
    for (int r = 0; r < 8; r++) {
      lo[r] = _mm256_load_si256((const __m256i *)(blocked + r * M + q));
      hi[r] = _mm256_load_si256((const __m256i *)(blocked + (r + 8) * M + q));
    }
    transpose8_complex_i16_256(&lo[0], &lo[1], &lo[2], &lo[3], &lo[4], &lo[5], &lo[6], &lo[7]);
    transpose8_complex_i16_256(&hi[0], &hi[1], &hi[2], &hi[3], &hi[4], &hi[5], &hi[6], &hi[7]);
    for (int lane = 0; lane < 8; lane++) {
      _mm256_storeu_si256((__m256i *)(dst + 16 * (q + lane)), lo[lane]);
      _mm256_storeu_si256((__m256i *)(dst + 16 * (q + lane) + 8), hi[lane]);
    }
  }
}

static void dft16384_split_selected(const c16_t *src, c16_t *dst, c16_t *work, dft_dir_t dir)
{
  const int N = 16384;
  const int half = N >> 1;
  const int quarter = N >> 2;
  c16_t *sub_in = work;
  c16_t *sub_out = work + N;
  c16_t *E = sub_out;
  c16_t *O1 = sub_out + half;
  c16_t *O3 = sub_out + half + quarter;

  pack_split_radix_input_avx2_fused(src, sub_in, N);
  dft8192_radix8x16_fused_selected(sub_in, E, dir);
  dft4096_radix8x8_fused_selected(sub_in + half, O1, dir);
  dft4096_radix8x8_fused_selected(sub_in + half + quarter, O3, dir);

  const sr_twiddle_simd_t *table = sr_twiddle_table_get(N, dir);
  if (!table)
    return;
  sr_combine_simd(E, O1, O3, dst, N, table, dir);
}

static inline int dft_power2_mixed_large_radix(int N)
{
  return N == 16384 ? 16 : 8;
}

static size_t dft_power2_mixed_large_work_len(int N)
{
  size_t need = 0;
  while (N > 1024 && N <= DFT_C16_SR_MAX_N) {
    const int radix = dft_power2_mixed_large_radix(N);
    need += 2u * (size_t)N;
    N /= radix;
  }
  return need;
}

static void dft_power2_mixed_large_core(const c16_t *src, c16_t *dst, int N, dft_dir_t dir, c16_t *work)
{
  if (N == 256) {
    dft256_radix16_selected_w16folded(src, dst, dir);
    return;
  }
  if (N == 512) {
    dft512_radix8_selected(src, dst, dir);
    return;
  }
  if (N == 1024) {
    dft1024_radix16_selected(src, dst, dir);
    return;
  }
  if (N == 2048) {
    dft2048_radix8x16_fused_selected_w16folded(src, dst, dir);
    return;
  }
  if (N == 4096) {
    dft4096_radix8x8_fused_selected(src, dst, dir);
    return;
  }
  if (N == 8192) {
    dft8192_radix8x16_fused_selected(src, dst, dir);
    return;
  }
  if (N == 16384) {
    dft16384_split_selected(src, dst, work, dir);
    return;
  }
  if (N <= 128) {
    dft_mixed_radix_c16_scaled_strided(src, 1, dst, N, dir);
    return;
  }
  if (N > DFT_C16_SR_MAX_N) {
    dft_split_radix_pure_simd((c16_t *)src, dst, N, dir);
    return;
  }

  const int radix = dft_power2_mixed_large_radix(N);
  const int M = N / radix;
  c16_t *stage = work;
  c16_t *blocked = work + N;
  c16_t *child_work = work + 2 * N;

  const int ok = radix == 16 ? dft_power2_radix16_stage_large(src, stage, N, dir)
                             : dft_power2_radix8_stage_large(src, stage, N, dir);
  if (!ok)
    return;

  for (int r = 0; r < radix; r++)
    dft_power2_mixed_large_core(stage + r * M, blocked + r * M, M, dir, child_work);

  if (radix == 16)
    radix16_blocked_to_natural_q15_large(blocked, dst, M);
  else
    radix8_blocked_to_natural_q15_large(blocked, dst, M);
}

static void dft_power2_mixed_large_q15(const c16_t *src, c16_t *dst, int N, dft_dir_t dir)
{
  if (N > DFT_C16_SR_MAX_N) {
    dft_split_radix_pure_simd((c16_t *)src, dst, N, dir);
    return;
  }

  const size_t need = dft_power2_mixed_large_work_len(N);
  if (!need) {
    dft_power2_mixed_large_core(src, dst, N, dir, NULL);
    return;
  }

  c16_t *work = x86_dft_tls_work(need);
  if (!work)
    return;
  dft_power2_mixed_large_core(src, dst, N, dir, work);
}

static void dft_power2_mixed_large_q15_strided(const c16_t *src, int stride, c16_t *dst, int N, dft_dir_t dir)
{
  if (stride == 1) {
    dft_power2_mixed_large_q15(src, dst, N, dir);
    return;
  }

  if (N > DFT_C16_SR_MAX_N) {
    dft_split_radix_pure_simd_strided(src, stride, dst, N, dir);
    return;
  }

  const size_t need = (size_t)N + dft_power2_mixed_large_work_len(N);

  c16_t *mem = x86_dft_tls_work(need);
  if (!mem)
    return;
  for (int i = 0; i < N; i++)
    mem[i] = src[i * stride];

  if (N == 2048) {
    /* The recursive/strided DFT2048 path uses the unfused W16 scaling order;
     * the contiguous entry combines W16 multiplication and normalization. */
    dft2048_radix8x16_fused_selected(mem, dst, dir);
    return;
  }

  dft_power2_mixed_large_core(mem, dst, N, dir, mem + N);
}

static inline void pack_radix3_selected(const c16_t *src, c16_t *packed, int size)
{
  int n = 0;
  for (; n + 3 < size; n += 4) {
    const __m128 l0 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)(src + 3 * n)));
    const __m128 l1 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)(src + 3 * n + 4)));
    const __m128 l2 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)(src + 3 * n + 8)));

    const __m128 a01 = _mm_shuffle_ps(l0, l0, _MM_SHUFFLE(3, 0, 3, 0));
    const __m128 a23 = _mm_shuffle_ps(l1, l2, _MM_SHUFFLE(1, 1, 2, 2));
    const __m128 b01 = _mm_shuffle_ps(l0, l1, _MM_SHUFFLE(0, 0, 1, 1));
    const __m128 b23 = _mm_shuffle_ps(l1, l2, _MM_SHUFFLE(2, 2, 3, 3));
    const __m128 c01 = _mm_shuffle_ps(l0, l1, _MM_SHUFFLE(1, 1, 2, 2));
    const __m128 c23 = _mm_shuffle_ps(l2, l2, _MM_SHUFFLE(3, 3, 0, 0));

    _mm_storeu_si128((__m128i *)(packed + n), _mm_castps_si128(_mm_shuffle_ps(a01, a23, _MM_SHUFFLE(2, 0, 1, 0))));
    _mm_storeu_si128((__m128i *)(packed + size + n), _mm_castps_si128(_mm_shuffle_ps(b01, b23, _MM_SHUFFLE(2, 0, 2, 0))));
    _mm_storeu_si128((__m128i *)(packed + 2 * size + n), _mm_castps_si128(_mm_shuffle_ps(c01, c23, _MM_SHUFFLE(2, 0, 2, 0))));
  }
  for (; n < size; n++) {
    packed[n] = src[3 * n];
    packed[size + n] = src[3 * n + 1];
    packed[2 * size + n] = src[3 * n + 2];
  }
}

static inline void radix3_combine4_q15_128_fast(__m128i A,
                                                __m128i X1,
                                                __m128i X2,
                                                __m128i w1_re_re,
                                                __m128i w1_im_im,
                                                __m128i w2_re_re,
                                                __m128i w2_im_im,
                                                __m128i *Y0,
                                                __m128i *Y1,
                                                __m128i *Y2,
                                                dft_dir_t dir)
{
  /*
   * B = W1 * X1
   * C = W2 * X2
   */
  const __m128i Bs = complex_mul4_prepack_q15_128(X1, w1_re_re, w1_im_im);
  const __m128i Cs = complex_mul4_prepack_q15_128(X2, w2_re_re, w2_im_im);

  /*
   * Correct scaling for N = 3 * size when sub-FFTs are already scaled:
   *
   * final scale = 1 / sqrt(3)
   *
   * Y0 = (A + B + C) / sqrt(3)
   */
  const __m128i As = q15_mul_i16_128(A, Q15_INV_SQRT3);
  const __m128i S = _mm_adds_epi16(Bs, Cs);
  const __m128i D = _mm_subs_epi16(Bs, Cs);
  *Y0 = _mm_adds_epi16(As, S);

  /*
   * base = A/sqrt(3) - B/(2*sqrt(3)) - C/(2*sqrt(3))
   */
  const __m128i Sh = _mm_srai_epi16(S, 1);

  const __m128i base = _mm_subs_epi16(As, Sh);

  /*
   * Z = c3 * (B - C) / sqrt(3)
   *   = 0.5 * (B - C)
   */

  const __m128i Z = q15_mul_i16_128(D, Q15_HALF_SQRT3);

  /*
   * Y1 = base - j*Z
   * Y2 = base + j*Z
   */
  *Y1 = _mm_adds_epi16(base, (dir == DFT_DIR_FORWARD) ? mul_minus_q15_128(Z) : mul_q15_128(Z));
  *Y2 = _mm_adds_epi16(base, (dir == DFT_DIR_FORWARD) ? mul_q15_128(Z) : mul_minus_q15_128(Z));
}

/* Raw radix-3 butterfly used by the fused radix-9 DIF front end.
 * Scaling and parent twiddles are applied by the caller after the butterfly. */
static inline void radix3_butterfly4_q15_128(__m128i x0,
                                             __m128i x1,
                                             __m128i x2,
                                             __m128i *y0,
                                             __m128i *y1,
                                             __m128i *y2,
                                             dft_dir_t dir)
{
  const __m128i sum = _mm_adds_epi16(x1, x2);
  const __m128i diff = _mm_subs_epi16(x1, x2);
  const __m128i base = _mm_subs_epi16(x0, q15_mul_i16_128(sum, Q15_HALF));
  const __m128i imag = q15_mul_i16_128(diff, Q15_HALF_SQRT3);

  *y0 = _mm_adds_epi16(x0, sum);
  *y1 = _mm_adds_epi16(base, (dir == DFT_DIR_FORWARD) ? mul_minus_q15_128(imag) : mul_q15_128(imag));
  *y2 = _mm_adds_epi16(base, (dir == DFT_DIR_FORWARD) ? mul_q15_128(imag) : mul_minus_q15_128(imag));
}

/* Scatter four child-frequency bins from 9 branch-major child transforms to
 * natural output order. AVX2 has no general integer scatter, so two 4x4
 * transposes turn branches 0..7 into contiguous 128-bit stores; branch 8 is
 * handled as one scalar complex value per output row. */
static inline void radix9_scatter4_q15_128(const c16_t *y, c16_t *dst, int M, int k)
{
  __m128i a0, a1, a2, a3;
  __m128i b0, b1, b2, b3;

  a0 = _mm_load_si128((const __m128i *)(y + 0 * M + k));
  a1 = _mm_load_si128((const __m128i *)(y + 1 * M + k));
  a2 = _mm_load_si128((const __m128i *)(y + 2 * M + k));
  a3 = _mm_load_si128((const __m128i *)(y + 3 * M + k));
  b0 = _mm_load_si128((const __m128i *)(y + 4 * M + k));
  b1 = _mm_load_si128((const __m128i *)(y + 5 * M + k));
  b2 = _mm_load_si128((const __m128i *)(y + 6 * M + k));
  b3 = _mm_load_si128((const __m128i *)(y + 7 * M + k));

  transpose4_complex_i16_128(&a0, &a1, &a2, &a3);
  transpose4_complex_i16_128(&b0, &b1, &b2, &b3);

  c16_t tail[4] __attribute__((aligned(16)));
  _mm_store_si128((__m128i *)tail, _mm_load_si128((const __m128i *)(y + 8 * M + k)));

#define R9_STORE_ROW(LANE, A, B)                          \
  do {                                                     \
    c16_t *d = dst + 9 * (k + (LANE));                    \
    _mm_storeu_si128((__m128i *)(d + 0), (A));            \
    _mm_storeu_si128((__m128i *)(d + 4), (B));            \
    d[8] = tail[(LANE)];                                  \
  } while (0)

  R9_STORE_ROW(0, a0, b0);
  R9_STORE_ROW(1, a1, b1);
  R9_STORE_ROW(2, a2, b2);
  R9_STORE_ROW(3, a3, b3);

#undef R9_STORE_ROW
}

/* Scatter eight child-frequency bins from 9 branch-major child transforms to
 * natural output order. Selected AVX2 path for M divisible by eight: branches
 * 0..7 are transposed as one 8x8 complex matrix and stored contiguously, while
 * branch 8 supplies the final complex value of each output row. */
static inline void radix9_scatter8_q15_256_selected(const c16_t *y, c16_t *dst, int M, int k)
{
  __m256i row0 = _mm256_load_si256((const __m256i *)(y + 0 * M + k));
  __m256i row1 = _mm256_load_si256((const __m256i *)(y + 1 * M + k));
  __m256i row2 = _mm256_load_si256((const __m256i *)(y + 2 * M + k));
  __m256i row3 = _mm256_load_si256((const __m256i *)(y + 3 * M + k));
  __m256i row4 = _mm256_load_si256((const __m256i *)(y + 4 * M + k));
  __m256i row5 = _mm256_load_si256((const __m256i *)(y + 5 * M + k));
  __m256i row6 = _mm256_load_si256((const __m256i *)(y + 6 * M + k));
  __m256i row7 = _mm256_load_si256((const __m256i *)(y + 7 * M + k));

  transpose8_complex_i16_256(&row0, &row1, &row2, &row3, &row4, &row5, &row6, &row7);

  c16_t tail[8] __attribute__((aligned(32)));
  _mm256_store_si256((__m256i *)tail, _mm256_load_si256((const __m256i *)(y + 8 * M + k)));

#define R9_STORE_ROW8(LANE, ROW)                              \
  do {                                                        \
    c16_t *d = dst + 9 * (k + (LANE));                       \
    _mm256_storeu_si256((__m256i *)(d + 0), (ROW));           \
    d[8] = tail[(LANE)];                                     \
  } while (0)

  R9_STORE_ROW8(0, row0);
  R9_STORE_ROW8(1, row1);
  R9_STORE_ROW8(2, row2);
  R9_STORE_ROW8(3, row3);
  R9_STORE_ROW8(4, row4);
  R9_STORE_ROW8(5, row5);
  R9_STORE_ROW8(6, row6);
  R9_STORE_ROW8(7, row7);

#undef R9_STORE_ROW8
}

/* Fused R9 = R3 x R3 DIF stage. The first R3 uses the N-point radix-3
 * twiddles, the second uses the (N/3)-point radix-3 twiddles. Both twiddle
 * families already include 1/sqrt(3), giving the unitary 1/3 normalization
 * across the two fused stages. Output is branch-major for nine child DFT_Ms. */
static inline void radix9_stage_to_branch_major_q15_128_plan(const c16_t *src,
                                                               c16_t *b,
                                                               int N,
                                                               dft_dir_t dir,
                                                               const r3_twiddle_t *twB,
                                                               const r3_twiddle_t *twA)
{
  const int M = N / 9;

  const __m128i *B1r = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_re : twB->r3_q15_w1_re_inv;
  const __m128i *B1i = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_im : twB->r3_q15_w1_im_inv;
  const __m128i *B2r = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_re : twB->r3_q15_w2_re_inv;
  const __m128i *B2i = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_im : twB->r3_q15_w2_im_inv;
  const __m128i *A1r = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_re : twA->r3_q15_w1_re_inv;
  const __m128i *A1i = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_im : twA->r3_q15_w1_im_inv;
  const __m128i *A2r = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_re : twA->r3_q15_w2_re_inv;
  const __m128i *A2i = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_im : twA->r3_q15_w2_im_inv;

  for (int off = 0; off < M; off += 4) {
    __m128i s00, s01, s02, s10, s11, s12, s20, s21, s22;

#define R9_FIRST_STAGE(AIDX, O0, O1, O2)                                                        \
    do {                                                                                         \
      const int first_off = (AIDX) * M + off;                                                    \
      __m128i z0, z1, z2;                                                                        \
      radix3_butterfly4_q15_128(_mm_loadu_si128((const __m128i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm_loadu_si128((const __m128i *)(src + ((AIDX) + 3) * M + off)), \
                                 _mm_loadu_si128((const __m128i *)(src + ((AIDX) + 6) * M + off)), \
                                 &z0, &z1, &z2, dir);                                             \
      (O0) = q15_mul_i16_128(z0, Q15_INV_SQRT3);                                                 \
      (O1) = complex_mul4_prepack_q15_128(z1, B1r[first_off >> 2], B1i[first_off >> 2]);          \
      (O2) = complex_mul4_prepack_q15_128(z2, B2r[first_off >> 2], B2i[first_off >> 2]);          \
    } while (0)

    R9_FIRST_STAGE(0, s00, s01, s02);
    R9_FIRST_STAGE(1, s10, s11, s12);
    R9_FIRST_STAGE(2, s20, s21, s22);

#undef R9_FIRST_STAGE

#define R9_SECOND_STAGE(BETA, X0, X1, X2)                                                     \
    do {                                                                                      \
      __m128i z0, z1, z2;                                                                     \
      radix3_butterfly4_q15_128((X0), (X1), (X2), &z0, &z1, &z2, dir);                        \
      z0 = q15_mul_i16_128(z0, Q15_INV_SQRT3);                                                \
      z1 = complex_mul4_prepack_q15_128(z1, A1r[off >> 2], A1i[off >> 2]);                    \
      z2 = complex_mul4_prepack_q15_128(z2, A2r[off >> 2], A2i[off >> 2]);                    \
      _mm_store_si128((__m128i *)(b + ((BETA) + 0 * 3) * M + off), z0);                       \
      _mm_store_si128((__m128i *)(b + ((BETA) + 1 * 3) * M + off), z1);                       \
      _mm_store_si128((__m128i *)(b + ((BETA) + 2 * 3) * M + off), z2);                       \
    } while (0)

    R9_SECOND_STAGE(0, s00, s10, s20);
    R9_SECOND_STAGE(1, s01, s11, s21);
    R9_SECOND_STAGE(2, s02, s12, s22);

#undef R9_SECOND_STAGE
  }
}

static inline void radix3_butterfly8_q15_256(__m256i x0,
                                             __m256i x1,
                                             __m256i x2,
                                             __m256i *y0,
                                             __m256i *y1,
                                             __m256i *y2,
                                             dft_dir_t dir)
{
  const __m256i sum = _mm256_adds_epi16(x1, x2);
  const __m256i diff = _mm256_subs_epi16(x1, x2);
  const __m256i base = _mm256_subs_epi16(x0, _mm256_mulhrs_epi16(sum, _mm256_set1_epi16(Q15_HALF)));
  const __m256i imag = _mm256_mulhrs_epi16(diff, _mm256_set1_epi16(Q15_HALF_SQRT3));

  *y0 = _mm256_adds_epi16(x0, sum);
  *y1 = _mm256_adds_epi16(base, mul_minus_j_dir_i16_256(imag, dir));
  *y2 = _mm256_adds_epi16(base, mul_plus_j_dir_i16_256(imag, dir));
}

/* N=288 R9 parent. Two adjacent 128-bit pre-packed twiddle blocks are
 * contiguous and 32-byte aligned, so each AVX2 load covers eight complex
 * samples without repacking coefficients in the hot loop. */
static inline void radix9_stage_to_branch_major_q15_256_288(const c16_t *src,
                                                              c16_t *b,
                                                              dft_dir_t dir,
                                                              const r3_twiddle_t *twB,
                                                              const r3_twiddle_t *twA)
{
  enum { N = 288, M = 32 };

  const __m128i *B1r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_re : twB->r3_q15_w1_re_inv;
  const __m128i *B1i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_im : twB->r3_q15_w1_im_inv;
  const __m128i *B2r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_re : twB->r3_q15_w2_re_inv;
  const __m128i *B2i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_im : twB->r3_q15_w2_im_inv;
  const __m128i *A1r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_re : twA->r3_q15_w1_re_inv;
  const __m128i *A1i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_im : twA->r3_q15_w1_im_inv;
  const __m128i *A2r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_re : twA->r3_q15_w2_re_inv;
  const __m128i *A2i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_im : twA->r3_q15_w2_im_inv;

  for (int off = 0; off < M; off += 8) {
    __m256i s00, s01, s02, s10, s11, s12, s20, s21, s22;

#define R9_FIRST_STAGE_256(AIDX, O0, O1, O2)                                                    \
    do {                                                                                         \
      const int first_off = (AIDX) * M + off;                                                    \
      const int tidx = first_off >> 2;                                                           \
      __m256i z0, z1, z2;                                                                        \
      radix3_butterfly8_q15_256(_mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 3) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 6) * M + off)), \
                                 &z0, &z1, &z2, dir);                                             \
      (O0) = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                         \
      (O1) = complex_mul8_prepack_q15_256(z1,                                                    \
                                           _mm256_load_si256((const __m256i *)(B1r128 + tidx)),   \
                                           _mm256_load_si256((const __m256i *)(B1i128 + tidx)));  \
      (O2) = complex_mul8_prepack_q15_256(z2,                                                    \
                                           _mm256_load_si256((const __m256i *)(B2r128 + tidx)),   \
                                           _mm256_load_si256((const __m256i *)(B2i128 + tidx)));  \
    } while (0)

    R9_FIRST_STAGE_256(0, s00, s01, s02);
    R9_FIRST_STAGE_256(1, s10, s11, s12);
    R9_FIRST_STAGE_256(2, s20, s21, s22);

#undef R9_FIRST_STAGE_256

    const int aidx = off >> 2;
    const __m256i A1r = _mm256_load_si256((const __m256i *)(A1r128 + aidx));
    const __m256i A1i = _mm256_load_si256((const __m256i *)(A1i128 + aidx));
    const __m256i A2r = _mm256_load_si256((const __m256i *)(A2r128 + aidx));
    const __m256i A2i = _mm256_load_si256((const __m256i *)(A2i128 + aidx));

#define R9_SECOND_STAGE_256(BETA, X0, X1, X2)                                                   \
    do {                                                                                         \
      __m256i z0, z1, z2;                                                                        \
      radix3_butterfly8_q15_256((X0), (X1), (X2), &z0, &z1, &z2, dir);                          \
      z0 = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                           \
      z1 = complex_mul8_prepack_q15_256(z1, A1r, A1i);                                          \
      z2 = complex_mul8_prepack_q15_256(z2, A2r, A2i);                                          \
      _mm256_store_si256((__m256i *)(b + ((BETA) + 0 * 3) * M + off), z0);                      \
      _mm256_store_si256((__m256i *)(b + ((BETA) + 1 * 3) * M + off), z1);                      \
      _mm256_store_si256((__m256i *)(b + ((BETA) + 2 * 3) * M + off), z2);                      \
    } while (0)

    R9_SECOND_STAGE_256(0, s00, s10, s20);
    R9_SECOND_STAGE_256(1, s01, s11, s21);
    R9_SECOND_STAGE_256(2, s02, s12, s22);

#undef R9_SECOND_STAGE_256
  }
}

/* AVX2 R9 parent for selected sizes with M divisible by eight. */
static inline void radix9_stage_to_branch_major_q15_256_selected(const c16_t *src,
                                                                   c16_t *b,
                                                                   int N,
                                                                   dft_dir_t dir,
                                                                   const r3_twiddle_t *twB,
                                                                   const r3_twiddle_t *twA)
{
  const int M = N / 9;

  const __m128i *B1r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_re : twB->r3_q15_w1_re_inv;
  const __m128i *B1i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_im : twB->r3_q15_w1_im_inv;
  const __m128i *B2r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_re : twB->r3_q15_w2_re_inv;
  const __m128i *B2i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_im : twB->r3_q15_w2_im_inv;
  const __m128i *A1r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_re : twA->r3_q15_w1_re_inv;
  const __m128i *A1i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_im : twA->r3_q15_w1_im_inv;
  const __m128i *A2r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_re : twA->r3_q15_w2_re_inv;
  const __m128i *A2i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_im : twA->r3_q15_w2_im_inv;

  for (int off = 0; off < M; off += 8) {
    __m256i s00, s01, s02, s10, s11, s12, s20, s21, s22;

#define R9_FIRST_STAGE_SELECTED_256(AIDX, O0, O1, O2)                                             \
    do {                                                                                            \
      const int first_off = (AIDX) * M + off;                                                      \
      const int tidx = first_off >> 2;                                                             \
      __m256i z0, z1, z2;                                                                          \
      radix3_butterfly8_q15_256(_mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 3) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 6) * M + off)), \
                                 &z0, &z1, &z2, dir);                                               \
      (O0) = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                           \
      (O1) = complex_mul8_prepack_q15_256(z1,                                                      \
                                           _mm256_load_si256((const __m256i *)(B1r128 + tidx)),     \
                                           _mm256_load_si256((const __m256i *)(B1i128 + tidx)));    \
      (O2) = complex_mul8_prepack_q15_256(z2,                                                      \
                                           _mm256_load_si256((const __m256i *)(B2r128 + tidx)),     \
                                           _mm256_load_si256((const __m256i *)(B2i128 + tidx)));    \
    } while (0)

    R9_FIRST_STAGE_SELECTED_256(0, s00, s01, s02);
    R9_FIRST_STAGE_SELECTED_256(1, s10, s11, s12);
    R9_FIRST_STAGE_SELECTED_256(2, s20, s21, s22);

#undef R9_FIRST_STAGE_SELECTED_256

    const int aidx = off >> 2;
    const __m256i A1r = _mm256_load_si256((const __m256i *)(A1r128 + aidx));
    const __m256i A1i = _mm256_load_si256((const __m256i *)(A1i128 + aidx));
    const __m256i A2r = _mm256_load_si256((const __m256i *)(A2r128 + aidx));
    const __m256i A2i = _mm256_load_si256((const __m256i *)(A2i128 + aidx));

#define R9_SECOND_STAGE_SELECTED_256(BETA, X0, X1, X2)                                            \
    do {                                                                                            \
      __m256i z0, z1, z2;                                                                          \
      radix3_butterfly8_q15_256((X0), (X1), (X2), &z0, &z1, &z2, dir);                            \
      z0 = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                             \
      z1 = complex_mul8_prepack_q15_256(z1, A1r, A1i);                                            \
      z2 = complex_mul8_prepack_q15_256(z2, A2r, A2i);                                            \
      _mm256_store_si256((__m256i *)(b + ((BETA) + 0 * 3) * M + off), z0);                        \
      _mm256_store_si256((__m256i *)(b + ((BETA) + 1 * 3) * M + off), z1);                        \
      _mm256_store_si256((__m256i *)(b + ((BETA) + 2 * 3) * M + off), z2);                        \
    } while (0)

    R9_SECOND_STAGE_SELECTED_256(0, s00, s10, s20);
    R9_SECOND_STAGE_SELECTED_256(1, s01, s11, s21);
    R9_SECOND_STAGE_SELECTED_256(2, s02, s12, s22);

#undef R9_SECOND_STAGE_SELECTED_256
  }
}


/* DFT576 parent R9 output directly in the lane-major format consumed by
 * the eight-way DFT64 child kernel. Branch 8 remains contiguous for the
 * standalone DFT64 tail. */
static inline void radix9_stage_to_dft64x8_576(const c16_t *src,
                                                __m256i leaf[64],
                                                c16_t branch8[64],
                                                dft_dir_t dir,
                                                const r3_twiddle_t *twB,
                                                const r3_twiddle_t *twA)
{
  enum { M = 64 };

  const __m128i *B1r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_re : twB->r3_q15_w1_re_inv;
  const __m128i *B1i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_im : twB->r3_q15_w1_im_inv;
  const __m128i *B2r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_re : twB->r3_q15_w2_re_inv;
  const __m128i *B2i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_im : twB->r3_q15_w2_im_inv;
  const __m128i *A1r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_re : twA->r3_q15_w1_re_inv;
  const __m128i *A1i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_im : twA->r3_q15_w1_im_inv;
  const __m128i *A2r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_re : twA->r3_q15_w2_re_inv;
  const __m128i *A2i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_im : twA->r3_q15_w2_im_inv;

  for (int off = 0; off < M; off += 8) {
    __m256i s00, s01, s02, s10, s11, s12, s20, s21, s22;

#define R9_FIRST_STAGE_576(AIDX, O0, O1, O2)                                                       \
    do {                                                                                            \
      const int first_off = (AIDX) * M + off;                                                      \
      const int tidx = first_off >> 2;                                                             \
      __m256i z0, z1, z2;                                                                          \
      radix3_butterfly8_q15_256(_mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 3) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 6) * M + off)), \
                                 &z0, &z1, &z2, dir);                                               \
      (O0) = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                           \
      (O1) = complex_mul8_prepack_q15_256(z1,                                                      \
                                           _mm256_load_si256((const __m256i *)(B1r128 + tidx)),     \
                                           _mm256_load_si256((const __m256i *)(B1i128 + tidx)));    \
      (O2) = complex_mul8_prepack_q15_256(z2,                                                      \
                                           _mm256_load_si256((const __m256i *)(B2r128 + tidx)),     \
                                           _mm256_load_si256((const __m256i *)(B2i128 + tidx)));    \
    } while (0)

    R9_FIRST_STAGE_576(0, s00, s01, s02);
    R9_FIRST_STAGE_576(1, s10, s11, s12);
    R9_FIRST_STAGE_576(2, s20, s21, s22);

#undef R9_FIRST_STAGE_576

    const int aidx = off >> 2;
    const __m256i A1r = _mm256_load_si256((const __m256i *)(A1r128 + aidx));
    const __m256i A1i = _mm256_load_si256((const __m256i *)(A1i128 + aidx));
    const __m256i A2r = _mm256_load_si256((const __m256i *)(A2r128 + aidx));
    const __m256i A2i = _mm256_load_si256((const __m256i *)(A2i128 + aidx));

#define R9_SECOND_STAGE_576(X0, X1, X2, O0, O1, O2)                                               \
    do {                                                                                            \
      radix3_butterfly8_q15_256((X0), (X1), (X2), &(O0), &(O1), &(O2), dir);                      \
      (O0) = _mm256_mulhrs_epi16((O0), _mm256_set1_epi16(Q15_INV_SQRT3));                         \
      (O1) = complex_mul8_prepack_q15_256((O1), A1r, A1i);                                        \
      (O2) = complex_mul8_prepack_q15_256((O2), A2r, A2i);                                        \
    } while (0)

    __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8;
    R9_SECOND_STAGE_576(s00, s10, s20, r0, r3, r6);
    R9_SECOND_STAGE_576(s01, s11, s21, r1, r4, r7);
    R9_SECOND_STAGE_576(s02, s12, s22, r2, r5, r8);

#undef R9_SECOND_STAGE_576

    transpose8_complex_i16_256(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    leaf[off + 0] = r0;
    leaf[off + 1] = r1;
    leaf[off + 2] = r2;
    leaf[off + 3] = r3;
    leaf[off + 4] = r4;
    leaf[off + 5] = r5;
    leaf[off + 6] = r6;
    leaf[off + 7] = r7;

    _mm256_store_si256((__m256i *)(branch8 + off), r8);
  }
}

/* Dedicated N=972 outer R9 hybrid. M=108 is 13*8+4, so run the
 * lane-independent AVX2 arithmetic for offsets 0..103 and finish the last
 * four complex samples with the existing 128-bit arithmetic.  Unlike the
 * selected M%8==0 path, branch strides and some pre-packed twiddle streams
 * are only 16-byte aligned, so DFT972 uses unaligned 256-bit twiddle loads
 * and output stores. */
static inline void radix9_stage_to_branch_major_q15_256_128_hybrid_972(const c16_t *src,
                                                                        c16_t *b,
                                                                        dft_dir_t dir,
                                                                        const r3_twiddle_t *twB,
                                                                        const r3_twiddle_t *twA)
{
  enum { N = 972, M = 108, AVX2_END = 104 };

  const __m128i *B1r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_re : twB->r3_q15_w1_re_inv;
  const __m128i *B1i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_im : twB->r3_q15_w1_im_inv;
  const __m128i *B2r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_re : twB->r3_q15_w2_re_inv;
  const __m128i *B2i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_im : twB->r3_q15_w2_im_inv;
  const __m128i *A1r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_re : twA->r3_q15_w1_re_inv;
  const __m128i *A1i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_im : twA->r3_q15_w1_im_inv;
  const __m128i *A2r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_re : twA->r3_q15_w2_re_inv;
  const __m128i *A2i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_im : twA->r3_q15_w2_im_inv;

  for (int off = 0; off < AVX2_END; off += 8) {
    __m256i s00, s01, s02, s10, s11, s12, s20, s21, s22;

#define R9_972_FIRST_STAGE_256(AIDX, O0, O1, O2)                                                  \
    do {                                                                                            \
      const int first_off = (AIDX) * M + off;                                                      \
      const int tidx = first_off >> 2;                                                             \
      __m256i z0, z1, z2;                                                                          \
      radix3_butterfly8_q15_256(_mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 3) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 6) * M + off)), \
                                 &z0, &z1, &z2, dir);                                               \
      (O0) = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                           \
      (O1) = complex_mul8_prepack_q15_256(z1,                                                      \
                                           _mm256_loadu_si256((const __m256i *)(B1r128 + tidx)),    \
                                           _mm256_loadu_si256((const __m256i *)(B1i128 + tidx)));   \
      (O2) = complex_mul8_prepack_q15_256(z2,                                                      \
                                           _mm256_loadu_si256((const __m256i *)(B2r128 + tidx)),    \
                                           _mm256_loadu_si256((const __m256i *)(B2i128 + tidx)));   \
    } while (0)

    R9_972_FIRST_STAGE_256(0, s00, s01, s02);
    R9_972_FIRST_STAGE_256(1, s10, s11, s12);
    R9_972_FIRST_STAGE_256(2, s20, s21, s22);

#undef R9_972_FIRST_STAGE_256

    const int aidx = off >> 2;
    const __m256i A1r = _mm256_loadu_si256((const __m256i *)(A1r128 + aidx));
    const __m256i A1i = _mm256_loadu_si256((const __m256i *)(A1i128 + aidx));
    const __m256i A2r = _mm256_loadu_si256((const __m256i *)(A2r128 + aidx));
    const __m256i A2i = _mm256_loadu_si256((const __m256i *)(A2i128 + aidx));

#define R9_972_SECOND_STAGE_256(BETA, X0, X1, X2)                                                \
    do {                                                                                            \
      __m256i z0, z1, z2;                                                                          \
      radix3_butterfly8_q15_256((X0), (X1), (X2), &z0, &z1, &z2, dir);                            \
      z0 = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                             \
      z1 = complex_mul8_prepack_q15_256(z1, A1r, A1i);                                            \
      z2 = complex_mul8_prepack_q15_256(z2, A2r, A2i);                                            \
      _mm256_storeu_si256((__m256i *)(b + ((BETA) + 0 * 3) * M + off), z0);                       \
      _mm256_storeu_si256((__m256i *)(b + ((BETA) + 1 * 3) * M + off), z1);                       \
      _mm256_storeu_si256((__m256i *)(b + ((BETA) + 2 * 3) * M + off), z2);                       \
    } while (0)

    R9_972_SECOND_STAGE_256(0, s00, s10, s20);
    R9_972_SECOND_STAGE_256(1, s01, s11, s21);
    R9_972_SECOND_STAGE_256(2, s02, s12, s22);

#undef R9_972_SECOND_STAGE_256
  }

  /* Exact 128-bit tail corresponding to off=104 in the original parent. */
  {
    const int off = AVX2_END;
    __m128i s00, s01, s02, s10, s11, s12, s20, s21, s22;

#define R9_972_FIRST_STAGE_128(AIDX, O0, O1, O2)                                                  \
    do {                                                                                            \
      const int first_off = (AIDX) * M + off;                                                      \
      __m128i z0, z1, z2;                                                                          \
      radix3_butterfly4_q15_128(_mm_loadu_si128((const __m128i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm_loadu_si128((const __m128i *)(src + ((AIDX) + 3) * M + off)), \
                                 _mm_loadu_si128((const __m128i *)(src + ((AIDX) + 6) * M + off)), \
                                 &z0, &z1, &z2, dir);                                               \
      (O0) = q15_mul_i16_128(z0, Q15_INV_SQRT3);                                                   \
      (O1) = complex_mul4_prepack_q15_128(z1, B1r128[first_off >> 2], B1i128[first_off >> 2]);     \
      (O2) = complex_mul4_prepack_q15_128(z2, B2r128[first_off >> 2], B2i128[first_off >> 2]);     \
    } while (0)

    R9_972_FIRST_STAGE_128(0, s00, s01, s02);
    R9_972_FIRST_STAGE_128(1, s10, s11, s12);
    R9_972_FIRST_STAGE_128(2, s20, s21, s22);

#undef R9_972_FIRST_STAGE_128

#define R9_972_SECOND_STAGE_128(BETA, X0, X1, X2)                                                 \
    do {                                                                                            \
      __m128i z0, z1, z2;                                                                          \
      radix3_butterfly4_q15_128((X0), (X1), (X2), &z0, &z1, &z2, dir);                            \
      z0 = q15_mul_i16_128(z0, Q15_INV_SQRT3);                                                     \
      z1 = complex_mul4_prepack_q15_128(z1, A1r128[off >> 2], A1i128[off >> 2]);                   \
      z2 = complex_mul4_prepack_q15_128(z2, A2r128[off >> 2], A2i128[off >> 2]);                   \
      _mm_store_si128((__m128i *)(b + ((BETA) + 0 * 3) * M + off), z0);                           \
      _mm_store_si128((__m128i *)(b + ((BETA) + 1 * 3) * M + off), z1);                           \
      _mm_store_si128((__m128i *)(b + ((BETA) + 2 * 3) * M + off), z2);                           \
    } while (0)

    R9_972_SECOND_STAGE_128(0, s00, s10, s20);
    R9_972_SECOND_STAGE_128(1, s01, s11, s21);
    R9_972_SECOND_STAGE_128(2, s02, s12, s22);

#undef R9_972_SECOND_STAGE_128
  }
}

/* Dedicated N=108 inner R9 hybrid for DFT972. M=12 is 1*8+4:
 * process offsets 0..7 with the same AVX2 radix-9 arithmetic and finish
 * offsets 8..11 with the existing 128-bit arithmetic.  N=108 twiddle
 * streams and the DFT972 inner scratch are only guaranteed 16-byte
 * alignment here, so use unaligned 256-bit coefficient loads and stores. */
static inline void radix9_stage_to_branch_major_q15_256_128_hybrid_108(const c16_t *src,
                                                                        c16_t *b,
                                                                        dft_dir_t dir,
                                                                        const r3_twiddle_t *twB,
                                                                        const r3_twiddle_t *twA)
{
  enum { N = 108, M = 12, AVX2_END = 8 };

  const __m128i *B1r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_re : twB->r3_q15_w1_re_inv;
  const __m128i *B1i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_im : twB->r3_q15_w1_im_inv;
  const __m128i *B2r128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_re : twB->r3_q15_w2_re_inv;
  const __m128i *B2i128 = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_im : twB->r3_q15_w2_im_inv;
  const __m128i *A1r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_re : twA->r3_q15_w1_re_inv;
  const __m128i *A1i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w1_im : twA->r3_q15_w1_im_inv;
  const __m128i *A2r128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_re : twA->r3_q15_w2_re_inv;
  const __m128i *A2i128 = dir == DFT_DIR_FORWARD ? twA->r3_q15_w2_im : twA->r3_q15_w2_im_inv;

  {
    const int off = 0;
    __m256i s00, s01, s02, s10, s11, s12, s20, s21, s22;

#define R9_108_FIRST_STAGE_256(AIDX, O0, O1, O2)                                                  \
    do {                                                                                            \
      const int first_off = (AIDX) * M + off;                                                      \
      const int tidx = first_off >> 2;                                                             \
      __m256i z0, z1, z2;                                                                          \
      radix3_butterfly8_q15_256(_mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 3) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 6) * M + off)), \
                                 &z0, &z1, &z2, dir);                                               \
      (O0) = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                           \
      (O1) = complex_mul8_prepack_q15_256(z1,                                                      \
                                           _mm256_loadu_si256((const __m256i *)(B1r128 + tidx)),    \
                                           _mm256_loadu_si256((const __m256i *)(B1i128 + tidx)));   \
      (O2) = complex_mul8_prepack_q15_256(z2,                                                      \
                                           _mm256_loadu_si256((const __m256i *)(B2r128 + tidx)),    \
                                           _mm256_loadu_si256((const __m256i *)(B2i128 + tidx)));   \
    } while (0)

    R9_108_FIRST_STAGE_256(0, s00, s01, s02);
    R9_108_FIRST_STAGE_256(1, s10, s11, s12);
    R9_108_FIRST_STAGE_256(2, s20, s21, s22);

#undef R9_108_FIRST_STAGE_256

    const int aidx = off >> 2;
    const __m256i A1r = _mm256_loadu_si256((const __m256i *)(A1r128 + aidx));
    const __m256i A1i = _mm256_loadu_si256((const __m256i *)(A1i128 + aidx));
    const __m256i A2r = _mm256_loadu_si256((const __m256i *)(A2r128 + aidx));
    const __m256i A2i = _mm256_loadu_si256((const __m256i *)(A2i128 + aidx));

#define R9_108_SECOND_STAGE_256(BETA, X0, X1, X2)                                                \
    do {                                                                                            \
      __m256i z0, z1, z2;                                                                          \
      radix3_butterfly8_q15_256((X0), (X1), (X2), &z0, &z1, &z2, dir);                            \
      z0 = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));                             \
      z1 = complex_mul8_prepack_q15_256(z1, A1r, A1i);                                            \
      z2 = complex_mul8_prepack_q15_256(z2, A2r, A2i);                                            \
      _mm256_storeu_si256((__m256i *)(b + ((BETA) + 0 * 3) * M + off), z0);                       \
      _mm256_storeu_si256((__m256i *)(b + ((BETA) + 1 * 3) * M + off), z1);                       \
      _mm256_storeu_si256((__m256i *)(b + ((BETA) + 2 * 3) * M + off), z2);                       \
    } while (0)

    R9_108_SECOND_STAGE_256(0, s00, s10, s20);
    R9_108_SECOND_STAGE_256(1, s01, s11, s21);
    R9_108_SECOND_STAGE_256(2, s02, s12, s22);

#undef R9_108_SECOND_STAGE_256
  }

  /* Exact 128-bit tail corresponding to off=8 in the original N=108 parent. */
  {
    const int off = AVX2_END;
    __m128i s00, s01, s02, s10, s11, s12, s20, s21, s22;

#define R9_108_FIRST_STAGE_128(AIDX, O0, O1, O2)                                                  \
    do {                                                                                            \
      const int first_off = (AIDX) * M + off;                                                      \
      __m128i z0, z1, z2;                                                                          \
      radix3_butterfly4_q15_128(_mm_loadu_si128((const __m128i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm_loadu_si128((const __m128i *)(src + ((AIDX) + 3) * M + off)), \
                                 _mm_loadu_si128((const __m128i *)(src + ((AIDX) + 6) * M + off)), \
                                 &z0, &z1, &z2, dir);                                               \
      (O0) = q15_mul_i16_128(z0, Q15_INV_SQRT3);                                                   \
      (O1) = complex_mul4_prepack_q15_128(z1, B1r128[first_off >> 2], B1i128[first_off >> 2]);     \
      (O2) = complex_mul4_prepack_q15_128(z2, B2r128[first_off >> 2], B2i128[first_off >> 2]);     \
    } while (0)

    R9_108_FIRST_STAGE_128(0, s00, s01, s02);
    R9_108_FIRST_STAGE_128(1, s10, s11, s12);
    R9_108_FIRST_STAGE_128(2, s20, s21, s22);

#undef R9_108_FIRST_STAGE_128

#define R9_108_SECOND_STAGE_128(BETA, X0, X1, X2)                                                 \
    do {                                                                                            \
      __m128i z0, z1, z2;                                                                          \
      radix3_butterfly4_q15_128((X0), (X1), (X2), &z0, &z1, &z2, dir);                            \
      z0 = q15_mul_i16_128(z0, Q15_INV_SQRT3);                                                     \
      z1 = complex_mul4_prepack_q15_128(z1, A1r128[off >> 2], A1i128[off >> 2]);                   \
      z2 = complex_mul4_prepack_q15_128(z2, A2r128[off >> 2], A2i128[off >> 2]);                   \
      _mm_store_si128((__m128i *)(b + ((BETA) + 0 * 3) * M + off), z0);                           \
      _mm_store_si128((__m128i *)(b + ((BETA) + 1 * 3) * M + off), z1);                           \
      _mm_store_si128((__m128i *)(b + ((BETA) + 2 * 3) * M + off), z2);                           \
    } while (0)

    R9_108_SECOND_STAGE_128(0, s00, s10, s20);
    R9_108_SECOND_STAGE_128(1, s01, s11, s21);
    R9_108_SECOND_STAGE_128(2, s02, s12, s22);

#undef R9_108_SECOND_STAGE_128
  }
}


static inline void dft4x4_q15_256x2(const __m256i x0,
                                     const __m256i x1,
                                     const __m256i x2,
                                     const __m256i x3,
                                     __m256i *Y0,
                                     __m256i *Y1,
                                     __m256i *Y2,
                                     __m256i *Y3,
                                     dft_dir_t dir)
{
  const __m256i x0s = _mm256_srai_epi16(x0, 1);
  const __m256i x1s = _mm256_srai_epi16(x1, 1);
  const __m256i x2s = _mm256_srai_epi16(x2, 1);
  const __m256i x3s = _mm256_srai_epi16(x3, 1);

  const __m256i s02 = _mm256_adds_epi16(x0s, x2s);
  const __m256i d02 = _mm256_subs_epi16(x0s, x2s);
  const __m256i s13 = _mm256_adds_epi16(x1s, x3s);
  const __m256i d13 = _mm256_subs_epi16(x1s, x3s);

  *Y0 = _mm256_adds_epi16(s02, s13);
  *Y2 = _mm256_subs_epi16(s02, s13);
  *Y1 = _mm256_adds_epi16(d02, mul_minus_j_dir_i16_256(d13, dir));
  *Y3 = _mm256_adds_epi16(d02, mul_plus_j_dir_i16_256(d13, dir));
}

static inline void dft32_q15_256_contiguous(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  const __m256i x0 = _mm256_loadu_si256((const __m256i *)(src + 0));
  const __m256i x1 = _mm256_loadu_si256((const __m256i *)(src + 8));
  const __m256i x2 = _mm256_loadu_si256((const __m256i *)(src + 16));
  const __m256i x3 = _mm256_loadu_si256((const __m256i *)(src + 24));

  __m256i H0, H1, H2, H3;
  dft4x4_q15_256x2(x0, x1, x2, x3, &H0, &H1, &H2, &H3, dir);

  const __m256i W1_RE = _mm256_load_si256((const __m256i *)W32_1_RE_RE_256);
  const __m256i W1_IM = twiddle_im_dir_256(_mm256_load_si256((const __m256i *)W32_1_IM_SIGNED_256), dir);
  const __m256i W2_RE = _mm256_load_si256((const __m256i *)W32_2_RE_RE_256);
  const __m256i W2_IM = twiddle_im_dir_256(_mm256_load_si256((const __m256i *)W32_2_IM_SIGNED_256), dir);
  const __m256i W3_RE = _mm256_load_si256((const __m256i *)W32_3_RE_RE_256);
  const __m256i W3_IM = twiddle_im_dir_256(_mm256_load_si256((const __m256i *)W32_3_IM_SIGNED_256), dir);

  H1 = complex_mul8_prepack_q15_256(H1, W1_RE, W1_IM);
  H2 = complex_mul8_prepack_q15_256(H2, W2_RE, W2_IM);
  H3 = complex_mul8_prepack_q15_256(H3, W3_RE, W3_IM);

  transpose4_complex_i16_256x2_shuffle(&H0, &H1, &H2, &H3);

  const __m128i lo0 = _mm256_castsi256_si128(H0);
  const __m128i lo1 = _mm256_castsi256_si128(H1);
  const __m128i lo2 = _mm256_castsi256_si128(H2);
  const __m128i lo3 = _mm256_castsi256_si128(H3);
  const __m128i hi0 = _mm256_extracti128_si256(H0, 1);
  const __m128i hi1 = _mm256_extracti128_si256(H1, 1);
  const __m128i hi2 = _mm256_extracti128_si256(H2, 1);
  const __m128i hi3 = _mm256_extracti128_si256(H3, 1);

  __m128i Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7;
  dft8x4_q15_128(lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3,
                  &Y0, &Y1, &Y2, &Y3, &Y4, &Y5, &Y6, &Y7, dir);

  _mm_storeu_si128((__m128i *)(dst + 0), Y0);
  _mm_storeu_si128((__m128i *)(dst + 4), Y1);
  _mm_storeu_si128((__m128i *)(dst + 8), Y2);
  _mm_storeu_si128((__m128i *)(dst + 12), Y3);
  _mm_storeu_si128((__m128i *)(dst + 16), Y4);
  _mm_storeu_si128((__m128i *)(dst + 20), Y5);
  _mm_storeu_si128((__m128i *)(dst + 24), Y6);
  _mm_storeu_si128((__m128i *)(dst + 28), Y7);
}


static inline void radix9_dft8x4_parent_store(const c16_t *src, c16_t *dst, int M, int parent_branch, dft_dir_t dir)
{
  __m128i x0 = _mm_load_si128((const __m128i *)(src + 0 * M + 0));
  __m128i x1 = _mm_load_si128((const __m128i *)(src + 1 * M + 0));
  __m128i x2 = _mm_load_si128((const __m128i *)(src + 2 * M + 0));
  __m128i x3 = _mm_load_si128((const __m128i *)(src + 3 * M + 0));
  transpose4_complex_i16_128(&x0, &x1, &x2, &x3);

  __m128i x4 = _mm_load_si128((const __m128i *)(src + 0 * M + 4));
  __m128i x5 = _mm_load_si128((const __m128i *)(src + 1 * M + 4));
  __m128i x6 = _mm_load_si128((const __m128i *)(src + 2 * M + 4));
  __m128i x7 = _mm_load_si128((const __m128i *)(src + 3 * M + 4));
  transpose4_complex_i16_128(&x4, &x5, &x6, &x7);

  __m128i y0, y1, y2, y3, y4, y5, y6, y7;
  dft8x4_q15_128(x0, x1, x2, x3, x4, x5, x6, x7, &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7, dir);

  _mm_storeu_si128((__m128i *)(dst + 9 * 0 + parent_branch), y0);
  _mm_storeu_si128((__m128i *)(dst + 9 * 1 + parent_branch), y1);
  _mm_storeu_si128((__m128i *)(dst + 9 * 2 + parent_branch), y2);
  _mm_storeu_si128((__m128i *)(dst + 9 * 3 + parent_branch), y3);
  _mm_storeu_si128((__m128i *)(dst + 9 * 4 + parent_branch), y4);
  _mm_storeu_si128((__m128i *)(dst + 9 * 5 + parent_branch), y5);
  _mm_storeu_si128((__m128i *)(dst + 9 * 6 + parent_branch), y6);
  _mm_storeu_si128((__m128i *)(dst + 9 * 7 + parent_branch), y7);
}


/* Four DFT32 branches are packed into the final interleaved R9 layout.
 * Two adjacent four-output blocks are transposed in parallel in the two
 * 128-bit lanes of one YMM register. */
static inline __m256i complex_mul8_bcast_q15_256_parent(__m256i a, int16_t wr, int16_t wi)
{
  const __m256i w_re_re = _mm256_set1_epi16(wr);
  const __m256i w_im_signed = _mm256_setr_epi16(-wi,
                                                 wi,
                                                 -wi,
                                                 wi,
                                                 -wi,
                                                 wi,
                                                 -wi,
                                                 wi,
                                                 -wi,
                                                 wi,
                                                 -wi,
                                                 wi,
                                                 -wi,
                                                 wi,
                                                 -wi,
                                                 wi);
  return complex_mul8_prepack_q15_256(a, w_re_re, w_im_signed);
}


/* -------------------------------------------------------------------------
 * AVX2 batched twiddle-scaled leaf kernels.
 *
 * Leaf normalization is carried by the internal Cooley-Tukey twiddles:
 *
 *   DFT8  = raw DFT2 -> W8  / sqrt(8)  -> raw DFT4
 *   DFT16 = raw DFT4 -> W16 / sqrt(16) -> raw DFT4
 *   DFT32 = raw DFT4 -> W32 / sqrt(32) -> raw DFT8
 *
 * Eight independent leaves are carried in the eight 32-bit complex lanes of
 * one YMM register. Partial final groups use AVX2 mask stores, not XMM tails.
 * ------------------------------------------------------------------------- */

static const int16_t dft8_leaf_twiddle_re[8] = {11585, 8192, 0, -8192, -11585, -8192, 0, 8192};
static const int16_t dft8_leaf_twiddle_im_fwd[8] = {0, -8192, -11585, -8192, 0, 8192, 11585, 8192};
static const int16_t dft16_leaf_twiddle_re[16] = {8192, 7568, 5792, 3135, 0, -3135, -5792, -7568, -8192, -7568, -5792, -3135, 0, 3135, 5792, 7568};
static const int16_t dft16_leaf_twiddle_im_fwd[16] = {0, -3135, -5792, -7568, -8192, -7568, -5792, -3135, 0, 3135, 5792, 7568, 8192, 7568, 5792, 3135};
static const int16_t dft32_leaf_twiddle_re[32] = {5792, 5681, 5352, 4816, 4096, 3218, 2217, 1130, 0, -1130, -2217, -3218, -4096, -4816, -5352, -5681, -5792, -5681, -5352, -4816, -4096, -3218, -2217, -1130, 0, 1130, 2217, 3218, 4096, 4816, 5352, 5681};
static const int16_t dft32_leaf_twiddle_im_fwd[32] = {0, -1130, -2217, -3218, -4096, -4816, -5352, -5681, -5792, -5681, -5352, -4816, -4096, -3218, -2217, -1130, 0, 1130, 2217, 3218, 4096, 4816, 5352, 5681, 5792, 5681, 5352, 4816, 4096, 3218, 2217, 1130};

static __attribute__((always_inline)) inline __m256i
complex_lane_mask_epi32(int valid)
{
  return _mm256_setr_epi32(valid > 0 ? -1 : 0,
                           valid > 1 ? -1 : 0,
                           valid > 2 ? -1 : 0,
                           valid > 3 ? -1 : 0,
                           valid > 4 ? -1 : 0,
                           valid > 5 ? -1 : 0,
                           valid > 6 ? -1 : 0,
                           valid > 7 ? -1 : 0);
}

static __attribute__((always_inline)) inline __m256i
leaf_twiddle_mul_q15_256(__m256i x, const int16_t *wr, const int16_t *wi,
           int N, int power, dft_dir_t dir)
{
  power %= N;
  if (power < 0) power += N;
  return complex_mul8_bcast_q15_256_parent(
      x, wr[power], twiddle_im_scalar_dir_i16(wi[power], dir));
}

static __attribute__((always_inline)) inline void
dft4x8_raw_q15_256(__m256i x0, __m256i x1, __m256i x2, __m256i x3,
             __m256i y[4], dft_dir_t dir)
{
  const __m256i s02 = _mm256_adds_epi16(x0, x2);
  const __m256i d02 = _mm256_subs_epi16(x0, x2);
  const __m256i s13 = _mm256_adds_epi16(x1, x3);
  const __m256i d13 = _mm256_subs_epi16(x1, x3);
  y[0] = _mm256_adds_epi16(s02, s13);
  y[2] = _mm256_subs_epi16(s02, s13);
  y[1] = _mm256_adds_epi16(d02, mul_minus_j_dir_i16_256(d13, dir));
  y[3] = _mm256_adds_epi16(d02, mul_plus_j_dir_i16_256(d13, dir));
}

static __attribute__((always_inline)) inline void
load_batched_leaf_inputs(const c16_t *b, int M, int first, int valid,
                    int leaf_n, __m256i *x)
{
  const __m256i zero = _mm256_setzero_si256();
  for (int off = 0; off < leaf_n; off += 8) {
    __m256i z[8];
    for (int lane = 0; lane < 8; lane++)
      z[lane] = lane < valid
                    ? _mm256_loadu_si256((const __m256i *)(b + (first + lane) * M + off))
                    : zero;
    transpose8_complex_i16_256(&z[0], &z[1], &z[2], &z[3],
                               &z[4], &z[5], &z[6], &z[7]);
    for (int lane = 0; lane < 8; lane++)
      x[off + lane] = z[lane];
  }
}

static __attribute__((always_inline)) inline void
store_batched_leaf_outputs(c16_t *dst, int R, int first, int valid,
                     int leaf_n, const __m256i *y)
{
  const __m256i mask = complex_lane_mask_epi32(valid);
  for (int k = 0; k < leaf_n; k++)
    _mm256_maskstore_epi32((int *)(dst + R * k + first), mask, y[k]);
}

static __attribute__((always_inline)) inline void
dft8x8_twiddle_scaled_store(const c16_t *b, int M, c16_t *dst,
                          int R, int first, int valid, dft_dir_t dir)
{
  __m256i x[8], a[4][2], y[8];
  load_batched_leaf_inputs(b, M, first, valid, 8, x);

  for (int n1 = 0; n1 < 4; n1++) {
    const __m256i s = _mm256_adds_epi16(x[n1], x[n1 + 4]);
    const __m256i d = _mm256_subs_epi16(x[n1], x[n1 + 4]);
    a[n1][0] = leaf_twiddle_mul_q15_256(s, dft8_leaf_twiddle_re, dft8_leaf_twiddle_im_fwd, 8, 0, dir);
    a[n1][1] = leaf_twiddle_mul_q15_256(d, dft8_leaf_twiddle_re, dft8_leaf_twiddle_im_fwd, 8, n1, dir);
  }

  for (int k2 = 0; k2 < 2; k2++) {
    __m256i z[4];
    dft4x8_raw_q15_256(a[0][k2], a[1][k2], a[2][k2], a[3][k2], z, dir);
    for (int k1 = 0; k1 < 4; k1++)
      y[k2 + 2 * k1] = z[k1];
  }
  store_batched_leaf_outputs(dst, R, first, valid, 8, y);
}

static __attribute__((always_inline)) inline void
dft16x8_twiddle_scaled_store(const c16_t *b, int M, c16_t *dst,
                           int R, int first, int valid, dft_dir_t dir)
{
  __m256i x[16], a[4][4], y[16];
  load_batched_leaf_inputs(b, M, first, valid, 16, x);

  for (int n1 = 0; n1 < 4; n1++) {
    __m256i z[4];
    dft4x8_raw_q15_256(x[n1 + 4 * 0], x[n1 + 4 * 1],
                 x[n1 + 4 * 2], x[n1 + 4 * 3], z, dir);
    for (int k2 = 0; k2 < 4; k2++)
      a[n1][k2] = leaf_twiddle_mul_q15_256(z[k2], dft16_leaf_twiddle_re, dft16_leaf_twiddle_im_fwd,
                             16, n1 * k2, dir);
  }

  for (int k2 = 0; k2 < 4; k2++) {
    __m256i z[4];
    dft4x8_raw_q15_256(a[0][k2], a[1][k2], a[2][k2], a[3][k2], z, dir);
    for (int k1 = 0; k1 < 4; k1++)
      y[k2 + 4 * k1] = z[k1];
  }
  store_batched_leaf_outputs(dst, R, first, valid, 16, y);
}

static __attribute__((always_inline)) inline void
dft32x8_twiddle_scaled_store(const c16_t *b, int M, c16_t *dst,
                           int R, int first, int valid, dft_dir_t dir)
{
  __m256i x[32], a[8][4], y[32];
  load_batched_leaf_inputs(b, M, first, valid, 32, x);

  for (int n1 = 0; n1 < 8; n1++) {
    __m256i z[4];
    dft4x8_raw_q15_256(x[n1 + 8 * 0], x[n1 + 8 * 1],
                 x[n1 + 8 * 2], x[n1 + 8 * 3], z, dir);
    for (int k2 = 0; k2 < 4; k2++)
      a[n1][k2] = leaf_twiddle_mul_q15_256(z[k2], dft32_leaf_twiddle_re, dft32_leaf_twiddle_im_fwd,
                             32, n1 * k2, dir);
  }

  for (int k2 = 0; k2 < 4; k2++) {
    __m256i z0,z1,z2,z3,z4,z5,z6,z7;
    dft8x8_q15_256_dir(a[0][k2], a[1][k2], a[2][k2], a[3][k2],
                       a[4][k2], a[5][k2], a[6][k2], a[7][k2],
                       &z0,&z1,&z2,&z3,&z4,&z5,&z6,&z7,dir);
    const __m256i z[8] = {z0,z1,z2,z3,z4,z5,z6,z7};
    for (int k1 = 0; k1 < 8; k1++)
      y[k2 + 4 * k1] = z[k1];
  }
  store_batched_leaf_outputs(dst, R, first, valid, 32, y);
}

static const int16_t dft12_leaf_twiddle_re[12]={9459,8192,4730,0,-4730,-8192,-9459,-8192,-4730,0,4730,8192};
static const int16_t dft12_leaf_twiddle_im_fwd[12]={0,-4730,-8192,-9459,-8192,-4730,0,4730,8192,9459,8192,4730};

static __attribute__((always_inline)) inline void
load_four_leaf_inputs_gathered(const c16_t*b,int M,int first,int valid,int leaf_n,__m256i*x)
{
  const __m256i idx=_mm256_setr_epi32(0,M,2*M,3*M,0,0,0,0);
  const __m256i mask=_mm256_setr_epi32(valid>0?-1:0,valid>1?-1:0,valid>2?-1:0,valid>3?-1:0,0,0,0,0);
  const __m256i zero=_mm256_setzero_si256();
  for(int k=0;k<leaf_n;k++)
    x[k]=_mm256_mask_i32gather_epi32(zero,(const int*)(b+first*M+k),idx,mask,4);
}
static __attribute__((always_inline)) inline void
dft12x8_twiddle_scaled_store(const c16_t*b,int M,c16_t*dst,int R,int first,int valid,dft_dir_t dir)
{
  __m256i x[12],a[4][3],y[12];load_batched_leaf_inputs(b,M,first,valid,12,x);
  for(int n1=0;n1<4;n1++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(x[n1],x[n1+4],x[n1+8],&z0,&z1,&z2,dir);a[n1][0]=leaf_twiddle_mul_q15_256(z0,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,0,dir);a[n1][1]=leaf_twiddle_mul_q15_256(z1,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,n1,dir);a[n1][2]=leaf_twiddle_mul_q15_256(z2,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,2*n1,dir);}
  for(int k2=0;k2<3;k2++){__m256i z[4];dft4x8_raw_q15_256(a[0][k2],a[1][k2],a[2][k2],a[3][k2],z,dir);for(int k1=0;k1<4;k1++)y[k2+3*k1]=z[k1];}
  store_batched_leaf_outputs(dst,R,first,valid,12,y);
}
static __attribute__((always_inline)) inline void
dft12x4_twiddle_scaled_store(const c16_t*b,int M,c16_t*dst,int R,int first,int valid,dft_dir_t dir)
{
  __m256i x[12],a[4][3],y[12];load_four_leaf_inputs_gathered(b,M,first,valid,12,x);
  for(int n1=0;n1<4;n1++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(x[n1],x[n1+4],x[n1+8],&z0,&z1,&z2,dir);a[n1][0]=leaf_twiddle_mul_q15_256(z0,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,0,dir);a[n1][1]=leaf_twiddle_mul_q15_256(z1,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,n1,dir);a[n1][2]=leaf_twiddle_mul_q15_256(z2,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,2*n1,dir);}
  for(int k2=0;k2<3;k2++){__m256i z[4];dft4x8_raw_q15_256(a[0][k2],a[1][k2],a[2][k2],a[3][k2],z,dir);for(int k1=0;k1<4;k1++)y[k2+3*k1]=z[k1];}
  store_batched_leaf_outputs(dst,R,first,valid,12,y);
}
static __attribute__((always_inline)) inline void
load_batched_leaf_pointers(const c16_t*const p[8],int valid,int leaf_n,__m256i*x)
{
  const __m256i zero=_mm256_setzero_si256();
  for(int off=0;off<leaf_n;off+=8){__m256i z[8];for(int lane=0;lane<8;lane++)z[lane]=lane<valid?_mm256_loadu_si256((const __m256i*)(p[lane]+off)):zero;transpose8_complex_i16_256(&z[0],&z[1],&z[2],&z[3],&z[4],&z[5],&z[6],&z[7]);for(int lane=0;lane<8;lane++)x[off+lane]=z[lane];}
}
static __attribute__((always_inline)) inline void
store_batched_leaf_pointer_outputs(c16_t*dst,int R,int branch,int valid,int leaf_n,const __m256i*y)
{
  const __m256i mask=complex_lane_mask_epi32(valid);for(int k=0;k<leaf_n;k++)_mm256_maskstore_epi32((int*)(dst+R*k+branch),mask,y[k]);
}
static __attribute__((always_inline)) inline void
dft12x8_pointer_batch_store(const c16_t*const p[8],c16_t*dst,int R,int branch,int valid,dft_dir_t dir)
{
  __m256i x[12],a[4][3],y[12];load_batched_leaf_pointers(p,valid,12,x);for(int n1=0;n1<4;n1++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(x[n1],x[n1+4],x[n1+8],&z0,&z1,&z2,dir);a[n1][0]=leaf_twiddle_mul_q15_256(z0,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,0,dir);a[n1][1]=leaf_twiddle_mul_q15_256(z1,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,n1,dir);a[n1][2]=leaf_twiddle_mul_q15_256(z2,dft12_leaf_twiddle_re,dft12_leaf_twiddle_im_fwd,12,2*n1,dir);}for(int k2=0;k2<3;k2++){__m256i z[4];dft4x8_raw_q15_256(a[0][k2],a[1][k2],a[2][k2],a[3][k2],z,dir);for(int k1=0;k1<4;k1++)y[k2+3*k1]=z[k1];}store_batched_leaf_pointer_outputs(dst,R,branch,valid,12,y);
}

static __attribute__((always_inline)) inline void dft4x8_q15_256_exact(const __m256i x0,
                                                                               const __m256i x1,
                                                                               const __m256i x2,
                                                                               const __m256i x3,
                                                                               __m256i *Y0,
                                                                               __m256i *Y1,
                                                                               __m256i *Y2,
                                                                               __m256i *Y3,
                                                                               dft_dir_t dir)
{
  const __m256i x0s = _mm256_srai_epi16(x0, 1);
  const __m256i x1s = _mm256_srai_epi16(x1, 1);
  const __m256i x2s = _mm256_srai_epi16(x2, 1);
  const __m256i x3s = _mm256_srai_epi16(x3, 1);

  const __m256i s02 = _mm256_adds_epi16(x0s, x2s);
  const __m256i d02 = _mm256_subs_epi16(x0s, x2s);
  const __m256i s13 = _mm256_adds_epi16(x1s, x3s);
  const __m256i d13 = _mm256_subs_epi16(x1s, x3s);

  *Y0 = _mm256_adds_epi16(s02, s13);
  *Y2 = _mm256_subs_epi16(s02, s13);
  *Y1 = _mm256_adds_epi16(d02, mul_minus_j_dir_i16_256(d13, dir));
  *Y3 = _mm256_adds_epi16(d02, mul_plus_j_dir_i16_256(d13, dir));
}

static __attribute__((always_inline)) inline void dft8x8_q15_256_unitary_exact(const __m256i x0,
                                                                                 const __m256i x1,
                                                                                 const __m256i x2,
                                                                                 const __m256i x3,
                                                                                 const __m256i x4,
                                                                                 const __m256i x5,
                                                                                 const __m256i x6,
                                                                                 const __m256i x7,
                                                                                 __m256i Y[8],
                                                                                 dft_dir_t dir)
{
  __m256i E0, E1, E2, E3;
  __m256i O0, O1, O2, O3;

  dft4x8_q15_256_exact(x0, x2, x4, x6, &E0, &E1, &E2, &E3, dir);
  dft4x8_q15_256_exact(x1, x3, x5, x7, &O0, &O1, &O2, &O3, dir);

  const __m256i qinv = _mm256_set1_epi16(Q15_INV_SQRT2);
  const __m256i E0s = _mm256_mulhrs_epi16(E0, qinv);
  const __m256i E1s = _mm256_mulhrs_epi16(E1, qinv);
  const __m256i E2s = _mm256_mulhrs_epi16(E2, qinv);
  const __m256i E3s = _mm256_mulhrs_epi16(E3, qinv);

  const __m256i T0 = _mm256_mulhrs_epi16(O0, qinv);
  const __m256i T1 = complex_mul8_bcast_q15_256_parent(O1, Q15_HALF, twiddle_im_scalar_dir_i16(-Q15_HALF, dir));
  const __m256i T2 = complex_mul8_bcast_q15_256_parent(O2, 0, twiddle_im_scalar_dir_i16(-Q15_INV_SQRT2, dir));
  const __m256i T3 = complex_mul8_bcast_q15_256_parent(O3, -Q15_HALF, twiddle_im_scalar_dir_i16(-Q15_HALF, dir));

  Y[0] = _mm256_adds_epi16(E0s, T0);
  Y[4] = _mm256_subs_epi16(E0s, T0);
  Y[1] = _mm256_adds_epi16(E1s, T1);
  Y[5] = _mm256_subs_epi16(E1s, T1);
  Y[2] = _mm256_adds_epi16(E2s, T2);
  Y[6] = _mm256_subs_epi16(E2s, T2);
  Y[3] = _mm256_adds_epi16(E3s, T3);
  Y[7] = _mm256_subs_epi16(E3s, T3);
}

static inline void dft8x8_q15_256_parent(const __m256i x0,
                                          const __m256i x1,
                                          const __m256i x2,
                                          const __m256i x3,
                                          const __m256i x4,
                                          const __m256i x5,
                                          const __m256i x6,
                                          const __m256i x7,
                                          __m256i Y[8],
                                          dft_dir_t dir)
{
  /* The DFT32 decomposition requires a unitary DFT8. Scaling the inputs by
   * 1/2 first keeps the radix-8 butterfly inside Q15 range. The final
   * 1/sqrt(2) multiplication gives the same 1/sqrt(8) normalization while
   * allowing the optimized direct radix-8 kernel to handle all W8 factors. */
  const __m256i x0s = _mm256_srai_epi16(x0, 1);
  const __m256i x1s = _mm256_srai_epi16(x1, 1);
  const __m256i x2s = _mm256_srai_epi16(x2, 1);
  const __m256i x3s = _mm256_srai_epi16(x3, 1);
  const __m256i x4s = _mm256_srai_epi16(x4, 1);
  const __m256i x5s = _mm256_srai_epi16(x5, 1);
  const __m256i x6s = _mm256_srai_epi16(x6, 1);
  const __m256i x7s = _mm256_srai_epi16(x7, 1);

  __m256i T0, T1, T2, T3, T4, T5, T6, T7;
  dft8x8_q15_256_dir(x0s,
                      x1s,
                      x2s,
                      x3s,
                      x4s,
                      x5s,
                      x6s,
                      x7s,
                      &T0,
                      &T1,
                      &T2,
                      &T3,
                      &T4,
                      &T5,
                      &T6,
                      &T7,
                      dir);

  const __m256i qinv = _mm256_set1_epi16(Q15_INV_SQRT2);
  Y[0] = _mm256_mulhrs_epi16(T0, qinv);
  Y[1] = _mm256_mulhrs_epi16(T1, qinv);
  Y[2] = _mm256_mulhrs_epi16(T2, qinv);
  Y[3] = _mm256_mulhrs_epi16(T3, qinv);
  Y[4] = _mm256_mulhrs_epi16(T4, qinv);
  Y[5] = _mm256_mulhrs_epi16(T5, qinv);
  Y[6] = _mm256_mulhrs_epi16(T6, qinv);
  Y[7] = _mm256_mulhrs_epi16(T7, qinv);
}

static __attribute__((always_inline)) inline void dft32_front_q15_256_parent(
    const c16_t *src,
    const __m256i W1_RE,
    const __m256i W1_IM,
    const __m256i W2_RE,
    const __m256i W2_IM,
    const __m256i W3_RE,
    const __m256i W3_IM,
    __m256i *H0,
    __m256i *H1,
    __m256i *H2,
    __m256i *H3,
    dft_dir_t dir)
{
  const __m256i x0 = _mm256_loadu_si256((const __m256i *)(src + 0));
  const __m256i x1 = _mm256_loadu_si256((const __m256i *)(src + 8));
  const __m256i x2 = _mm256_loadu_si256((const __m256i *)(src + 16));
  const __m256i x3 = _mm256_loadu_si256((const __m256i *)(src + 24));

  dft4x4_q15_256x2(x0, x1, x2, x3, H0, H1, H2, H3, dir);

  *H1 = complex_mul8_prepack_q15_256(*H1, W1_RE, W1_IM);
  *H2 = complex_mul8_prepack_q15_256(*H2, W2_RE, W2_IM);
  *H3 = complex_mul8_prepack_q15_256(*H3, W3_RE, W3_IM);

  transpose4_complex_i16_256x2_shuffle(H0, H1, H2, H3);
}

/* Two DFT32s share the W32 coefficient loads. Their four parallel DFT8
 * tails are packed into the two 128-bit lanes of YMM registers and executed
 * together, so the final radix-8 stage handles eight transforms per AVX2 op. */
static __attribute__((always_inline)) inline void dft32x2_q15_256_parent(const c16_t *src0,
                                                                          const c16_t *src1,
                                                                          __m256i Y[8],
                                                                          dft_dir_t dir)
{
  const __m256i W1_RE = _mm256_load_si256((const __m256i *)W32_1_RE_RE_256);
  const __m256i W1_IM =
      twiddle_im_dir_256(_mm256_load_si256((const __m256i *)W32_1_IM_SIGNED_256), dir);
  const __m256i W2_RE = _mm256_load_si256((const __m256i *)W32_2_RE_RE_256);
  const __m256i W2_IM =
      twiddle_im_dir_256(_mm256_load_si256((const __m256i *)W32_2_IM_SIGNED_256), dir);
  const __m256i W3_RE = _mm256_load_si256((const __m256i *)W32_3_RE_RE_256);
  const __m256i W3_IM =
      twiddle_im_dir_256(_mm256_load_si256((const __m256i *)W32_3_IM_SIGNED_256), dir);

  __m256i A0, A1, A2, A3;
  __m256i B0, B1, B2, B3;

  dft32_front_q15_256_parent(
      src0, W1_RE, W1_IM, W2_RE, W2_IM, W3_RE, W3_IM, &A0, &A1, &A2, &A3, dir);
  dft32_front_q15_256_parent(
      src1, W1_RE, W1_IM, W2_RE, W2_IM, W3_RE, W3_IM, &B0, &B1, &B2, &B3, dir);

  const __m256i x0 = _mm256_permute2x128_si256(A0, B0, 0x20);
  const __m256i x1 = _mm256_permute2x128_si256(A1, B1, 0x20);
  const __m256i x2 = _mm256_permute2x128_si256(A2, B2, 0x20);
  const __m256i x3 = _mm256_permute2x128_si256(A3, B3, 0x20);
  const __m256i x4 = _mm256_permute2x128_si256(A0, B0, 0x31);
  const __m256i x5 = _mm256_permute2x128_si256(A1, B1, 0x31);
  const __m256i x6 = _mm256_permute2x128_si256(A2, B2, 0x31);
  const __m256i x7 = _mm256_permute2x128_si256(A3, B3, 0x31);

  dft8x8_q15_256_parent(x0, x1, x2, x3, x4, x5, x6, x7, Y, dir);
}

static __attribute__((always_inline)) inline void dft32x4_parent_store(const c16_t *src0,
                                                                        const c16_t *src1,
                                                                        const c16_t *src2,
                                                                        const c16_t *src3,
                                                                        c16_t *dst,
                                                                        int output_stride,
                                                                        int parent_branch,
                                                                        dft_dir_t dir)
{
  __m256i ab[8] __attribute__((aligned(32)));
  __m256i cd[8] __attribute__((aligned(32)));

  dft32x2_q15_256_parent(src0, src1, ab, dir);
  dft32x2_q15_256_parent(src2, src3, cd, dir);

  for (int block = 0; block < 8; block += 2) {
    __m256i A = _mm256_permute2x128_si256(ab[block], ab[block + 1], 0x20);
    __m256i B = _mm256_permute2x128_si256(ab[block], ab[block + 1], 0x31);
    __m256i C = _mm256_permute2x128_si256(cd[block], cd[block + 1], 0x20);
    __m256i D = _mm256_permute2x128_si256(cd[block], cd[block + 1], 0x31);

    transpose4_complex_i16_256x2_shuffle(&A, &B, &C, &D);

    const int k0 = 4 * block;

#define DFT32X4_STORE_ROW(ROW, KOFF)                                                                \
    do {                                                                                             \
      _mm_storeu_si128((__m128i *)(dst + output_stride * ((KOFF) + 0) + parent_branch),              \
                       _mm256_castsi256_si128((ROW)));                                                \
      _mm_storeu_si128((__m128i *)(dst + output_stride * ((KOFF) + 4) + parent_branch),              \
                       _mm256_extracti128_si256((ROW), 1));                                           \
    } while (0)

    DFT32X4_STORE_ROW(A, k0);
    DFT32X4_STORE_ROW(B, k0 + 1);
    DFT32X4_STORE_ROW(C, k0 + 2);
    DFT32X4_STORE_ROW(D, k0 + 3);

#undef DFT32X4_STORE_ROW
  }
}

static void radix9_terminal_leaf8_direct(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 72, M = 8 };
  c16_t b[N] __attribute__((aligned(64)));
  c16_t tail[M] __attribute__((aligned(16)));

  const radix9_plan_t *plan = radix9_plan_get(N);
  AssertFatal(plan, "Missing radix-9 direct plan N=%d\n", N);
  radix9_stage_to_branch_major_q15_256_selected(src, b, N, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));
  radix9_dft8x4_parent_store(b + 0 * M, dst, M, 0, dir);
  radix9_dft8x4_parent_store(b + 4 * M, dst, M, 4, dir);

  dft8_avx(b + 8 * M, tail, dir);
  for (int k = 0; k < M; k++)
    dst[9 * k + 8] = tail[k];
}


static __attribute__((always_inline)) inline void radix9_terminal_leaf32_direct(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 288, M = 32 };
  c16_t b[N] __attribute__((aligned(64)));
  c16_t tail[M] __attribute__((aligned(64)));

  const radix9_plan_t *plan = radix9_plan_get(N);
  AssertFatal(plan, "Missing radix-9 direct plan N=%d\n", N);
  radix9_stage_to_branch_major_q15_256_288(src, b, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));
  dft32x4_parent_store(b + 0 * M, b + 1 * M, b + 2 * M, b + 3 * M, dst, 9, 0, dir);
  dft32x4_parent_store(b + 4 * M, b + 5 * M, b + 6 * M, b + 7 * M, dst, 9, 4, dir);

  dft32_q15_128(b + 8 * M, tail, dir);
  for (int k = 0; k < M; k++)
    dst[9 * k + 8] = tail[k];
}


static void radix9_terminal_leaf12_direct(const c16_t*src,c16_t*dst,dft_dir_t dir)
{
  enum{N=108,R=9,M=12};c16_t b[N] __attribute__((aligned(64)));
  const radix9_plan_t*plan=radix9_plan_get(N);AssertFatal(plan,"Missing radix-9 direct plan N=%d\n",N);
  radix9_stage_to_branch_major_q15_256_128_hybrid_108(src,b,dir,r3_twiddle_slot(N),r3_twiddle_slot(N/3));
  dft12x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft12x4_twiddle_scaled_store(b,M,dst,R,8,1,dir);
}


static __attribute__((always_inline)) inline void radix81_dft8x8_parent_store(const c16_t *src0,
                                                                                const c16_t *src1,
                                                                                const c16_t *src2,
                                                                                const c16_t *src3,
                                                                                const c16_t *src4,
                                                                                const c16_t *src5,
                                                                                const c16_t *src6,
                                                                                const c16_t *src7,
                                                                                c16_t *dst,
                                                                                int parent_branch,
                                                                                dft_dir_t dir)
{
  __m256i x0 = _mm256_load_si256((const __m256i *)src0);
  __m256i x1 = _mm256_load_si256((const __m256i *)src1);
  __m256i x2 = _mm256_load_si256((const __m256i *)src2);
  __m256i x3 = _mm256_load_si256((const __m256i *)src3);
  __m256i x4 = _mm256_load_si256((const __m256i *)src4);
  __m256i x5 = _mm256_load_si256((const __m256i *)src5);
  __m256i x6 = _mm256_load_si256((const __m256i *)src6);
  __m256i x7 = _mm256_load_si256((const __m256i *)src7);

  transpose8_complex_i16_256(&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7);

  __m256i y[8];
  dft8x8_q15_256_unitary_exact(x0, x1, x2, x3, x4, x5, x6, x7, y, dir);

  for (int k = 0; k < 8; k++)
    _mm256_storeu_si256((__m256i *)(dst + 81 * k + parent_branch), y[k]);
}

static void radix81_terminal_leaf8_direct(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 648, M1 = 72, M2 = 8 };
  c16_t work[2 * N] __attribute__((aligned(64)));
  c16_t *outer_b = work;
  c16_t *stage2 = work + N;
  c16_t tail[M2] __attribute__((aligned(16)));

  const radix9_plan_t *outer_plan = radix9_plan_get(N);
  const radix9_plan_t *inner_plan = radix9_plan_get(M1);
  AssertFatal(outer_plan && inner_plan, "Missing radix-81 direct plans N=%d M1=%d\n", N, M1);
  radix9_stage_to_branch_major_q15_256_selected(src, outer_b, N, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));
  for (int outer = 0; outer < 9; outer++)
    radix9_stage_to_branch_major_q15_256_selected(outer_b + outer * M1,
                                                   stage2 + outer * M1,
                                                   M1,
                                                   dir,
                                                   r3_twiddle_slot(M1),
                                                   r3_twiddle_slot(M1 / 3));

  for (int inner = 0; inner < 9; inner++) {
    const int branch = 9 * inner;
    const c16_t *p0 = stage2 + 0 * M1 + inner * M2;
    const c16_t *p1 = stage2 + 1 * M1 + inner * M2;
    const c16_t *p2 = stage2 + 2 * M1 + inner * M2;
    const c16_t *p3 = stage2 + 3 * M1 + inner * M2;
    const c16_t *p4 = stage2 + 4 * M1 + inner * M2;
    const c16_t *p5 = stage2 + 5 * M1 + inner * M2;
    const c16_t *p6 = stage2 + 6 * M1 + inner * M2;
    const c16_t *p7 = stage2 + 7 * M1 + inner * M2;
    const c16_t *p8 = stage2 + 8 * M1 + inner * M2;

    radix81_dft8x8_parent_store(p0, p1, p2, p3, p4, p5, p6, p7, dst, branch, dir);
    dft8_avx(p8, tail, dir);
    for (int k = 0; k < M2; k++)
      dst[81 * k + branch + 8] = tail[k];
  }
}


static void radix81_terminal_leaf16_direct_w16folded(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 1296, M1 = 144, M2 = 16 };
  c16_t work[2 * N] __attribute__((aligned(64)));
  c16_t *outer_b = work;
  c16_t *stage2 = work + N;
  c16_t tail[M2] __attribute__((aligned(16)));

  const radix9_plan_t *outer_plan = radix9_plan_get(N);
  const radix9_plan_t *inner_plan = radix9_plan_get(M1);
  AssertFatal(outer_plan && inner_plan, "Missing radix-81 direct plans N=%d M1=%d\n", N, M1);
  radix9_stage_to_branch_major_q15_256_selected(src, outer_b, N, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));
  for (int outer = 0; outer < 9; outer++)
    radix9_stage_to_branch_major_q15_256_selected(outer_b + outer * M1,
                                                   stage2 + outer * M1,
                                                   M1,
                                                   dir,
                                                   r3_twiddle_slot(M1),
                                                   r3_twiddle_slot(M1 / 3));

  selected_q15_twiddles_init();
  const __m256i dft16x8_scale = _mm256_set1_epi16(Q15_INV_SQRT8);

  for (int inner = 0; inner < 9; inner++) {
    const int branch = 9 * inner;
    const c16_t *p0 = stage2 + 0 * M1 + inner * M2;
    const c16_t *p1 = stage2 + 1 * M1 + inner * M2;
    const c16_t *p2 = stage2 + 2 * M1 + inner * M2;
    const c16_t *p3 = stage2 + 3 * M1 + inner * M2;
    const c16_t *p4 = stage2 + 4 * M1 + inner * M2;
    const c16_t *p5 = stage2 + 5 * M1 + inner * M2;
    const c16_t *p6 = stage2 + 6 * M1 + inner * M2;
    const c16_t *p7 = stage2 + 7 * M1 + inner * M2;
    const c16_t *p8 = stage2 + 8 * M1 + inner * M2;

    __m256i x[16] __attribute__((aligned(64)));
    __m256i y[16] __attribute__((aligned(64)));

    __m256i z0 = _mm256_loadu_si256((const __m256i *)(p0 + 0));
    __m256i z1 = _mm256_loadu_si256((const __m256i *)(p1 + 0));
    __m256i z2 = _mm256_loadu_si256((const __m256i *)(p2 + 0));
    __m256i z3 = _mm256_loadu_si256((const __m256i *)(p3 + 0));
    __m256i z4 = _mm256_loadu_si256((const __m256i *)(p4 + 0));
    __m256i z5 = _mm256_loadu_si256((const __m256i *)(p5 + 0));
    __m256i z6 = _mm256_loadu_si256((const __m256i *)(p6 + 0));
    __m256i z7 = _mm256_loadu_si256((const __m256i *)(p7 + 0));
    transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    x[0] = z0;
    x[1] = z1;
    x[2] = z2;
    x[3] = z3;
    x[4] = z4;
    x[5] = z5;
    x[6] = z6;
    x[7] = z7;

    z0 = _mm256_loadu_si256((const __m256i *)(p0 + 8));
    z1 = _mm256_loadu_si256((const __m256i *)(p1 + 8));
    z2 = _mm256_loadu_si256((const __m256i *)(p2 + 8));
    z3 = _mm256_loadu_si256((const __m256i *)(p3 + 8));
    z4 = _mm256_loadu_si256((const __m256i *)(p4 + 8));
    z5 = _mm256_loadu_si256((const __m256i *)(p5 + 8));
    z6 = _mm256_loadu_si256((const __m256i *)(p6 + 8));
    z7 = _mm256_loadu_si256((const __m256i *)(p7 + 8));
    transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    x[8] = z0;
    x[9] = z1;
    x[10] = z2;
    x[11] = z3;
    x[12] = z4;
    x[13] = z5;
    x[14] = z6;
    x[15] = z7;

    dft16x8_selected_w16folded(x, y, dir);
    for (int k = 0; k < M2; k++) {
      y[k] = _mm256_mulhrs_epi16(y[k], dft16x8_scale);
      _mm256_storeu_si256((__m256i *)(dst + 81 * k + branch), y[k]);
    }

    dft16_q15_128(p8, tail, dir);
    for (int k = 0; k < M2; k++)
      dst[81 * k + branch + 8] = tail[k];
  }
}

static void radix81_terminal_leaf32_direct_2592(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { N = 2592, M1 = 288, M2 = 32 };
  c16_t work[2 * N] __attribute__((aligned(64)));
  c16_t *outer_b = work;
  c16_t *stage2 = work + N;
  c16_t tail[M2] __attribute__((aligned(64)));

  const radix9_plan_t *outer_plan = radix9_plan_get(N);
  const radix9_plan_t *inner_plan = radix9_plan_get(M1);
  AssertFatal(outer_plan && inner_plan, "Missing radix-81 direct plans N=%d M1=%d\n", N, M1);

  radix9_stage_to_branch_major_q15_256_selected(src,
                                                 outer_b,
                                                 N,
                                                 dir,
                                                 r3_twiddle_slot(N),
                                                 r3_twiddle_slot(N / 3));

  for (int outer = 0; outer < 9; outer++)
    radix9_stage_to_branch_major_q15_256_selected(outer_b + outer * M1,
                                                   stage2 + outer * M1,
                                                   M1,
                                                   dir,
                                                   r3_twiddle_slot(M1),
                                                   r3_twiddle_slot(M1 / 3));

  for (int inner = 0; inner < 9; inner++) {
    const int branch = 9 * inner;
    const c16_t *p0 = stage2 + 0 * M1 + inner * M2;
    const c16_t *p1 = stage2 + 1 * M1 + inner * M2;
    const c16_t *p2 = stage2 + 2 * M1 + inner * M2;
    const c16_t *p3 = stage2 + 3 * M1 + inner * M2;
    const c16_t *p4 = stage2 + 4 * M1 + inner * M2;
    const c16_t *p5 = stage2 + 5 * M1 + inner * M2;
    const c16_t *p6 = stage2 + 6 * M1 + inner * M2;
    const c16_t *p7 = stage2 + 7 * M1 + inner * M2;
    const c16_t *p8 = stage2 + 8 * M1 + inner * M2;

    dft32x4_parent_store(p0, p1, p2, p3, dst, 81, branch + 0, dir);
    dft32x4_parent_store(p4, p5, p6, p7, dst, 81, branch + 4, dir);

    dft32_q15_128(p8, tail, dir);
    for (int k = 0; k < M2; k++)
      dst[81 * k + branch + 8] = tail[k];
  }
}

static void radix81_terminal_leaf12_direct_972(const c16_t*src,c16_t*dst,dft_dir_t dir)
{
  enum{N=972,M1=108,M2=12};c16_t work[2*N] __attribute__((aligned(64)));c16_t*outer_b=work,*stage2=work+N;
    const radix9_plan_t *outer_plan = radix9_plan_get(N);
  const radix9_plan_t *inner_plan = radix9_plan_get(M1);
  AssertFatal(outer_plan && inner_plan, "Missing radix-81 plans N=%d M1=%d\n", N, M1);
radix9_stage_to_branch_major_q15_256_128_hybrid_972(src,outer_b,dir,r3_twiddle_slot(N),r3_twiddle_slot(N/3));
  for(int outer=0;outer<9;outer++)radix9_stage_to_branch_major_q15_256_128_hybrid_108(outer_b+outer*M1,stage2+outer*M1,dir,r3_twiddle_slot(M1),r3_twiddle_slot(M1/3));
  for(int inner=0;inner<9;inner++){
    const int branch=9*inner;const c16_t*p[9];for(int outer=0;outer<9;outer++)p[outer]=stage2+outer*M1+inner*M2;
    const c16_t*p8a[8]={p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]},*p1a[8]={p[8],0,0,0,0,0,0,0};
    dft12x8_pointer_batch_store(p8a,dst,81,branch,8,dir);dft12x8_pointer_batch_store(p1a,dst,81,branch+8,1,dir);
  }
}

static inline void radix81_scatter4_q15_128(const c16_t *y, c16_t *dst, int M, int k)
{
  for (int br = 0; br < 80; br += 4) {
    __m128i r0 = _mm_load_si128((const __m128i *)(y + (br + 0) * M + k));
    __m128i r1 = _mm_load_si128((const __m128i *)(y + (br + 1) * M + k));
    __m128i r2 = _mm_load_si128((const __m128i *)(y + (br + 2) * M + k));
    __m128i r3 = _mm_load_si128((const __m128i *)(y + (br + 3) * M + k));

    transpose4_complex_i16_128(&r0, &r1, &r2, &r3);

    _mm_storeu_si128((__m128i *)(dst + 81 * (k + 0) + br), r0);
    _mm_storeu_si128((__m128i *)(dst + 81 * (k + 1) + br), r1);
    _mm_storeu_si128((__m128i *)(dst + 81 * (k + 2) + br), r2);
    _mm_storeu_si128((__m128i *)(dst + 81 * (k + 3) + br), r3);
  }

  c16_t tail[4] __attribute__((aligned(16)));
  _mm_store_si128((__m128i *)tail, _mm_load_si128((const __m128i *)(y + 80 * M + k)));

  dst[81 * (k + 0) + 80] = tail[0];
  dst[81 * (k + 1) + 80] = tail[1];
  dst[81 * (k + 2) + 80] = tail[2];
  dst[81 * (k + 3) + 80] = tail[3];
}

static inline int radix81_selected_size(int N)
{
  return N == 972 || N == 2592;
}

static inline int selected_radix9_radix18_size(int N)
{
  return N == 576 || N == 1152 || N == 2304;
}

static void radix81_selected(const c16_t *src, c16_t *dst, int N, dft_dir_t dir)
{
  const int M1 = N / 9;
  const int M2 = N / 81;
  AssertFatal((N % 81) == 0 && (M2 & 3) == 0, "Invalid radix-81 selected N=%d M2=%d\n", N, M2);
  AssertFatal(M2 == 8 || M2 == 12 || M2 == 16 || M2 == 32, "Unsupported radix-81 leaf N=%d M2=%d\n", N, M2);

  c16_t work[2 * N + M1] __attribute__((aligned(64)));
  c16_t *outer_b = work;
  c16_t *inner_b = outer_b + N;
  c16_t *combined_y = inner_b + M1;

  const radix9_plan_t *outer_plan = radix9_plan_get(N);
  const radix9_plan_t *inner_plan = radix9_plan_get(M1);
  AssertFatal(outer_plan && inner_plan, "Missing radix-81 selected plans N=%d M1=%d\n", N, M1);
  if (N == 972)
    radix9_stage_to_branch_major_q15_256_128_hybrid_972(src, outer_b, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));
  else if (N == 2592)
    radix9_stage_to_branch_major_q15_256_selected(src, outer_b, N, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));
  else
    radix9_stage_to_branch_major_q15_128_plan(src, outer_b, N, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));

#define R81_RUN_LEAVES(LEAF_FN)                                                                     \
  do {                                                                                               \
    for (int outer = 0; outer < 9; outer++) {                                                        \
      if (N == 972)                                                                                   \
        radix9_stage_to_branch_major_q15_256_128_hybrid_108(outer_b + outer * M1, inner_b, dir,      \
                                                             r3_twiddle_slot(M1), r3_twiddle_slot(M1 / 3)); \
      else if (N == 2592)                                                                             \
        radix9_stage_to_branch_major_q15_256_selected(outer_b + outer * M1, inner_b, M1, dir,        \
                                                       r3_twiddle_slot(M1), r3_twiddle_slot(M1 / 3)); \
      else                                                                                             \
        radix9_stage_to_branch_major_q15_128_plan(outer_b + outer * M1, inner_b, M1, dir,            \
                                                   r3_twiddle_slot(M1), r3_twiddle_slot(M1 / 3));     \
      for (int inner = 0; inner < 9; inner++) {                                                       \
        const int combined = 9 * inner + outer;                                                       \
        LEAF_FN(inner_b + inner * M2, combined_y + combined * M2, dir);                               \
      }                                                                                               \
    }                                                                                                 \
  } while (0)

  switch (M2) {
    case 8:
      R81_RUN_LEAVES(dft8_avx);
      break;
    case 12:
      R81_RUN_LEAVES(dft12_q15_128);
      break;
    case 16:
      R81_RUN_LEAVES(dft16_q15_128);
      break;
    case 32:
      R81_RUN_LEAVES(dft32_q15_128);
      break;
    default:
      AssertFatal(0, "Unsupported radix-81 leaf N=%d M2=%d\n", N, M2);
  }

#undef R81_RUN_LEAVES

  for (int k = 0; k < M2; k += 4)
    radix81_scatter4_q15_128(combined_y, dst, M2, k);
}

/* Size-specific dispatcher for transforms whose generic family is radix-9.
 * N=1152 uses the dedicated radix-18 decomposition; the remaining supported
 * sizes keep their radix-9 parent paths. */
static void dispatch_selected_radix9_radix18(const c16_t *src, c16_t *dst, int N, dft_dir_t dir)
{
  const int M = N / 9;
  AssertFatal((N % 9) == 0 && (M & 3) == 0, "Invalid radix-9 selected N=%d M=%d\n", N, M);

  const radix9_plan_t *plan = radix9_plan_get(N);
  AssertFatal(plan, "Missing radix-9 selected plan N=%d\n", N);

  if (N == 1152) {
    dft1152_radix18_leaf64_avx2_selected(src, dst, dir);
    return;
  }

  /* DFT576 uses an AVX2 radix-9 parent. Branches 0..7 are packed directly
   * for the lane-parallel DFT64x8 leaf; branch 8 uses one DFT64 leaf. */
  if (N == 576) {
    __m256i leaf[64] __attribute__((aligned(64)));
    c16_t branch8[64] __attribute__((aligned(64)));
    c16_t tail[64] __attribute__((aligned(64)));

    radix9_stage_to_dft64x8_576(src, leaf, branch8, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));

    selected_q15_twiddles_init();
    const __m256i dc = dft64x8_dc_q15_256(leaf);
    dft64x8_selected_store(leaf, dst, 9, 0, dir);
    _mm256_storeu_si256((__m256i *)dst, dc);

    dft64_avx(branch8, tail, dir);
    for (int k = 0; k < 64; k++)
      dst[9 * k + 8] = tail[k];
    return;
  }

  c16_t work[2 * N] __attribute__((aligned(64)));
  c16_t *b = work;
  c16_t *y = work + N;

  if (N == 1152 || N == 2304)
    radix9_stage_to_branch_major_q15_256_selected(src, b, N, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));
  else
    radix9_stage_to_branch_major_q15_128_plan(src, b, N, dir, r3_twiddle_slot(N), r3_twiddle_slot(N / 3));

  for (int br = 0; br < 9; br++) {
    if (N == 2304)
      dft256_radix16_selected(b + br * M, y + br * M, dir);
    else
      dft_mixed_radix_c16_scaled(b + br * M, y + br * M, M, dir);
  }

  if (N == 2304) {
    for (int k = 0; k < M; k += 8)
      radix9_scatter8_q15_256_selected(y, dst, M, k);
  } else {
    for (int k = 0; k < M; k += 4)
      radix9_scatter4_q15_128(y, dst, M, k);
  }
}

static inline void dft_power2_selected_child(const c16_t *src, c16_t *dst, int N, dft_dir_t dir, c16_t *work)
{
  switch (N) {
    case 8:
      dft8_avx(src, dst, dir);
      return;
    case 16:
      dft16_q15_128(src, dst, dir);
      return;
    case 32:
      dft32_q15_128(src, dst, dir);
      return;
    case 64:
      dft64_avx(src, dst, dir);
      return;
    case 128:
      dft128_dir(src, dst, dir);
      return;
    case 256:
      /* Nested DFT256 uses the unfused W16 scaling order required by its
       * parent path; the contiguous entry uses folded W16 scaling. */
      dft256_radix16_selected(src, dst, dir);
      return;
    default:
      dft_power2_mixed_large_core(src, dst, N, dir, work);
      return;
  }
}

/* R15 = R3 then R5. Both parent stages remain branch-major until the power-of-two leaves. */

static inline void radix5_butterfly8_q15_256(__m256i x0,
                                             __m256i x1,
                                             __m256i x2,
                                             __m256i x3,
                                             __m256i x4,
                                             __m256i *y0,
                                             __m256i *y1,
                                             __m256i *y2,
                                             __m256i *y3,
                                             __m256i *y4,
                                             dft_dir_t dir)
{
  const __m256i t1 = _mm256_adds_epi16(x1, x4);
  const __m256i t2 = _mm256_adds_epi16(x2, x3);
  const __m256i d1 = _mm256_subs_epi16(x1, x4);
  const __m256i d2 = _mm256_subs_epi16(x2, x3);

  *y0 = _mm256_adds_epi16(x0, _mm256_adds_epi16(t1, t2));

  const __m256i b1 = _mm256_adds_epi16(
      x0,
      _mm256_adds_epi16(_mm256_mulhrs_epi16(t1, _mm256_set1_epi16(Q15_COS_2PI_5)),
                        _mm256_mulhrs_epi16(t2, _mm256_set1_epi16(Q15_COS_4PI_5))));
  const __m256i q1 = _mm256_adds_epi16(
      _mm256_mulhrs_epi16(d1, _mm256_set1_epi16(Q15_SIN_2PI_5)),
      _mm256_mulhrs_epi16(d2, _mm256_set1_epi16(Q15_SIN_4PI_5)));
  *y1 = _mm256_adds_epi16(b1, mul_minus_j_dir_i16_256(q1, dir));
  *y4 = _mm256_adds_epi16(b1, mul_plus_j_dir_i16_256(q1, dir));

  const __m256i b2 = _mm256_adds_epi16(
      x0,
      _mm256_adds_epi16(_mm256_mulhrs_epi16(t1, _mm256_set1_epi16(Q15_COS_4PI_5)),
                        _mm256_mulhrs_epi16(t2, _mm256_set1_epi16(Q15_COS_2PI_5))));
  const __m256i q2 = _mm256_subs_epi16(
      _mm256_mulhrs_epi16(d1, _mm256_set1_epi16(Q15_SIN_4PI_5)),
      _mm256_mulhrs_epi16(d2, _mm256_set1_epi16(Q15_SIN_2PI_5)));
  *y2 = _mm256_adds_epi16(b2, mul_minus_j_dir_i16_256(q2, dir));
  *y3 = _mm256_adds_epi16(b2, mul_plus_j_dir_i16_256(q2, dir));
}

static inline void radix15_scatter4_q15_128(const c16_t *y, c16_t *dst, int M, int k)
{
  __m128i a0 = _mm_load_si128((const __m128i *)(y + 0 * M + k));
  __m128i a1 = _mm_load_si128((const __m128i *)(y + 1 * M + k));
  __m128i a2 = _mm_load_si128((const __m128i *)(y + 2 * M + k));
  __m128i a3 = _mm_load_si128((const __m128i *)(y + 3 * M + k));
  __m128i b0 = _mm_load_si128((const __m128i *)(y + 4 * M + k));
  __m128i b1 = _mm_load_si128((const __m128i *)(y + 5 * M + k));
  __m128i b2 = _mm_load_si128((const __m128i *)(y + 6 * M + k));
  __m128i b3 = _mm_load_si128((const __m128i *)(y + 7 * M + k));
  __m128i c0 = _mm_load_si128((const __m128i *)(y + 8 * M + k));
  __m128i c1 = _mm_load_si128((const __m128i *)(y + 9 * M + k));
  __m128i c2 = _mm_load_si128((const __m128i *)(y + 10 * M + k));
  __m128i c3 = _mm_load_si128((const __m128i *)(y + 11 * M + k));

  transpose4_complex_i16_128(&a0, &a1, &a2, &a3);
  transpose4_complex_i16_128(&b0, &b1, &b2, &b3);
  transpose4_complex_i16_128(&c0, &c1, &c2, &c3);

  c16_t t12[4] __attribute__((aligned(16)));
  c16_t t13[4] __attribute__((aligned(16)));
  c16_t t14[4] __attribute__((aligned(16)));
  _mm_store_si128((__m128i *)t12, _mm_load_si128((const __m128i *)(y + 12 * M + k)));
  _mm_store_si128((__m128i *)t13, _mm_load_si128((const __m128i *)(y + 13 * M + k)));
  _mm_store_si128((__m128i *)t14, _mm_load_si128((const __m128i *)(y + 14 * M + k)));

#define R15_STORE_ROW(LANE, A, B, C)                      \
  do {                                                     \
    c16_t *d = dst + 15 * (k + (LANE));                   \
    _mm_storeu_si128((__m128i *)(d + 0), (A));            \
    _mm_storeu_si128((__m128i *)(d + 4), (B));            \
    _mm_storeu_si128((__m128i *)(d + 8), (C));            \
    d[12] = t12[(LANE)];                                  \
    d[13] = t13[(LANE)];                                  \
    d[14] = t14[(LANE)];                                  \
  } while (0)

  R15_STORE_ROW(0, a0, b0, c0);
  R15_STORE_ROW(1, a1, b1, c1);
  R15_STORE_ROW(2, a2, b2, c2);
  R15_STORE_ROW(3, a3, b3, c3);

#undef R15_STORE_ROW
}


static inline void radix15_35_stage_to_branch_major_q15_256(const c16_t *src,
                                                             c16_t *b,
                                                             int N,
                                                             dft_dir_t dir,
                                                             const r3_twiddle_t *twB,
                                                             const r5_twiddle_t *twA)
{
  const int M = N / 15;

  const __m128i *B1r = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_re : twB->r3_q15_w1_re_inv;
  const __m128i *B1i = dir == DFT_DIR_FORWARD ? twB->r3_q15_w1_im : twB->r3_q15_w1_im_inv;
  const __m128i *B2r = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_re : twB->r3_q15_w2_re_inv;
  const __m128i *B2i = dir == DFT_DIR_FORWARD ? twB->r3_q15_w2_im : twB->r3_q15_w2_im_inv;

  const __m128i *A1r = dir == DFT_DIR_FORWARD ? twA->r5_q15_w1_re : twA->r5_q15_w1_re_inv;
  const __m128i *A1i = dir == DFT_DIR_FORWARD ? twA->r5_q15_w1_im : twA->r5_q15_w1_im_inv;
  const __m128i *A2r = dir == DFT_DIR_FORWARD ? twA->r5_q15_w2_re : twA->r5_q15_w2_re_inv;
  const __m128i *A2i = dir == DFT_DIR_FORWARD ? twA->r5_q15_w2_im : twA->r5_q15_w2_im_inv;
  const __m128i *A3r = dir == DFT_DIR_FORWARD ? twA->r5_q15_w3_re : twA->r5_q15_w3_re_inv;
  const __m128i *A3i = dir == DFT_DIR_FORWARD ? twA->r5_q15_w3_im : twA->r5_q15_w3_im_inv;
  const __m128i *A4r = dir == DFT_DIR_FORWARD ? twA->r5_q15_w4_re : twA->r5_q15_w4_re_inv;
  const __m128i *A4i = dir == DFT_DIR_FORWARD ? twA->r5_q15_w4_im : twA->r5_q15_w4_im_inv;

  for (int off = 0; off < M; off += 8) {
    __m256i s[15];

#define R15_35_FIRST_256(AIDX)                                                                    \
    do {                                                                                           \
      const int first_off = (AIDX) * M + off;                                                      \
      const int ti = first_off >> 2;                                                               \
      __m256i z0, z1, z2;                                                                          \
      radix3_butterfly8_q15_256(_mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 0) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 5) * M + off)), \
                                 _mm256_loadu_si256((const __m256i *)(src + ((AIDX) + 10) * M + off)), \
                                 &z0, &z1, &z2, dir);                                               \
      s[(AIDX) * 3 + 0] = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT3));             \
      s[(AIDX) * 3 + 1] = complex_mul8_prepack_q15_256(                                           \
          z1, _mm256_load_si256((const __m256i *)(B1r + ti)), _mm256_load_si256((const __m256i *)(B1i + ti))); \
      s[(AIDX) * 3 + 2] = complex_mul8_prepack_q15_256(                                           \
          z2, _mm256_load_si256((const __m256i *)(B2r + ti)), _mm256_load_si256((const __m256i *)(B2i + ti))); \
    } while (0)

    R15_35_FIRST_256(0);
    R15_35_FIRST_256(1);
    R15_35_FIRST_256(2);
    R15_35_FIRST_256(3);
    R15_35_FIRST_256(4);
#undef R15_35_FIRST_256

    const int ai = off >> 2;
    const __m256i A1rv = _mm256_load_si256((const __m256i *)(A1r + ai));
    const __m256i A1iv = _mm256_load_si256((const __m256i *)(A1i + ai));
    const __m256i A2rv = _mm256_load_si256((const __m256i *)(A2r + ai));
    const __m256i A2iv = _mm256_load_si256((const __m256i *)(A2i + ai));
    const __m256i A3rv = _mm256_load_si256((const __m256i *)(A3r + ai));
    const __m256i A3iv = _mm256_load_si256((const __m256i *)(A3i + ai));
    const __m256i A4rv = _mm256_load_si256((const __m256i *)(A4r + ai));
    const __m256i A4iv = _mm256_load_si256((const __m256i *)(A4i + ai));

#define R15_35_SECOND_256(BIDX)                                                                  \
    do {                                                                                          \
      __m256i z0, z1, z2, z3, z4;                                                               \
      radix5_butterfly8_q15_256(s[(BIDX) + 0], s[(BIDX) + 3], s[(BIDX) + 6],                    \
                                 s[(BIDX) + 9], s[(BIDX) + 12],                                   \
                                 &z0, &z1, &z2, &z3, &z4, dir);                                   \
      z0 = _mm256_mulhrs_epi16(z0, _mm256_set1_epi16(Q15_INV_SQRT5));                           \
      z1 = complex_mul8_prepack_q15_256(z1, A1rv, A1iv);                                        \
      z2 = complex_mul8_prepack_q15_256(z2, A2rv, A2iv);                                        \
      z3 = complex_mul8_prepack_q15_256(z3, A3rv, A3iv);                                        \
      z4 = complex_mul8_prepack_q15_256(z4, A4rv, A4iv);                                        \
      _mm256_store_si256((__m256i *)(b + ((BIDX) + 0 * 3) * M + off), z0);                      \
      _mm256_store_si256((__m256i *)(b + ((BIDX) + 1 * 3) * M + off), z1);                      \
      _mm256_store_si256((__m256i *)(b + ((BIDX) + 2 * 3) * M + off), z2);                      \
      _mm256_store_si256((__m256i *)(b + ((BIDX) + 3 * 3) * M + off), z3);                      \
      _mm256_store_si256((__m256i *)(b + ((BIDX) + 4 * 3) * M + off), z4);                      \
    } while (0)

    R15_35_SECOND_256(0);
    R15_35_SECOND_256(1);
    R15_35_SECOND_256(2);
#undef R15_35_SECOND_256
  }
}

/* Radix-18 helpers for DFT144 and DFT1152 use a Good-Thomas 2 x 9 PFA.
 * 18 = 2 x 9 with gcd(2,9)=1, so there are no twiddles between the
 * radix-2 and radix-9 dimensions.  DFT9 is implemented as two raw radix-3
 * stages with the four required W9 factors.  The complete 1/sqrt(18)
 * normalization is folded into the final W576 parent twiddles.
 *
 * Input CRT mapping:  n = 9*n2 + 10*n9 (mod 18)
 * Output mapping:     k = 9*k2 + 2*k9 (mod 18)
 */


/* Raw, unnormalised DFT9: 9 = 3 x 3 Cooley-Tukey. */


/* DFT1152: true R18 = 2 x 9 PFA, M=64. */

static __attribute__((always_inline)) inline __m256i dft1152_radix9_twiddle_mul(__m256i x,int p,dft_dir_t dir){int16_t wr,wi;switch(p){case 1:wr=25102;wi=-21063;break;case 2:wr=5690;wi=-32270;break;default:wr=-30792;wi=-11207;break;}return complex_mul8_bcast_q15_256_parent(x,wr,twiddle_im_scalar_dir_i16(wi,dir));}
static __attribute__((always_inline)) inline void dft1152_radix9x8(__m256i x[9],__m256i y[9],dft_dir_t dir){__m256i a[3][3];for(int n=0;n<3;n++)radix3_butterfly8_q15_256(x[n],x[n+3],x[n+6],&a[n][0],&a[n][1],&a[n][2],dir);a[1][1]=dft1152_radix9_twiddle_mul(a[1][1],1,dir);a[2][1]=dft1152_radix9_twiddle_mul(a[2][1],2,dir);a[1][2]=dft1152_radix9_twiddle_mul(a[1][2],2,dir);a[2][2]=dft1152_radix9_twiddle_mul(a[2][2],4,dir);for(int k=0;k<3;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+3]=z1;y[k+6]=z2;}}

static pthread_mutex_t dft1152_radix18_twiddle_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft1152_radix18_twiddle_ready;static __m256i dft1152_radix18_twiddle_re[2][8][18] __attribute__((aligned(64))),dft1152_radix18_twiddle_im[2][8][18] __attribute__((aligned(64)));
static void dft1152_radix18_twiddles_init(void){if(__builtin_expect(__atomic_load_n(&dft1152_radix18_twiddle_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft1152_radix18_twiddle_mutex);if(__atomic_load_n(&dft1152_radix18_twiddle_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft1152_radix18_twiddle_mutex);return;}const float sc=1.0f/sqrtf(18.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<8;bl++){int off=8*bl;for(int br=0;br<18;br++){dft1152_radix18_twiddle_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,1152,sc);dft1152_radix18_twiddle_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,1152,sc,dir);}}}__atomic_store_n(&dft1152_radix18_twiddle_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft1152_radix18_twiddle_mutex);}
static inline void dft1152_radix18_stage(const c16_t*src,c16_t*b,dft_dir_t dir){enum{M=64};const int ds=dir==DFT_DIR_FORWARD?0:1;dft1152_radix18_twiddles_init();static const int i0[9]={0,10,2,12,4,14,6,16,8},i1[9]={9,1,11,3,13,5,15,7,17};for(int off=0;off<M;off+=8){__m256i x0[9],x1[9],s0[9],s1[9];for(int j=0;j<9;j++){x0[j]=_mm256_loadu_si256((const __m256i*)(src+i0[j]*M+off));x1[j]=_mm256_loadu_si256((const __m256i*)(src+i1[j]*M+off));}dft1152_radix9x8(x0,s0,dir);dft1152_radix9x8(x1,s1,dir);for(int k=0;k<9;k++){int b0=2*k,b1=(9+2*k)%18;__m256i z0=_mm256_adds_epi16(s0[k],s1[k]),z1=_mm256_subs_epi16(s0[k],s1[k]);__m256i v0=b0==0?_mm256_mulhrs_epi16(z0,_mm256_set1_epi16(Q15_INV_SQRT18)):complex_mul8_prepack_q15_256(z0,dft1152_radix18_twiddle_re[ds][off>>3][b0],dft1152_radix18_twiddle_im[ds][off>>3][b0]);__m256i v1=complex_mul8_prepack_q15_256(z1,dft1152_radix18_twiddle_re[ds][off>>3][b1],dft1152_radix18_twiddle_im[ds][off>>3][b1]);_mm256_store_si256((__m256i*)(b+b0*M+off),v0);_mm256_store_si256((__m256i*)(b+b1*M+off),v1);}}}
static void dft1152_radix18_leaf64_avx2_selected(const c16_t*src,c16_t*dst,dft_dir_t dir){enum{M=64};c16_t b[18*M] __attribute__((aligned(64)));dft1152_radix18_stage(src,b,dir);dft64x8_branch_major_store(b,M,dst,18,0,dir);dft64x8_branch_major_store(b,M,dst,18,8,dir);c16_t tail[64] __attribute__((aligned(64)));for(int br=16;br<18;br++){dft64_avx(b+br*M,tail,dir);for(int k=0;k<M;k++)dst[18*k+br]=tail[k];}}


/* DFT96: true R12 PFA, M=8. */
static __attribute__((always_inline)) inline void dft96_radix4x8_raw(__m256i x0,__m256i x1,__m256i x2,__m256i x3,__m256i y[4],dft_dir_t dir){const __m256i s02=_mm256_adds_epi16(x0,x2),d02=_mm256_subs_epi16(x0,x2),s13=_mm256_adds_epi16(x1,x3),d13=_mm256_subs_epi16(x1,x3);y[0]=_mm256_adds_epi16(s02,s13);y[2]=_mm256_subs_epi16(s02,s13);y[1]=_mm256_adds_epi16(d02,mul_minus_j_dir_i16_256(d13,dir));y[3]=_mm256_adds_epi16(d02,mul_plus_j_dir_i16_256(d13,dir));}
static inline void dft96_radix12_raw(const c16_t*src,int M,int off,__m256i y[12],dft_dir_t dir){static const int ids[3][4]={{0,9,6,3},{4,1,10,7},{8,5,2,11}};__m256i s[3][4];for(int n=0;n<3;n++){__m256i x0=_mm256_loadu_si256((const __m256i*)(src+ids[n][0]*M+off)),x1=_mm256_loadu_si256((const __m256i*)(src+ids[n][1]*M+off)),x2=_mm256_loadu_si256((const __m256i*)(src+ids[n][2]*M+off)),x3=_mm256_loadu_si256((const __m256i*)(src+ids[n][3]*M+off));dft96_radix4x8_raw(x0,x1,x2,x3,s[n],dir);}for(int k=0;k<4;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(s[0][k],s[1][k],s[2][k],&z0,&z1,&z2,dir);y[(3*k)%12]=z0;y[(4+3*k)%12]=z1;y[(8+3*k)%12]=z2;}}

static pthread_mutex_t dft96_radix12_twiddle_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft96_radix12_twiddle_ready;static __m256i dft96_radix12_twiddle_re[2][1][12] __attribute__((aligned(64))),dft96_radix12_twiddle_im[2][1][12] __attribute__((aligned(64)));
static void dft96_radix12_twiddles_init(void){if(__builtin_expect(__atomic_load_n(&dft96_radix12_twiddle_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft96_radix12_twiddle_mutex);if(__atomic_load_n(&dft96_radix12_twiddle_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft96_radix12_twiddle_mutex);return;}const float sc=1.0f/sqrtf((float)12);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<1;bl++){int off=8*bl;for(int br=0;br<12;br++){dft96_radix12_twiddle_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,96,sc);dft96_radix12_twiddle_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,96,sc,dir);}}}__atomic_store_n(&dft96_radix12_twiddle_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft96_radix12_twiddle_mutex);}
static inline void dft96_radix12_stage(const c16_t*src,c16_t*b,dft_dir_t dir){enum{M=8};const int ds=dir==DFT_DIR_FORWARD?0:1;dft96_radix12_twiddles_init();for(int off=0;off<M;off+=8){__m256i z[12];dft96_radix12_raw(src,M,off,z,dir);for(int br=0;br<12;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT12)):complex_mul8_prepack_q15_256(z[br],dft96_radix12_twiddle_re[ds][off>>3][br],dft96_radix12_twiddle_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}
static void dft96_radix12_pfa_leaf8(const c16_t*src,c16_t*dst,dft_dir_t dir){enum{R=12,M=8};c16_t b[R*M] __attribute__((aligned(64)));dft96_radix12_stage(src,b,dir);dft8x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft8x8_twiddle_scaled_store(b,M,dst,R,8,4,dir);}


/* DFT120: true R15 = 3 x 5 Good-Thomas PFA. */

static inline void r15_pfa_raw_block_n120(const c16_t *src,int M,int off,__m256i y[15],dft_dir_t dir){
  static const int ids[3][5]={{0,6,12,3,9},{10,1,7,13,4},{5,11,2,8,14}};__m256i s[3][5];
  for(int n3=0;n3<3;n3++){__m256i x[5];for(int j=0;j<5;j++)x[j]=_mm256_loadu_si256((const __m256i*)(src+ids[n3][j]*M+off));radix5_butterfly8_q15_256(x[0],x[1],x[2],x[3],x[4],&s[n3][0],&s[n3][1],&s[n3][2],&s[n3][3],&s[n3][4],dir);}
  for(int k5=0;k5<5;k5++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(s[0][k5],s[1][k5],s[2][k5],&z0,&z1,&z2,dir);y[(3*k5)%15]=z0;y[(5+3*k5)%15]=z1;y[(10+3*k5)%15]=z2;}
}

static pthread_mutex_t dft120_r15p_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft120_r15p_ready;static __m256i dft120_r15p_re[2][1][15] __attribute__((aligned(64))),dft120_r15p_im[2][1][15] __attribute__((aligned(64)));
static void dft120_r15p_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft120_r15p_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft120_r15p_tw_mutex);if(__atomic_load_n(&dft120_r15p_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft120_r15p_tw_mutex);return;}const float sc=1.0f/sqrtf(15.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<1;bl++){int off=8*bl;for(int br=0;br<15;br++){dft120_r15p_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,120,sc);dft120_r15p_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,120,sc,dir);}}}__atomic_store_n(&dft120_r15p_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft120_r15p_tw_mutex);}
static inline void radix15_pfa_stage_120(const c16_t *src,c16_t*b,dft_dir_t dir){enum{M=8};const int ds=dir==DFT_DIR_FORWARD?0:1;dft120_r15p_tw_init();for(int off=0;off<M;off+=8){__m256i z[15];r15_pfa_raw_block_n120(src,M,off,z,dir);for(int br=0;br<15;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT15)):complex_mul8_prepack_q15_256(z[br],dft120_r15p_re[ds][off>>3][br],dft120_r15p_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}
static void dft120_radix15_pfa_avx2_selected(const c16_t *src,c16_t*dst,dft_dir_t dir){enum{R=15,M=8};c16_t b[R*M] __attribute__((aligned(64)));radix15_pfa_stage_120(src,b,dir);dft8x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft8x8_twiddle_scaled_store(b,M,dst,R,8,7,dir);}


/* DFT144: true R18 = 2 x 9 PFA, M=8. */

static __attribute__((always_inline)) inline __m256i dft144_radix9_twiddle_mul(__m256i x,int p,dft_dir_t dir){int16_t wr,wi;switch(p){case 1:wr=25102;wi=-21063;break;case 2:wr=5690;wi=-32270;break;default:wr=-30792;wi=-11207;break;}return complex_mul8_bcast_q15_256_parent(x,wr,twiddle_im_scalar_dir_i16(wi,dir));}
static __attribute__((always_inline)) inline void dft144_radix9x8(__m256i x[9],__m256i y[9],dft_dir_t dir){__m256i a[3][3];for(int n=0;n<3;n++)radix3_butterfly8_q15_256(x[n],x[n+3],x[n+6],&a[n][0],&a[n][1],&a[n][2],dir);a[1][1]=dft144_radix9_twiddle_mul(a[1][1],1,dir);a[2][1]=dft144_radix9_twiddle_mul(a[2][1],2,dir);a[1][2]=dft144_radix9_twiddle_mul(a[1][2],2,dir);a[2][2]=dft144_radix9_twiddle_mul(a[2][2],4,dir);for(int k=0;k<3;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+3]=z1;y[k+6]=z2;}}

static pthread_mutex_t dft144_radix18_twiddle_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft144_radix18_twiddle_ready;static __m256i dft144_radix18_twiddle_re[2][1][18] __attribute__((aligned(64))),dft144_radix18_twiddle_im[2][1][18] __attribute__((aligned(64)));
static void dft144_radix18_twiddles_init(void){if(__builtin_expect(__atomic_load_n(&dft144_radix18_twiddle_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft144_radix18_twiddle_mutex);if(__atomic_load_n(&dft144_radix18_twiddle_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft144_radix18_twiddle_mutex);return;}const float sc=1.0f/sqrtf(18.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<1;bl++){int off=8*bl;for(int br=0;br<18;br++){dft144_radix18_twiddle_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,144,sc);dft144_radix18_twiddle_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,144,sc,dir);}}}__atomic_store_n(&dft144_radix18_twiddle_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft144_radix18_twiddle_mutex);}
static inline void dft144_radix18_stage(const c16_t*src,c16_t*b,dft_dir_t dir){enum{M=8};const int ds=dir==DFT_DIR_FORWARD?0:1;dft144_radix18_twiddles_init();static const int i0[9]={0,10,2,12,4,14,6,16,8},i1[9]={9,1,11,3,13,5,15,7,17};for(int off=0;off<M;off+=8){__m256i x0[9],x1[9],s0[9],s1[9];for(int j=0;j<9;j++){x0[j]=_mm256_loadu_si256((const __m256i*)(src+i0[j]*M+off));x1[j]=_mm256_loadu_si256((const __m256i*)(src+i1[j]*M+off));}dft144_radix9x8(x0,s0,dir);dft144_radix9x8(x1,s1,dir);for(int k=0;k<9;k++){int b0=2*k,b1=(9+2*k)%18;__m256i z0=_mm256_adds_epi16(s0[k],s1[k]),z1=_mm256_subs_epi16(s0[k],s1[k]);__m256i v0=b0==0?_mm256_mulhrs_epi16(z0,_mm256_set1_epi16(Q15_INV_SQRT18)):complex_mul8_prepack_q15_256(z0,dft144_radix18_twiddle_re[ds][off>>3][b0],dft144_radix18_twiddle_im[ds][off>>3][b0]);__m256i v1=complex_mul8_prepack_q15_256(z1,dft144_radix18_twiddle_re[ds][off>>3][b1],dft144_radix18_twiddle_im[ds][off>>3][b1]);_mm256_store_si256((__m256i*)(b+b0*M+off),v0);_mm256_store_si256((__m256i*)(b+b1*M+off),v1);}}}
static void dft144_radix18_leaf8_avx2_selected(const c16_t*src,c16_t*dst,dft_dir_t dir){enum{R=18,M=8};c16_t b[R*M] __attribute__((aligned(64)));dft144_radix18_stage(src,b,dir);dft8x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft8x8_twiddle_scaled_store(b,M,dst,R,8,8,dir);dft8x8_twiddle_scaled_store(b,M,dst,R,16,2,dir);}

/* DFT192: true R12 PFA, M=16. */
static __attribute__((always_inline)) inline void dft192_radix4x8_raw(__m256i x0,__m256i x1,__m256i x2,__m256i x3,__m256i y[4],dft_dir_t dir){const __m256i s02=_mm256_adds_epi16(x0,x2),d02=_mm256_subs_epi16(x0,x2),s13=_mm256_adds_epi16(x1,x3),d13=_mm256_subs_epi16(x1,x3);y[0]=_mm256_adds_epi16(s02,s13);y[2]=_mm256_subs_epi16(s02,s13);y[1]=_mm256_adds_epi16(d02,mul_minus_j_dir_i16_256(d13,dir));y[3]=_mm256_adds_epi16(d02,mul_plus_j_dir_i16_256(d13,dir));}
static inline void dft192_radix12_raw(const c16_t*src,int M,int off,__m256i y[12],dft_dir_t dir){static const int ids[3][4]={{0,9,6,3},{4,1,10,7},{8,5,2,11}};__m256i s[3][4];for(int n=0;n<3;n++){__m256i x0=_mm256_loadu_si256((const __m256i*)(src+ids[n][0]*M+off)),x1=_mm256_loadu_si256((const __m256i*)(src+ids[n][1]*M+off)),x2=_mm256_loadu_si256((const __m256i*)(src+ids[n][2]*M+off)),x3=_mm256_loadu_si256((const __m256i*)(src+ids[n][3]*M+off));dft192_radix4x8_raw(x0,x1,x2,x3,s[n],dir);}for(int k=0;k<4;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(s[0][k],s[1][k],s[2][k],&z0,&z1,&z2,dir);y[(3*k)%12]=z0;y[(4+3*k)%12]=z1;y[(8+3*k)%12]=z2;}}

static pthread_mutex_t dft192_radix12_twiddle_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft192_radix12_twiddle_ready;static __m256i dft192_radix12_twiddle_re[2][2][12] __attribute__((aligned(64))),dft192_radix12_twiddle_im[2][2][12] __attribute__((aligned(64)));
static void dft192_radix12_twiddles_init(void){if(__builtin_expect(__atomic_load_n(&dft192_radix12_twiddle_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft192_radix12_twiddle_mutex);if(__atomic_load_n(&dft192_radix12_twiddle_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft192_radix12_twiddle_mutex);return;}const float sc=1.0f/sqrtf((float)12);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<2;bl++){int off=8*bl;for(int br=0;br<12;br++){dft192_radix12_twiddle_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,192,sc);dft192_radix12_twiddle_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,192,sc,dir);}}}__atomic_store_n(&dft192_radix12_twiddle_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft192_radix12_twiddle_mutex);}
static inline void dft192_radix12_stage(const c16_t*src,c16_t*b,dft_dir_t dir){enum{M=16};const int ds=dir==DFT_DIR_FORWARD?0:1;dft192_radix12_twiddles_init();for(int off=0;off<M;off+=8){__m256i z[12];dft192_radix12_raw(src,M,off,z,dir);for(int br=0;br<12;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT12)):complex_mul8_prepack_q15_256(z[br],dft192_radix12_twiddle_re[ds][off>>3][br],dft192_radix12_twiddle_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}
static void dft192_radix12_pfa_leaf16(const c16_t*src,c16_t*dst,dft_dir_t dir){enum{R=12,M=16};c16_t b[R*M] __attribute__((aligned(64)));dft192_radix12_stage(src,b,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,8,4,dir);}

/* DFT216: R27 x DFT8.  The R27 parent is an unscaled 3^3
 * AVX2 kernel; 1/sqrt(27) is folded into the final W216 parent twiddles. */
static __attribute__((always_inline)) inline __m256i r27_216_w9_mul(__m256i x,int p,dft_dir_t dir){int16_t wr,wi;switch(p){case 1:wr=25102;wi=-21063;break;case 2:wr=5690;wi=-32270;break;default:wr=-30792;wi=-11207;break;}return complex_mul8_bcast_q15_256_parent(x,wr,twiddle_im_scalar_dir_i16(wi,dir));}
static __attribute__((always_inline)) inline void r27_216_dft9_raw(__m256i x[9],__m256i y[9],dft_dir_t dir){__m256i a[3][3];for(int n=0;n<3;n++)radix3_butterfly8_q15_256(x[n],x[n+3],x[n+6],&a[n][0],&a[n][1],&a[n][2],dir);a[1][1]=r27_216_w9_mul(a[1][1],1,dir);a[2][1]=r27_216_w9_mul(a[2][1],2,dir);a[1][2]=r27_216_w9_mul(a[1][2],2,dir);a[2][2]=r27_216_w9_mul(a[2][2],4,dir);for(int k=0;k<3;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+3]=z1;y[k+6]=z2;}}
static __attribute__((always_inline)) inline __m256i r27_216_internal_tw(__m256i x,int n1,int k,dft_dir_t dir){static const int16_t wr[2][9]={{32767,31885,29283,25102,19568,12979,5690,-1905,-9398},{32767,29283,19568,5690,-9398,-22487,-30792,-32546,-27377}};static const int16_t wi[2][9]={{0,-7557,-14706,-21063,-26284,-30088,-32270,-32713,-31391},{0,-14706,-26284,-32270,-31391,-23835,-11207,3804,18006}};return complex_mul8_bcast_q15_256_parent(x,wr[n1-1][k],twiddle_im_scalar_dir_i16(wi[n1-1][k],dir));}
static __attribute__((always_inline)) inline void r27_216_raw8(const c16_t *src,int M,int off,__m256i y[27],dft_dir_t dir){__m256i a[3][9];for(int n1=0;n1<3;n1++){__m256i x[9];for(int n9=0;n9<9;n9++)x[n9]=_mm256_loadu_si256((const __m256i*)(src+(n1+3*n9)*M+off));r27_216_dft9_raw(x,a[n1],dir);}for(int n1=1;n1<3;n1++)for(int k=1;k<9;k++)a[n1][k]=r27_216_internal_tw(a[n1][k],n1,k,dir);for(int k=0;k<9;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+9]=z1;y[k+18]=z2;}}

static pthread_mutex_t dft216_r27_tw_mutex=PTHREAD_MUTEX_INITIALIZER;
static int dft216_r27_ready;
static __m256i dft216_r27_re[2][1][27] __attribute__((aligned(64))),dft216_r27_im[2][1][27] __attribute__((aligned(64)));
static void dft216_r27_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft216_r27_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft216_r27_tw_mutex);if(__atomic_load_n(&dft216_r27_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft216_r27_tw_mutex);return;}const float sc=1.0f/sqrtf(27.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int br=0;br<27;br++){dft216_r27_re[ds][0][br]=pack8_twiddle_q15_re_re_scaled(0,br,216,sc);dft216_r27_im[ds][0][br]=pack8_twiddle_q15_im_signed_scaled(0,br,216,sc,dir);}}__atomic_store_n(&dft216_r27_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft216_r27_tw_mutex);}
static inline void radix27_stage_216(const c16_t *src,c16_t *b,dft_dir_t dir){enum{M=8};const int ds=dir==DFT_DIR_FORWARD?0:1;dft216_r27_tw_init();__m256i z[27];r27_216_raw8(src,M,0,z,dir);for(int br=0;br<27;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT27)):complex_mul8_prepack_q15_256(z[br],dft216_r27_re[ds][0][br],dft216_r27_im[ds][0][br]);_mm256_store_si256((__m256i*)(b+br*M),v);}}
static void dft216_radix27_leaf8_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir){enum{R=27,M=8};c16_t b[R*M] __attribute__((aligned(64)));radix27_stage_216(src,b,dir);for(int first=0;first<R;first+=8){const int valid=(R-first)<8?(R-first):8;dft8x8_twiddle_scaled_store(b,M,dst,R,first,valid,dir);}}

/* DFT240: true R15 = 3 x 5 Good-Thomas PFA. */

static inline void r15_pfa_raw_block_n240(const c16_t *src,int M,int off,__m256i y[15],dft_dir_t dir){
  static const int ids[3][5]={{0,6,12,3,9},{10,1,7,13,4},{5,11,2,8,14}};__m256i s[3][5];
  for(int n3=0;n3<3;n3++){__m256i x[5];for(int j=0;j<5;j++)x[j]=_mm256_loadu_si256((const __m256i*)(src+ids[n3][j]*M+off));radix5_butterfly8_q15_256(x[0],x[1],x[2],x[3],x[4],&s[n3][0],&s[n3][1],&s[n3][2],&s[n3][3],&s[n3][4],dir);}
  for(int k5=0;k5<5;k5++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(s[0][k5],s[1][k5],s[2][k5],&z0,&z1,&z2,dir);y[(3*k5)%15]=z0;y[(5+3*k5)%15]=z1;y[(10+3*k5)%15]=z2;}
}

static pthread_mutex_t dft240_r15p_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft240_r15p_ready;static __m256i dft240_r15p_re[2][2][15] __attribute__((aligned(64))),dft240_r15p_im[2][2][15] __attribute__((aligned(64)));
static void dft240_r15p_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft240_r15p_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft240_r15p_tw_mutex);if(__atomic_load_n(&dft240_r15p_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft240_r15p_tw_mutex);return;}const float sc=1.0f/sqrtf(15.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<2;bl++){int off=8*bl;for(int br=0;br<15;br++){dft240_r15p_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,240,sc);dft240_r15p_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,240,sc,dir);}}}__atomic_store_n(&dft240_r15p_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft240_r15p_tw_mutex);}
static inline void radix15_pfa_stage_240(const c16_t *src,c16_t*b,dft_dir_t dir){enum{M=16};const int ds=dir==DFT_DIR_FORWARD?0:1;dft240_r15p_tw_init();for(int off=0;off<M;off+=8){__m256i z[15];r15_pfa_raw_block_n240(src,M,off,z,dir);for(int br=0;br<15;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT15)):complex_mul8_prepack_q15_256(z[br],dft240_r15p_re[ds][off>>3][br],dft240_r15p_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}
static void dft240_radix15_pfa_avx2_selected(const c16_t *src,c16_t*dst,dft_dir_t dir){enum{R=15,M=16};c16_t b[R*M] __attribute__((aligned(64)));radix15_pfa_stage_240(src,b,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,8,7,dir);}

/* DFT360: R30 PFA parent (1/sqrt30 in W360) + twiddle-scaled DFT12 leaves (1/sqrt12). */

static __attribute__((always_inline)) inline __m256i load_r30_block8_n360(const c16_t *p,int valid){if(valid==8)return _mm256_loadu_si256((const __m256i*)p);return _mm256_inserti128_si256(_mm256_setzero_si256(),_mm_loadu_si128((const __m128i*)p),0);}
static __attribute__((always_inline)) inline void store_r30_block8_n360(c16_t *p,__m256i v,int valid){if(valid==8)_mm256_store_si256((__m256i*)p,v);else _mm_store_si128((__m128i*)p,_mm256_castsi256_si128(v));}
static __attribute__((always_inline)) inline __m256i gather8_leaf_c16_n360(const c16_t *base,int M,int k){const __m256i idx=_mm256_setr_epi32(0,M,2*M,3*M,4*M,5*M,6*M,7*M);return _mm256_i32gather_epi32((const int*)(base+k),idx,4);}
static __attribute__((always_inline)) inline void store8_leaf_row_n360(c16_t *dst,int R,int k,int first,int valid,__m256i y){c16_t*d=dst+R*k+first;if(valid==8)_mm256_storeu_si256((__m256i*)d,y);else{_mm_storeu_si128((__m128i*)d,_mm256_castsi256_si128(y));_mm_storel_epi64((__m128i*)(d+4),_mm256_extracti128_si256(y,1));}}
static __attribute__((always_inline)) inline void raw_dft4_lane8_n360(__m256i x0,__m256i x1,__m256i x2,__m256i x3,__m256i y[4],dft_dir_t dir){const __m256i s02=_mm256_adds_epi16(x0,x2),d02=_mm256_subs_epi16(x0,x2),s13=_mm256_adds_epi16(x1,x3),d13=_mm256_subs_epi16(x1,x3);y[0]=_mm256_adds_epi16(s02,s13);y[2]=_mm256_subs_epi16(s02,s13);y[1]=_mm256_adds_epi16(d02,mul_minus_j_dir_i16_256(d13,dir));y[3]=_mm256_adds_epi16(d02,mul_plus_j_dir_i16_256(d13,dir));}


static pthread_mutex_t dft360_r30_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft360_r30_ready;static __m256i dft360_r30_re[2][2][30] __attribute__((aligned(64))),dft360_r30_im[2][2][30] __attribute__((aligned(64)));
static void dft360_r30_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft360_r30_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft360_r30_tw_mutex);if(__atomic_load_n(&dft360_r30_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft360_r30_tw_mutex);return;}const float sc=1.0f/sqrtf(30.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<2;bl++){int off=8*bl;for(int br=0;br<30;br++){dft360_r30_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,360,sc);dft360_r30_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,360,sc,dir);}}}__atomic_store_n(&dft360_r30_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft360_r30_tw_mutex);}
static inline void radix30_stage_360(const c16_t *src,c16_t*b,dft_dir_t dir){enum{M=12};const int ds=dir==DFT_DIR_FORWARD?0:1;dft360_r30_tw_init();for(int off=0;off<M;off+=8){const int valid=(M-off>=8)?8:4;__m256i s[5][6];
#define DN(C,I0,I1,I2,I3,I4,I5) dft6_pfa8_q15_256(load_r30_block8_n360(src+(I0)*M+off,valid),load_r30_block8_n360(src+(I1)*M+off,valid),load_r30_block8_n360(src+(I2)*M+off,valid),load_r30_block8_n360(src+(I3)*M+off,valid),load_r30_block8_n360(src+(I4)*M+off,valid),load_r30_block8_n360(src+(I5)*M+off,valid),s[C],dir)
 DN(0,0,25,20,15,10,5);DN(1,6,1,26,21,16,11);DN(2,12,7,2,27,22,17);DN(3,18,13,8,3,28,23);DN(4,24,19,14,9,4,29);
#undef DN
#define SN(BR,Z) do{__m256i v=(BR)==0?_mm256_mulhrs_epi16((Z),_mm256_set1_epi16(Q15_INV_SQRT30)):complex_mul8_prepack_q15_256((Z),dft360_r30_re[ds][off>>3][BR],dft360_r30_im[ds][off>>3][BR]);store_r30_block8_n360(b+(BR)*M+off,v,valid);}while(0)
#define FN(K,B0,B1,B2,B3,B4) do{__m256i z0,z1,z2,z3,z4;radix5_butterfly8_q15_256(s[0][K],s[1][K],s[2][K],s[3][K],s[4][K],&z0,&z1,&z2,&z3,&z4,dir);SN(B0,z0);SN(B1,z1);SN(B2,z2);SN(B3,z3);SN(B4,z4);}while(0)
 FN(0,0,6,12,18,24);FN(1,5,11,17,23,29);FN(2,10,16,22,28,4);FN(3,15,21,27,3,9);FN(4,20,26,2,8,14);FN(5,25,1,7,13,19);
#undef FN
#undef SN
}}

/* DFT384: true R24 = 3 x 8 Good-Thomas PFA, M=16. */
static pthread_mutex_t dft384_r24_twiddle_mutex=PTHREAD_MUTEX_INITIALIZER; static int dft384_r24_twiddles_ready;
static __m256i dft384_r24_re[2][2][24] __attribute__((aligned(64))); static __m256i dft384_r24_im[2][2][24] __attribute__((aligned(64)));
static void dft384_r24_twiddles_init(void){ if(__builtin_expect(__atomic_load_n(&dft384_r24_twiddles_ready,__ATOMIC_ACQUIRE),1))return; pthread_mutex_lock(&dft384_r24_twiddle_mutex);
 if(__atomic_load_n(&dft384_r24_twiddles_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft384_r24_twiddle_mutex);return;} const float sc=1.0f/sqrtf(24.0f);
 for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<2;bl++){int off=8*bl;for(int br=0;br<24;br++){dft384_r24_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,384,sc);dft384_r24_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,384,sc,dir);}}}
 __atomic_store_n(&dft384_r24_twiddles_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft384_r24_twiddle_mutex);}
static inline void radix24_pfa_stage_384(const c16_t *src,c16_t *b,dft_dir_t dir){enum{M=16};const int ds=dir==DFT_DIR_FORWARD?0:1;dft384_r24_twiddles_init();
 static const int ids[3][8]={{0,9,18,3,12,21,6,15},{16,1,10,19,4,13,22,7},{8,17,2,11,20,5,14,23}};
 for(int off=0;off<M;off+=8){__m256i s[3][8];for(int n3=0;n3<3;n3++){__m256i x[8];for(int j=0;j<8;j++)x[j]=_mm256_loadu_si256((const __m256i*)(src+ids[n3][j]*M+off));
 dft8x8_q15_256_dir(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],&s[n3][0],&s[n3][1],&s[n3][2],&s[n3][3],&s[n3][4],&s[n3][5],&s[n3][6],&s[n3][7],dir);}
 for(int k8=0;k8<8;k8++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(s[0][k8],s[1][k8],s[2][k8],&z0,&z1,&z2,dir);const int brs[3]={(3*k8)%24,(8+3*k8)%24,(16+3*k8)%24};const __m256i zs[3]={z0,z1,z2};
 for(int j=0;j<3;j++){int br=brs[j];__m256i v=br==0?_mm256_mulhrs_epi16(zs[j],_mm256_set1_epi16(Q15_INV_SQRT24)):complex_mul8_prepack_q15_256(zs[j],dft384_r24_re[ds][off>>3][br],dft384_r24_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}}
static void dft384_radix24_pfa_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir){enum{R=24,M=16};c16_t b[R*M] __attribute__((aligned(64)));radix24_pfa_stage_384(src,b,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,8,8,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,16,8,dir);}

/* DFT432: R27 x DFT16. */

static __attribute__((always_inline)) inline __m256i r27_w9_mul_n432(__m256i x,int p,dft_dir_t dir){int16_t wr,wi;switch(p){case 1:wr=25102;wi=-21063;break;case 2:wr=5690;wi=-32270;break;default:wr=-30792;wi=-11207;break;}return complex_mul8_bcast_q15_256_parent(x,wr,twiddle_im_scalar_dir_i16(wi,dir));}
static __attribute__((always_inline)) inline void r27_dft9_raw_n432(__m256i x[9],__m256i y[9],dft_dir_t dir){__m256i a[3][3];for(int n=0;n<3;n++)radix3_butterfly8_q15_256(x[n],x[n+3],x[n+6],&a[n][0],&a[n][1],&a[n][2],dir);a[1][1]=r27_w9_mul_n432(a[1][1],1,dir);a[2][1]=r27_w9_mul_n432(a[2][1],2,dir);a[1][2]=r27_w9_mul_n432(a[1][2],2,dir);a[2][2]=r27_w9_mul_n432(a[2][2],4,dir);for(int k=0;k<3;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+3]=z1;y[k+6]=z2;}}
static __attribute__((always_inline)) inline __m256i r27_internal_tw_n432(__m256i x,int n1,int k,dft_dir_t dir){static const int16_t wr[2][9]={{32767,31885,29283,25102,19568,12979,5690,-1905,-9398},{32767,29283,19568,5690,-9398,-22487,-30792,-32546,-27377}};static const int16_t wi[2][9]={{0,-7557,-14706,-21063,-26284,-30088,-32270,-32713,-31391},{0,-14706,-26284,-32270,-31391,-23835,-11207,3804,18006}};return complex_mul8_bcast_q15_256_parent(x,wr[n1-1][k],twiddle_im_scalar_dir_i16(wi[n1-1][k],dir));}
static __attribute__((always_inline)) inline void r27_raw8_n432(const c16_t *src,int M,int off,__m256i y[27],dft_dir_t dir){__m256i a[3][9];for(int n1=0;n1<3;n1++){__m256i x[9];for(int n9=0;n9<9;n9++)x[n9]=_mm256_loadu_si256((const __m256i*)(src+(n1+3*n9)*M+off));r27_dft9_raw_n432(x,a[n1],dir);}for(int n1=1;n1<3;n1++)for(int k=1;k<9;k++)a[n1][k]=r27_internal_tw_n432(a[n1][k],n1,k,dir);for(int k=0;k<9;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+9]=z1;y[k+18]=z2;}}

static pthread_mutex_t dft432_r27_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft432_r27_ready;static __m256i dft432_r27_re[2][2][27] __attribute__((aligned(64))),dft432_r27_im[2][2][27] __attribute__((aligned(64)));
static void dft432_r27_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft432_r27_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft432_r27_tw_mutex);if(__atomic_load_n(&dft432_r27_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft432_r27_tw_mutex);return;}const float sc=1.0f/sqrtf(27.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<2;bl++){int off=8*bl;for(int br=0;br<27;br++){dft432_r27_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,432,sc);dft432_r27_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,432,sc,dir);}}}__atomic_store_n(&dft432_r27_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft432_r27_tw_mutex);}
static inline void radix27_stage_432(const c16_t *src,c16_t *b,dft_dir_t dir){enum{M=16};const int ds=dir==DFT_DIR_FORWARD?0:1;dft432_r27_tw_init();for(int off=0;off<M;off+=8){__m256i z[27];r27_raw8_n432(src,M,off,z,dir);for(int br=0;br<27;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT27)):complex_mul8_prepack_q15_256(z[br],dft432_r27_re[ds][off>>3][br],dft432_r27_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}
static void dft432_radix27_leaf16_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir){enum{R=27,M=16};c16_t b[R*M] __attribute__((aligned(64)));radix27_stage_432(src,b,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,8,8,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,16,8,dir);dft16x8_twiddle_scaled_store(b,M,dst,R,24,3,dir);}

/* DFT600: R30 PFA parent (1/sqrt30 in W600) + twiddle-scaled DFT20 leaves (1/sqrt20). */

static __attribute__((always_inline)) inline __m256i load_r30_block8_n600(const c16_t *p,int valid){if(valid==8)return _mm256_loadu_si256((const __m256i*)p);return _mm256_inserti128_si256(_mm256_setzero_si256(),_mm_loadu_si128((const __m128i*)p),0);}
static __attribute__((always_inline)) inline void store_r30_block8_n600(c16_t *p,__m256i v,int valid){if(valid==8)_mm256_store_si256((__m256i*)p,v);else _mm_store_si128((__m128i*)p,_mm256_castsi256_si128(v));}
static __attribute__((always_inline)) inline __m256i gather8_leaf_c16_n600(const c16_t *base,int M,int k){const __m256i idx=_mm256_setr_epi32(0,M,2*M,3*M,4*M,5*M,6*M,7*M);return _mm256_i32gather_epi32((const int*)(base+k),idx,4);}
static __attribute__((always_inline)) inline void store8_leaf_row_n600(c16_t *dst,int R,int k,int first,int valid,__m256i y){c16_t*d=dst+R*k+first;if(valid==8)_mm256_storeu_si256((__m256i*)d,y);else{_mm_storeu_si128((__m128i*)d,_mm256_castsi256_si128(y));_mm_storel_epi64((__m128i*)(d+4),_mm256_extracti128_si256(y,1));}}
static __attribute__((always_inline)) inline void raw_dft4_lane8_n600(__m256i x0,__m256i x1,__m256i x2,__m256i x3,__m256i y[4],dft_dir_t dir){const __m256i s02=_mm256_adds_epi16(x0,x2),d02=_mm256_subs_epi16(x0,x2),s13=_mm256_adds_epi16(x1,x3),d13=_mm256_subs_epi16(x1,x3);y[0]=_mm256_adds_epi16(s02,s13);y[2]=_mm256_subs_epi16(s02,s13);y[1]=_mm256_adds_epi16(d02,mul_minus_j_dir_i16_256(d13,dir));y[3]=_mm256_adds_epi16(d02,mul_plus_j_dir_i16_256(d13,dir));}


static pthread_mutex_t dft600_r30_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft600_r30_ready;static __m256i dft600_r30_re[2][3][30] __attribute__((aligned(64))),dft600_r30_im[2][3][30] __attribute__((aligned(64)));
static void dft600_r30_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft600_r30_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft600_r30_tw_mutex);if(__atomic_load_n(&dft600_r30_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft600_r30_tw_mutex);return;}const float sc=1.0f/sqrtf(30.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<3;bl++){int off=8*bl;for(int br=0;br<30;br++){dft600_r30_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,600,sc);dft600_r30_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,600,sc,dir);}}}__atomic_store_n(&dft600_r30_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft600_r30_tw_mutex);}
static inline void radix30_stage_600(const c16_t *src,c16_t*b,dft_dir_t dir){enum{M=20};const int ds=dir==DFT_DIR_FORWARD?0:1;dft600_r30_tw_init();for(int off=0;off<M;off+=8){const int valid=(M-off>=8)?8:4;__m256i s[5][6];
#define DN(C,I0,I1,I2,I3,I4,I5) dft6_pfa8_q15_256(load_r30_block8_n600(src+(I0)*M+off,valid),load_r30_block8_n600(src+(I1)*M+off,valid),load_r30_block8_n600(src+(I2)*M+off,valid),load_r30_block8_n600(src+(I3)*M+off,valid),load_r30_block8_n600(src+(I4)*M+off,valid),load_r30_block8_n600(src+(I5)*M+off,valid),s[C],dir)
 DN(0,0,25,20,15,10,5);DN(1,6,1,26,21,16,11);DN(2,12,7,2,27,22,17);DN(3,18,13,8,3,28,23);DN(4,24,19,14,9,4,29);
#undef DN
#define SN(BR,Z) do{__m256i v=(BR)==0?_mm256_mulhrs_epi16((Z),_mm256_set1_epi16(Q15_INV_SQRT30)):complex_mul8_prepack_q15_256((Z),dft600_r30_re[ds][off>>3][BR],dft600_r30_im[ds][off>>3][BR]);store_r30_block8_n600(b+(BR)*M+off,v,valid);}while(0)
#define FN(K,B0,B1,B2,B3,B4) do{__m256i z0,z1,z2,z3,z4;radix5_butterfly8_q15_256(s[0][K],s[1][K],s[2][K],s[3][K],s[4][K],&z0,&z1,&z2,&z3,&z4,dir);SN(B0,z0);SN(B1,z1);SN(B2,z2);SN(B3,z3);SN(B4,z4);}while(0)
 FN(0,0,6,12,18,24);FN(1,5,11,17,23,29);FN(2,10,16,22,28,4);FN(3,15,21,27,3,9);FN(4,20,26,2,8,14);FN(5,25,1,7,13,19);
#undef FN
#undef SN
}}

/* DFT720: R30 PFA parent (1/sqrt30 in W720) + twiddle-scaled DFT24 leaves (1/sqrt24). */

static __attribute__((always_inline)) inline __m256i load_r30_block8_n720(const c16_t *p,int valid){if(valid==8)return _mm256_loadu_si256((const __m256i*)p);return _mm256_inserti128_si256(_mm256_setzero_si256(),_mm_loadu_si128((const __m128i*)p),0);}
static __attribute__((always_inline)) inline void store_r30_block8_n720(c16_t *p,__m256i v,int valid){if(valid==8)_mm256_store_si256((__m256i*)p,v);else _mm_store_si128((__m128i*)p,_mm256_castsi256_si128(v));}
static __attribute__((always_inline)) inline __m256i gather8_leaf_c16_n720(const c16_t *base,int M,int k){const __m256i idx=_mm256_setr_epi32(0,M,2*M,3*M,4*M,5*M,6*M,7*M);return _mm256_i32gather_epi32((const int*)(base+k),idx,4);}
static __attribute__((always_inline)) inline void store8_leaf_row_n720(c16_t *dst,int R,int k,int first,int valid,__m256i y){c16_t*d=dst+R*k+first;if(valid==8)_mm256_storeu_si256((__m256i*)d,y);else{_mm_storeu_si128((__m128i*)d,_mm256_castsi256_si128(y));_mm_storel_epi64((__m128i*)(d+4),_mm256_extracti128_si256(y,1));}}


static pthread_mutex_t dft720_r30_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft720_r30_ready;static __m256i dft720_r30_re[2][3][30] __attribute__((aligned(64))),dft720_r30_im[2][3][30] __attribute__((aligned(64)));
static void dft720_r30_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft720_r30_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft720_r30_tw_mutex);if(__atomic_load_n(&dft720_r30_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft720_r30_tw_mutex);return;}const float sc=1.0f/sqrtf(30.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<3;bl++){int off=8*bl;for(int br=0;br<30;br++){dft720_r30_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,720,sc);dft720_r30_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,720,sc,dir);}}}__atomic_store_n(&dft720_r30_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft720_r30_tw_mutex);}
static inline void radix30_stage_720(const c16_t *src,c16_t*b,dft_dir_t dir){enum{M=24};const int ds=dir==DFT_DIR_FORWARD?0:1;dft720_r30_tw_init();for(int off=0;off<M;off+=8){const int valid=(M-off>=8)?8:4;__m256i s[5][6];
#define DN(C,I0,I1,I2,I3,I4,I5) dft6_pfa8_q15_256(load_r30_block8_n720(src+(I0)*M+off,valid),load_r30_block8_n720(src+(I1)*M+off,valid),load_r30_block8_n720(src+(I2)*M+off,valid),load_r30_block8_n720(src+(I3)*M+off,valid),load_r30_block8_n720(src+(I4)*M+off,valid),load_r30_block8_n720(src+(I5)*M+off,valid),s[C],dir)
 DN(0,0,25,20,15,10,5);DN(1,6,1,26,21,16,11);DN(2,12,7,2,27,22,17);DN(3,18,13,8,3,28,23);DN(4,24,19,14,9,4,29);
#undef DN
#define SN(BR,Z) do{__m256i v=(BR)==0?_mm256_mulhrs_epi16((Z),_mm256_set1_epi16(Q15_INV_SQRT30)):complex_mul8_prepack_q15_256((Z),dft720_r30_re[ds][off>>3][BR],dft720_r30_im[ds][off>>3][BR]);store_r30_block8_n720(b+(BR)*M+off,v,valid);}while(0)
#define FN(K,B0,B1,B2,B3,B4) do{__m256i z0,z1,z2,z3,z4;radix5_butterfly8_q15_256(s[0][K],s[1][K],s[2][K],s[3][K],s[4][K],&z0,&z1,&z2,&z3,&z4,dir);SN(B0,z0);SN(B1,z1);SN(B2,z2);SN(B3,z3);SN(B4,z4);}while(0)
 FN(0,0,6,12,18,24);FN(1,5,11,17,23,29);FN(2,10,16,22,28,4);FN(3,15,21,27,3,9);FN(4,20,26,2,8,14);FN(5,25,1,7,13,19);
#undef FN
#undef SN
}}

/* DFT768: true R24 PFA, M=32. */


/* DFT864: R27 x DFT32. */

static __attribute__((always_inline)) inline __m256i r27_w9_mul_n864(__m256i x,int p,dft_dir_t dir){int16_t wr,wi;switch(p){case 1:wr=25102;wi=-21063;break;case 2:wr=5690;wi=-32270;break;default:wr=-30792;wi=-11207;break;}return complex_mul8_bcast_q15_256_parent(x,wr,twiddle_im_scalar_dir_i16(wi,dir));}
static __attribute__((always_inline)) inline void r27_dft9_raw_n864(__m256i x[9],__m256i y[9],dft_dir_t dir){__m256i a[3][3];for(int n=0;n<3;n++)radix3_butterfly8_q15_256(x[n],x[n+3],x[n+6],&a[n][0],&a[n][1],&a[n][2],dir);a[1][1]=r27_w9_mul_n864(a[1][1],1,dir);a[2][1]=r27_w9_mul_n864(a[2][1],2,dir);a[1][2]=r27_w9_mul_n864(a[1][2],2,dir);a[2][2]=r27_w9_mul_n864(a[2][2],4,dir);for(int k=0;k<3;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+3]=z1;y[k+6]=z2;}}
static __attribute__((always_inline)) inline __m256i r27_internal_tw_n864(__m256i x,int n1,int k,dft_dir_t dir){static const int16_t wr[2][9]={{32767,31885,29283,25102,19568,12979,5690,-1905,-9398},{32767,29283,19568,5690,-9398,-22487,-30792,-32546,-27377}};static const int16_t wi[2][9]={{0,-7557,-14706,-21063,-26284,-30088,-32270,-32713,-31391},{0,-14706,-26284,-32270,-31391,-23835,-11207,3804,18006}};return complex_mul8_bcast_q15_256_parent(x,wr[n1-1][k],twiddle_im_scalar_dir_i16(wi[n1-1][k],dir));}
static __attribute__((always_inline)) inline void r27_raw8_n864(const c16_t *src,int M,int off,__m256i y[27],dft_dir_t dir){__m256i a[3][9];for(int n1=0;n1<3;n1++){__m256i x[9];for(int n9=0;n9<9;n9++)x[n9]=_mm256_loadu_si256((const __m256i*)(src+(n1+3*n9)*M+off));r27_dft9_raw_n864(x,a[n1],dir);}for(int n1=1;n1<3;n1++)for(int k=1;k<9;k++)a[n1][k]=r27_internal_tw_n864(a[n1][k],n1,k,dir);for(int k=0;k<9;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+9]=z1;y[k+18]=z2;}}

static pthread_mutex_t dft864_r27_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft864_r27_ready;static __m256i dft864_r27_re[2][4][27] __attribute__((aligned(64))),dft864_r27_im[2][4][27] __attribute__((aligned(64)));
static void dft864_r27_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft864_r27_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft864_r27_tw_mutex);if(__atomic_load_n(&dft864_r27_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft864_r27_tw_mutex);return;}const float sc=1.0f/sqrtf(27.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<4;bl++){int off=8*bl;for(int br=0;br<27;br++){dft864_r27_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,864,sc);dft864_r27_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,864,sc,dir);}}}__atomic_store_n(&dft864_r27_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft864_r27_tw_mutex);}
static inline void radix27_stage_864(const c16_t *src,c16_t *b,dft_dir_t dir){enum{M=32};const int ds=dir==DFT_DIR_FORWARD?0:1;dft864_r27_tw_init();for(int off=0;off<M;off+=8){__m256i z[27];r27_raw8_n864(src,M,off,z,dir);for(int br=0;br<27;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT27)):complex_mul8_prepack_q15_256(z[br],dft864_r27_re[ds][off>>3][br],dft864_r27_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}
static void dft864_radix27_leaf32_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir){enum{R=27,M=32};c16_t b[R*M] __attribute__((aligned(64)));radix27_stage_864(src,b,dir);dft32x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);dft32x8_twiddle_scaled_store(b,M,dst,R,8,8,dir);dft32x8_twiddle_scaled_store(b,M,dst,R,16,8,dir);dft32x8_twiddle_scaled_store(b,M,dst,R,24,3,dir);}

/* Compute one AVX2 block of the radix-24 parent as a 3 x 8 PFA.
 * The fixed input index table performs the coprime 3/8 permutation, the
 * eight-point transforms are evaluated first, and the radix-3 butterflies
 * produce the 24 parent branches. The final assignments implement the
 * corresponding output permutation. No parent normalization is applied here. */
static inline void radix24_pfa_raw8_q15_256(const c16_t*src,int M,int off,__m256i y[24],dft_dir_t dir){static const int ids[3][8]={{0,9,18,3,12,21,6,15},{16,1,10,19,4,13,22,7},{8,17,2,11,20,5,14,23}};__m256i s[3][8];for(int n=0;n<3;n++){__m256i x[8];for(int j=0;j<8;j++)x[j]=_mm256_loadu_si256((const __m256i*)(src+ids[n][j]*M+off));dft8x8_q15_256_dir(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],&s[n][0],&s[n][1],&s[n][2],&s[n][3],&s[n][4],&s[n][5],&s[n][6],&s[n][7],dir);}for(int k=0;k<8;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(s[0][k],s[1][k],s[2][k],&z0,&z1,&z2,dir);y[(3*k)%24]=z0;y[(8+3*k)%24]=z1;y[(16+3*k)%24]=z2;}}

/* DFT1536 uses M=64. The parent W1536 coefficients include 1/sqrt(24);
 * branch 0 has no phase rotation and therefore uses the explicit Q15
 * 1/sqrt(24) multiply in the stage below. */
static pthread_mutex_t dft1536_radix24_twiddle_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft1536_radix24_twiddle_ready;static __m256i dft1536_radix24_twiddle_re[2][8][24] __attribute__((aligned(64))),dft1536_radix24_twiddle_im[2][8][24] __attribute__((aligned(64)));
static void dft1536_radix24_twiddles_init(void){if(__builtin_expect(__atomic_load_n(&dft1536_radix24_twiddle_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft1536_radix24_twiddle_mutex);if(__atomic_load_n(&dft1536_radix24_twiddle_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft1536_radix24_twiddle_mutex);return;}const float sc=1.0f/sqrtf((float)24);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<8;bl++){int off=8*bl;for(int br=0;br<24;br++){dft1536_radix24_twiddle_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,1536,sc);dft1536_radix24_twiddle_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,1536,sc,dir);}}}__atomic_store_n(&dft1536_radix24_twiddle_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft1536_radix24_twiddle_mutex);}
/* Store the radix-24 parent in branch-major order: b[branch * M + k].
 * This layout lets groups of eight DFT64 leaves be transposed into AVX2 lanes
 * without a whole-transform output buffer. */
static inline void dft1536_radix24_stage(const c16_t*src,c16_t*b,dft_dir_t dir){enum{M=64};const int ds=dir==DFT_DIR_FORWARD?0:1;dft1536_radix24_twiddles_init();for(int off=0;off<M;off+=8){__m256i z[24];radix24_pfa_raw8_q15_256(src,M,off,z,dir);for(int br=0;br<24;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT24)):complex_mul8_prepack_q15_256(z[br],dft1536_radix24_twiddle_re[ds][off>>3][br],dft1536_radix24_twiddle_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}
/* DFT1536 = radix-24 parent followed by 24 DFT64 leaves. The leaves are
 * processed as three groups of eight and written directly to the final
 * radix-24-interleaved output. */
static void dft1536_radix24_pfa_leaf64(const c16_t*src,c16_t*dst,dft_dir_t dir){enum{M=64};c16_t b[24*M] __attribute__((aligned(64)));dft1536_radix24_stage(src,b,dir);dft1536_radix24_dft64x8_store(b,M,dst,24,0,dir);dft1536_radix24_dft64x8_store(b,M,dst,24,8,dir);dft1536_radix24_dft64x8_store(b,M,dst,24,16,dir);}

/* DFT1728: R27 x DFT64. */
/* Convert eight branch-major DFT64 leaves into the lane-parallel layout used
 * by dft64x8_selected_store(), then write them with output_stride=27. The DC
 * vector is written explicitly because the dedicated DC accumulation uses a
 * wider intermediate sum. */
static __attribute__((always_inline)) inline void dft1728_radix27_dft64x8_store(const c16_t *b,int M,c16_t *dst,int output_stride,int first_branch,dft_dir_t dir){__m256i leaf[64] __attribute__((aligned(64)));selected_q15_twiddles_init();for(int n=0;n<64;n+=8){__m256i z[8];for(int br=0;br<8;br++)z[br]=_mm256_loadu_si256((const __m256i*)(b+(first_branch+br)*M+n));transpose8_complex_i16_256(&z[0],&z[1],&z[2],&z[3],&z[4],&z[5],&z[6],&z[7]);for(int l=0;l<8;l++)leaf[n+l]=z[l];}const __m256i dc=dft64x8_dc_q15_256(leaf);dft64x8_selected_store(leaf,dst,output_stride,first_branch,dir);_mm256_storeu_si256((__m256i*)(dst+first_branch),dc);}
static __attribute__((always_inline)) inline __m256i r27_1728_w9_mul(__m256i x,int p,dft_dir_t dir){int16_t wr,wi;switch(p){case 1:wr=25102;wi=-21063;break;case 2:wr=5690;wi=-32270;break;default:wr=-30792;wi=-11207;break;}return complex_mul8_bcast_q15_256_parent(x,wr,twiddle_im_scalar_dir_i16(wi,dir));}
static __attribute__((always_inline)) inline void r27_1728_dft9_raw(__m256i x[9],__m256i y[9],dft_dir_t dir){__m256i a[3][3];for(int n=0;n<3;n++)radix3_butterfly8_q15_256(x[n],x[n+3],x[n+6],&a[n][0],&a[n][1],&a[n][2],dir);a[1][1]=r27_1728_w9_mul(a[1][1],1,dir);a[2][1]=r27_1728_w9_mul(a[2][1],2,dir);a[1][2]=r27_1728_w9_mul(a[1][2],2,dir);a[2][2]=r27_1728_w9_mul(a[2][2],4,dir);for(int k=0;k<3;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+3]=z1;y[k+6]=z2;}}
static __attribute__((always_inline)) inline __m256i r27_1728_internal_tw(__m256i x,int n1,int k,dft_dir_t dir){static const int16_t wr[2][9]={{32767,31885,29283,25102,19568,12979,5690,-1905,-9398},{32767,29283,19568,5690,-9398,-22487,-30792,-32546,-27377}};static const int16_t wi[2][9]={{0,-7557,-14706,-21063,-26284,-30088,-32270,-32713,-31391},{0,-14706,-26284,-32270,-31391,-23835,-11207,3804,18006}};return complex_mul8_bcast_q15_256_parent(x,wr[n1-1][k],twiddle_im_scalar_dir_i16(wi[n1-1][k],dir));}
static __attribute__((always_inline)) inline void r27_1728_raw8(const c16_t *src,int M,int off,__m256i y[27],dft_dir_t dir){__m256i a[3][9];for(int n1=0;n1<3;n1++){__m256i x[9];for(int n9=0;n9<9;n9++)x[n9]=_mm256_loadu_si256((const __m256i*)(src+(n1+3*n9)*M+off));r27_1728_dft9_raw(x,a[n1],dir);}for(int n1=1;n1<3;n1++)for(int k=1;k<9;k++)a[n1][k]=r27_1728_internal_tw(a[n1][k],n1,k,dir);for(int k=0;k<9;k++){__m256i z0,z1,z2;radix3_butterfly8_q15_256(a[0][k],a[1][k],a[2][k],&z0,&z1,&z2,dir);y[k]=z0;y[k+9]=z1;y[k+18]=z2;}}
static pthread_mutex_t dft1728_r27_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft1728_r27_ready;static __m256i dft1728_r27_re[2][8][27] __attribute__((aligned(64))),dft1728_r27_im[2][8][27] __attribute__((aligned(64)));
static void dft1728_r27_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft1728_r27_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft1728_r27_tw_mutex);if(__atomic_load_n(&dft1728_r27_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft1728_r27_tw_mutex);return;}const float sc=1.0f/sqrtf(27.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<8;bl++){int off=8*bl;for(int br=0;br<27;br++){dft1728_r27_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,1728,sc);dft1728_r27_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,1728,sc,dir);}}}__atomic_store_n(&dft1728_r27_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft1728_r27_tw_mutex);}
static inline void radix27_stage_1728(const c16_t *src,c16_t *b,dft_dir_t dir){enum{M=64};const int ds=dir==DFT_DIR_FORWARD?0:1;dft1728_r27_tw_init();for(int off=0;off<M;off+=8){__m256i z[27];r27_1728_raw8(src,M,off,z,dir);for(int br=0;br<27;br++){__m256i v=br==0?_mm256_mulhrs_epi16(z[br],_mm256_set1_epi16(Q15_INV_SQRT27)):complex_mul8_prepack_q15_256(z[br],dft1728_r27_re[ds][off>>3][br],dft1728_r27_im[ds][off>>3][br]);_mm256_store_si256((__m256i*)(b+br*M+off),v);}}}
static void dft1728_radix27_leaf64_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir){enum{R=27,M=64};c16_t b[R*M] __attribute__((aligned(64)));radix27_stage_1728(src,b,dir);dft1728_radix27_dft64x8_store(b,M,dst,R,0,dir);dft1728_radix27_dft64x8_store(b,M,dst,R,8,dir);dft1728_radix27_dft64x8_store(b,M,dst,R,16,dir);c16_t tail[64] __attribute__((aligned(64)));for(int br=24;br<R;br++){dft64_avx(b+br*M,tail,dir);for(int k=0;k<M;k++)dst[R*k+br]=tail[k];}}

/* DFT960-only true radix-30 AVX2 parent using a Good-Thomas 6 x 5 PFA.
 *
 * 30 = 6 x 5 with gcd(6,5)=1, so the parent needs no internal twiddle
 * multiplications between the radix-6 and radix-5 transforms.  The radix-6
 * itself is a Good-Thomas 2 x 3 transform, also twiddle-free.
 *
 * Input CRT mapping for the 6 x 5 PFA:
 *   n = 25*n6 + 6*n5 (mod 30)
 * Output mapping:
 *   k = 5*k6 + 6*k5 (mod 30)
 *
 * The 6 x 5 PFA arithmetic itself is left unscaled. The complete unitary
 * 1/sqrt(30) parent scale is folded into the final W960 twiddles.
 */
static pthread_mutex_t dft960_r30_pfa_twiddle_mutex = PTHREAD_MUTEX_INITIALIZER;
static int dft960_r30_pfa_twiddles_ready;
static __m256i dft960_r30_pfa_re[2][4][30] __attribute__((aligned(64)));
static __m256i dft960_r30_pfa_im[2][4][30] __attribute__((aligned(64)));

static void dft960_r30_pfa_twiddles_init(void)
{
  if (__builtin_expect(__atomic_load_n(&dft960_r30_pfa_twiddles_ready, __ATOMIC_ACQUIRE), 1))
    return;

  pthread_mutex_lock(&dft960_r30_pfa_twiddle_mutex);
  if (__atomic_load_n(&dft960_r30_pfa_twiddles_ready, __ATOMIC_RELAXED)) {
    pthread_mutex_unlock(&dft960_r30_pfa_twiddle_mutex);
    return;
  }

  const float scale30 = 1.0f / sqrtf(30.0f);
  for (int ds = 0; ds < 2; ds++) {
    const dft_dir_t dir = ds == 0 ? DFT_DIR_FORWARD : DFT_DIR_INVERSE;
    for (int block = 0; block < 4; block++) {
      const int off = 8 * block;
      for (int br = 0; br < 30; br++) {
        dft960_r30_pfa_re[ds][block][br] = pack8_twiddle_q15_re_re_scaled(off, br, 960, scale30);
        dft960_r30_pfa_im[ds][block][br] = pack8_twiddle_q15_im_signed_scaled(off, br, 960, scale30, dir);
      }
    }
  }

  __atomic_store_n(&dft960_r30_pfa_twiddles_ready, 1, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&dft960_r30_pfa_twiddle_mutex);
}

/* Unitary DFT6 by Good-Thomas 2 x 3.
 * Input coordinate mapping n = 3*n2 + 4*n3 (mod 6) gives the three DFT2
 * pairs (0,3), (4,1), (2,5).  The DFT2 results are scaled once by
 * 1/sqrt(6), then two unscaled radix-3 butterflies complete the DFT6.
 * This is algebraically a unitary DFT6 while avoiding all W6 twiddles.
 */
static __attribute__((always_inline)) inline void dft6_pfa8_q15_256(__m256i x0,
                                                                     __m256i x1,
                                                                     __m256i x2,
                                                                     __m256i x3,
                                                                     __m256i x4,
                                                                     __m256i x5,
                                                                     __m256i y[6],
                                                                     dft_dir_t dir)
{
  const __m256i a0 = _mm256_adds_epi16(x0, x3);
  const __m256i b0 = _mm256_subs_epi16(x0, x3);
  const __m256i a1 = _mm256_adds_epi16(x4, x1);
  const __m256i b1 = _mm256_subs_epi16(x4, x1);
  const __m256i a2 = _mm256_adds_epi16(x2, x5);
  const __m256i b2 = _mm256_subs_epi16(x2, x5);

  __m256i e0, e1, e2, o0, o1, o2;
  radix3_butterfly8_q15_256(a0, a1, a2, &e0, &e1, &e2, dir);
  radix3_butterfly8_q15_256(b0, b1, b2, &o0, &o1, &o2, dir);

  /* k = 3*k2 + 2*k3 (mod 6). */
  y[0] = e0;
  y[2] = e1;
  y[4] = e2;
  y[3] = o0;
  y[5] = o1;
  y[1] = o2;
}

/* DFT480: R30 x DFT16. The parent uses the same 6 x 5 Good-Thomas mapping
 * as DFT960; the complete 1/sqrt(30) scale is in the final W480 twiddles. */
static pthread_mutex_t dft480_r30_twiddle_mutex = PTHREAD_MUTEX_INITIALIZER;
static int dft480_r30_twiddles_ready;
static __m256i dft480_r30_re[2][2][30] __attribute__((aligned(64)));
static __m256i dft480_r30_im[2][2][30] __attribute__((aligned(64)));

static void dft480_r30_twiddles_init(void)
{
  if (__builtin_expect(__atomic_load_n(&dft480_r30_twiddles_ready, __ATOMIC_ACQUIRE), 1)) return;
  pthread_mutex_lock(&dft480_r30_twiddle_mutex);
  if (__atomic_load_n(&dft480_r30_twiddles_ready, __ATOMIC_RELAXED)) {
    pthread_mutex_unlock(&dft480_r30_twiddle_mutex); return;
  }
  const float scale30 = 1.0f / sqrtf(30.0f);
  for (int ds=0; ds<2; ds++) {
    const dft_dir_t dir = ds==0 ? DFT_DIR_FORWARD : DFT_DIR_INVERSE;
    for (int block=0; block<2; block++) {
      const int off=8*block;
      for (int br=0; br<30; br++) {
        dft480_r30_re[ds][block][br]=pack8_twiddle_q15_re_re_scaled(off,br,480,scale30);
        dft480_r30_im[ds][block][br]=pack8_twiddle_q15_im_signed_scaled(off,br,480,scale30,dir);
      }
    }
  }
  __atomic_store_n(&dft480_r30_twiddles_ready,1,__ATOMIC_RELEASE);
  pthread_mutex_unlock(&dft480_r30_twiddle_mutex);
}

static inline void radix30_pfa_stage_to_branch_major_q15_256_480(const c16_t *src,c16_t *b,dft_dir_t dir)
{
  enum {M=16}; const int ds=dir==DFT_DIR_FORWARD?0:1; dft480_r30_twiddles_init();
  for (int off=0; off<M; off+=8) {
    __m256i s[5][6] __attribute__((aligned(64)));
#define R30_480_DFT6(C,I0,I1,I2,I3,I4,I5) do { \
  dft6_pfa8_q15_256(_mm256_loadu_si256((const __m256i*)(src+(I0)*M+off)), \
                       _mm256_loadu_si256((const __m256i*)(src+(I1)*M+off)), \
                       _mm256_loadu_si256((const __m256i*)(src+(I2)*M+off)), \
                       _mm256_loadu_si256((const __m256i*)(src+(I3)*M+off)), \
                       _mm256_loadu_si256((const __m256i*)(src+(I4)*M+off)), \
                       _mm256_loadu_si256((const __m256i*)(src+(I5)*M+off)),s[(C)],dir); } while(0)
    R30_480_DFT6(0,0,25,20,15,10,5); R30_480_DFT6(1,6,1,26,21,16,11);
    R30_480_DFT6(2,12,7,2,27,22,17); R30_480_DFT6(3,18,13,8,3,28,23);
    R30_480_DFT6(4,24,19,14,9,4,29);
#undef R30_480_DFT6
#define R30_480_STORE(BR,Z) do { __m256i v; \
  if ((BR)==0) v=_mm256_mulhrs_epi16((Z),_mm256_set1_epi16(Q15_INV_SQRT30)); \
  else v=complex_mul8_prepack_q15_256((Z),dft480_r30_re[ds][off>>3][(BR)],dft480_r30_im[ds][off>>3][(BR)]); \
  _mm256_store_si256((__m256i*)(b+(BR)*M+off),v); } while(0)
#define R30_480_DFT5(K6,B0,B1,B2,B3,B4) do { __m256i z0,z1,z2,z3,z4; \
  radix5_butterfly8_q15_256(s[0][K6],s[1][K6],s[2][K6],s[3][K6],s[4][K6],&z0,&z1,&z2,&z3,&z4,dir); \
  R30_480_STORE(B0,z0); R30_480_STORE(B1,z1); R30_480_STORE(B2,z2); R30_480_STORE(B3,z3); R30_480_STORE(B4,z4); } while(0)
    R30_480_DFT5(0,0,6,12,18,24); R30_480_DFT5(1,5,11,17,23,29);
    R30_480_DFT5(2,10,16,22,28,4); R30_480_DFT5(3,15,21,27,3,9);
    R30_480_DFT5(4,20,26,2,8,14); R30_480_DFT5(5,25,1,7,13,19);
#undef R30_480_DFT5
#undef R30_480_STORE
  }
}

static void dft480_radix30_pfa_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir)
{
  enum {R=30,M=16}; c16_t b[R*M] __attribute__((aligned(64)));
  radix30_pfa_stage_to_branch_major_q15_256_480(src,b,dir);
  dft16x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);
  dft16x8_twiddle_scaled_store(b,M,dst,R,8,8,dir);
  dft16x8_twiddle_scaled_store(b,M,dst,R,16,8,dir);
  dft16x8_twiddle_scaled_store(b,M,dst,R,24,6,dir);
}

/* DFT1920: true R30 PFA, M=64. */
static pthread_mutex_t dft1920_r30_tw_mutex=PTHREAD_MUTEX_INITIALIZER;static int dft1920_r30_ready;static __m256i dft1920_r30_re[2][8][30] __attribute__((aligned(64))),dft1920_r30_im[2][8][30] __attribute__((aligned(64)));
static void dft1920_r30_tw_init(void){if(__builtin_expect(__atomic_load_n(&dft1920_r30_ready,__ATOMIC_ACQUIRE),1))return;pthread_mutex_lock(&dft1920_r30_tw_mutex);if(__atomic_load_n(&dft1920_r30_ready,__ATOMIC_RELAXED)){pthread_mutex_unlock(&dft1920_r30_tw_mutex);return;}const float sc=1.0f/sqrtf(30.0f);for(int ds=0;ds<2;ds++){const dft_dir_t dir=ds==0?DFT_DIR_FORWARD:DFT_DIR_INVERSE;for(int bl=0;bl<8;bl++){int off=8*bl;for(int br=0;br<30;br++){dft1920_r30_re[ds][bl][br]=pack8_twiddle_q15_re_re_scaled(off,br,1920,sc);dft1920_r30_im[ds][bl][br]=pack8_twiddle_q15_im_signed_scaled(off,br,1920,sc,dir);}}}__atomic_store_n(&dft1920_r30_ready,1,__ATOMIC_RELEASE);pthread_mutex_unlock(&dft1920_r30_tw_mutex);}
static inline void radix30_stage_1920(const c16_t *src,c16_t *b,dft_dir_t dir){enum{M=64};const int ds=dir==DFT_DIR_FORWARD?0:1;dft1920_r30_tw_init();for(int off=0;off<M;off+=8){__m256i s[5][6];
#define D619(C,I0,I1,I2,I3,I4,I5) dft6_pfa8_q15_256(_mm256_loadu_si256((const __m256i*)(src+(I0)*M+off)),_mm256_loadu_si256((const __m256i*)(src+(I1)*M+off)),_mm256_loadu_si256((const __m256i*)(src+(I2)*M+off)),_mm256_loadu_si256((const __m256i*)(src+(I3)*M+off)),_mm256_loadu_si256((const __m256i*)(src+(I4)*M+off)),_mm256_loadu_si256((const __m256i*)(src+(I5)*M+off)),s[C],dir)
 D619(0,0,25,20,15,10,5);D619(1,6,1,26,21,16,11);D619(2,12,7,2,27,22,17);D619(3,18,13,8,3,28,23);D619(4,24,19,14,9,4,29);
#undef D619
#define ST19(BR,Z) do{__m256i v=(BR)==0?_mm256_mulhrs_epi16((Z),_mm256_set1_epi16(Q15_INV_SQRT30)):complex_mul8_prepack_q15_256((Z),dft1920_r30_re[ds][off>>3][BR],dft1920_r30_im[ds][off>>3][BR]);_mm256_store_si256((__m256i*)(b+(BR)*M+off),v);}while(0)
#define D519(K,B0,B1,B2,B3,B4) do{__m256i z0,z1,z2,z3,z4;radix5_butterfly8_q15_256(s[0][K],s[1][K],s[2][K],s[3][K],s[4][K],&z0,&z1,&z2,&z3,&z4,dir);ST19(B0,z0);ST19(B1,z1);ST19(B2,z2);ST19(B3,z3);ST19(B4,z4);}while(0)
 D519(0,0,6,12,18,24);D519(1,5,11,17,23,29);D519(2,10,16,22,28,4);D519(3,15,21,27,3,9);D519(4,20,26,2,8,14);D519(5,25,1,7,13,19);
#undef D519
#undef ST19
}}
static void dft1920_radix30_pfa_avx2_selected(const c16_t *src,c16_t *dst,dft_dir_t dir){enum{R=30,M=64};c16_t b[R*M] __attribute__((aligned(64)));radix30_stage_1920(src,b,dir);dft64x8_parent_store_from_branches(b,M,dst,R,0,dir);dft64x8_parent_store_from_branches(b,M,dst,R,8,dir);dft64x8_parent_store_from_branches(b,M,dst,R,16,dir);c16_t tail[64] __attribute__((aligned(64)));for(int br=24;br<30;br++){dft64_avx(b+br*M,tail,dir);for(int k=0;k<M;k++)dst[R*k+br]=tail[k];}}

static inline void radix30_pfa_stage_to_branch_major_q15_256_960(const c16_t *src,
                                                                   c16_t *b,
                                                                   dft_dir_t dir)
{
  enum { N = 960, R = 30, M = 32 };
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  dft960_r30_pfa_twiddles_init();

  for (int off = 0; off < M; off += 8) {
    __m256i s[5][6] __attribute__((aligned(64)));

#define R30_PFA_DFT6(C, I0, I1, I2, I3, I4, I5)                                                \
    do {                                                                                          \
      dft6_pfa8_q15_256(_mm256_loadu_si256((const __m256i *)(src + (I0) * M + off)),             \
                           _mm256_loadu_si256((const __m256i *)(src + (I1) * M + off)),             \
                           _mm256_loadu_si256((const __m256i *)(src + (I2) * M + off)),             \
                           _mm256_loadu_si256((const __m256i *)(src + (I3) * M + off)),             \
                           _mm256_loadu_si256((const __m256i *)(src + (I4) * M + off)),             \
                           _mm256_loadu_si256((const __m256i *)(src + (I5) * M + off)),             \
                           s[(C)], dir);                                                             \
    } while (0)

    /* n = 25*n6 + 6*n5 (mod 30), n6 = 0..5. */
    R30_PFA_DFT6(0,  0, 25, 20, 15, 10,  5);
    R30_PFA_DFT6(1,  6,  1, 26, 21, 16, 11);
    R30_PFA_DFT6(2, 12,  7,  2, 27, 22, 17);
    R30_PFA_DFT6(3, 18, 13,  8,  3, 28, 23);
    R30_PFA_DFT6(4, 24, 19, 14,  9,  4, 29);
#undef R30_PFA_DFT6

#define R30_PFA_STORE(BR, Z)                                                                      \
    do {                                                                                           \
      __m256i _v;                                                                                  \
      if ((BR) == 0)                                                                               \
        _v = _mm256_mulhrs_epi16((Z), _mm256_set1_epi16(Q15_INV_SQRT30));                         \
      else                                                                                         \
        _v = complex_mul8_prepack_q15_256((Z),                                                    \
                                           dft960_r30_pfa_re[ds][off >> 3][(BR)],                  \
                                           dft960_r30_pfa_im[ds][off >> 3][(BR)]);                 \
      _mm256_store_si256((__m256i *)(b + (BR) * M + off), _v);                                   \
    } while (0)

#define R30_PFA_DFT5(K6, B0, B1, B2, B3, B4)                                                      \
    do {                                                                                           \
      __m256i z0, z1, z2, z3, z4;                                                                \
      radix5_butterfly8_q15_256(s[0][(K6)], s[1][(K6)], s[2][(K6)], s[3][(K6)], s[4][(K6)],      \
                                 &z0, &z1, &z2, &z3, &z4, dir);                                    \
      R30_PFA_STORE((B0), z0);                                                                     \
      R30_PFA_STORE((B1), z1);                                                                     \
      R30_PFA_STORE((B2), z2);                                                                     \
      R30_PFA_STORE((B3), z3);                                                                     \
      R30_PFA_STORE((B4), z4);                                                                     \
    } while (0)

    /* k = 5*k6 + 6*k5 (mod 30), k5 = 0..4. */
    R30_PFA_DFT5(0,  0,  6, 12, 18, 24);
    R30_PFA_DFT5(1,  5, 11, 17, 23, 29);
    R30_PFA_DFT5(2, 10, 16, 22, 28,  4);
    R30_PFA_DFT5(3, 15, 21, 27,  3,  9);
    R30_PFA_DFT5(4, 20, 26,  2,  8, 14);
    R30_PFA_DFT5(5, 25,  1,  7, 13, 19);
#undef R30_PFA_DFT5
#undef R30_PFA_STORE
  }
}

static void dft960_radix30_pfa_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { R = 30, M = 32 };
  c16_t b[R * M] __attribute__((aligned(64)));

  radix30_pfa_stage_to_branch_major_q15_256_960(src, b, dir);

  dft32x8_twiddle_scaled_store(b,M,dst,R,0,8,dir);
  dft32x8_twiddle_scaled_store(b,M,dst,R,8,8,dir);
  dft32x8_twiddle_scaled_store(b,M,dst,R,16,8,dir);
  dft32x8_twiddle_scaled_store(b,M,dst,R,24,6,dir);
}

/* DFT2160 = true R30 PFA parent x 30 DFT72 leaves.  Four groups of eight
 * leaves are evaluated lane-wise and written directly to the final R30
 * layout; the last group uses the established AVX2 mask-store tail. */
static pthread_mutex_t dft2160_r30_tw_mutex = PTHREAD_MUTEX_INITIALIZER;
static int dft2160_r30_tw_ready;
static __m256i dft2160_r30_re[9][30] __attribute__((aligned(64)));
static __m256i dft2160_r30_im_fwd[9][30] __attribute__((aligned(64)));
static int16_t dft2160_r9_w72_re[2][24], dft2160_r9_w72_im_fwd[2][24];
static int16_t dft2160_r9_w24_re[2][8], dft2160_r9_w24_im_fwd[2][8];

static void dft2160_twiddles_init(void)
{
  if (__builtin_expect(__atomic_load_n(&dft2160_r30_tw_ready, __ATOMIC_ACQUIRE), 1))
    return;
  pthread_mutex_lock(&dft2160_r30_tw_mutex);
  if (!__atomic_load_n(&dft2160_r30_tw_ready, __ATOMIC_RELAXED)) {
    const float s30 = 1.0f / sqrtf(30.0f), s3 = 1.0f / sqrtf(3.0f);
    for (int block = 0; block < 9; block++)
      for (int br = 0; br < 30; br++) {
        dft2160_r30_re[block][br] = pack8_twiddle_q15_re_re_scaled(8 * block, br, 2160, s30);
        dft2160_r30_im_fwd[block][br] =
            pack8_twiddle_q15_im_signed_scaled(8 * block, br, 2160, s30, DFT_DIR_FORWARD);
      }
    for (int mul = 1; mul <= 2; mul++) {
      for (int k = 0; k < 24; k++) {
        const float a = 2.0f * (float)M_PI * (float)((mul * k) % 72) / 72.0f;
        dft2160_r9_w72_re[mul - 1][k] = q15_from_float(cosf(a) * s3);
        dft2160_r9_w72_im_fwd[mul - 1][k] = q15_from_float(-sinf(a) * s3);
      }
      for (int k = 0; k < 8; k++) {
        const float a = 2.0f * (float)M_PI * (float)((mul * k) % 24) / 24.0f;
        dft2160_r9_w24_re[mul - 1][k] = q15_from_float(cosf(a) * s3);
        dft2160_r9_w24_im_fwd[mul - 1][k] = q15_from_float(-sinf(a) * s3);
      }
    }
    __atomic_store_n(&dft2160_r30_tw_ready, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&dft2160_r30_tw_mutex);
}

static __attribute__((always_inline)) inline __m256i
dft2160_r9_tw_mul(__m256i x, int16_t wr, int16_t wi_fwd, dft_dir_t dir)
{
  return complex_mul8_bcast_q15_256_parent(x, wr, twiddle_im_scalar_dir_i16(wi_fwd, dir));
}

static __attribute__((always_inline)) inline void
dft2160_unitary_dft8x8(const __m256i x[8], __m256i y[8], dft_dir_t dir)
{
  __m256i a[4][2];
  for (int n = 0; n < 4; n++) {
    const __m256i s = _mm256_adds_epi16(x[n], x[n + 4]);
    const __m256i d = _mm256_subs_epi16(x[n], x[n + 4]);
    a[n][0] = leaf_twiddle_mul_q15_256(s, dft8_leaf_twiddle_re, dft8_leaf_twiddle_im_fwd, 8, 0, dir);
    a[n][1] = leaf_twiddle_mul_q15_256(d, dft8_leaf_twiddle_re, dft8_leaf_twiddle_im_fwd, 8, n, dir);
  }
  for (int k2 = 0; k2 < 2; k2++) {
    __m256i z[4];
    dft4x8_raw_q15_256(a[0][k2], a[1][k2], a[2][k2], a[3][k2], z, dir);
    for (int k1 = 0; k1 < 4; k1++)
      y[k2 + 2 * k1] = z[k1];
  }
}

static __attribute__((noinline)) void
dft2160_dft72x8_store(const c16_t *b, c16_t *dst, int first, int valid, dft_dir_t dir)
{
  __m256i x[72], r9[9][8];
  const __m256i s3 = _mm256_set1_epi16(Q15_INV_SQRT3);
  const __m256i mask = complex_lane_mask_epi32(valid);
  load_batched_leaf_inputs(b, 72, first, valid, 72, x);

  for (int off = 0; off < 8; off++) {
    __m256i s[3][3];
    for (int a = 0; a < 3; a++) {
      __m256i z0, z1, z2;
      radix3_butterfly8_q15_256(x[(a + 0) * 8 + off], x[(a + 3) * 8 + off],
                                 x[(a + 6) * 8 + off], &z0, &z1, &z2, dir);
      const int k = a * 8 + off;
      s[a][0] = _mm256_mulhrs_epi16(z0, s3);
      s[a][1] = dft2160_r9_tw_mul(z1, dft2160_r9_w72_re[0][k],
                                  dft2160_r9_w72_im_fwd[0][k], dir);
      s[a][2] = dft2160_r9_tw_mul(z2, dft2160_r9_w72_re[1][k],
                                  dft2160_r9_w72_im_fwd[1][k], dir);
    }
    for (int beta = 0; beta < 3; beta++) {
      __m256i z0, z1, z2;
      radix3_butterfly8_q15_256(s[0][beta], s[1][beta], s[2][beta], &z0, &z1, &z2, dir);
      r9[beta + 0 * 3][off] = _mm256_mulhrs_epi16(z0, s3);
      r9[beta + 1 * 3][off] = dft2160_r9_tw_mul(z1, dft2160_r9_w24_re[0][off],
                                                dft2160_r9_w24_im_fwd[0][off], dir);
      r9[beta + 2 * 3][off] = dft2160_r9_tw_mul(z2, dft2160_r9_w24_re[1][off],
                                                dft2160_r9_w24_im_fwd[1][off], dir);
    }
  }

  for (int br9 = 0; br9 < 9; br9++) {
    __m256i y[8];
    dft2160_unitary_dft8x8(r9[br9], y, dir);
    for (int k8 = 0; k8 < 8; k8++)
      _mm256_maskstore_epi32((int *)(dst + 30 * (9 * k8 + br9) + first), mask, y[k8]);
  }
}

static inline void dft2160_radix30_stage(const c16_t *src, c16_t *b, dft_dir_t dir)
{
  enum { M = 72 };
  dft2160_twiddles_init();
  for (int off = 0; off < M; off += 8) {
    __m256i s[5][6];
#define DFT2160_DFT6(C,I0,I1,I2,I3,I4,I5) \
    dft6_pfa8_q15_256(_mm256_loadu_si256((const __m256i *)(src + (I0) * M + off)), \
                        _mm256_loadu_si256((const __m256i *)(src + (I1) * M + off)), \
                        _mm256_loadu_si256((const __m256i *)(src + (I2) * M + off)), \
                        _mm256_loadu_si256((const __m256i *)(src + (I3) * M + off)), \
                        _mm256_loadu_si256((const __m256i *)(src + (I4) * M + off)), \
                        _mm256_loadu_si256((const __m256i *)(src + (I5) * M + off)), s[C], dir)
    DFT2160_DFT6(0,0,25,20,15,10,5); DFT2160_DFT6(1,6,1,26,21,16,11);
    DFT2160_DFT6(2,12,7,2,27,22,17); DFT2160_DFT6(3,18,13,8,3,28,23);
    DFT2160_DFT6(4,24,19,14,9,4,29);
#undef DFT2160_DFT6
#define DFT2160_STORE(BR,Z) do { \
    const __m256i _im = twiddle_im_dir_256(dft2160_r30_im_fwd[off >> 3][BR], dir); \
    const __m256i _v = (BR) == 0 \
        ? _mm256_mulhrs_epi16((Z), _mm256_set1_epi16(Q15_INV_SQRT30)) \
        : complex_mul8_prepack_q15_256((Z), dft2160_r30_re[off >> 3][BR], _im); \
    _mm256_store_si256((__m256i *)(b + (BR) * M + off), _v); \
  } while (0)
#define DFT2160_DFT5(K,B0,B1,B2,B3,B4) do { \
    __m256i z0,z1,z2,z3,z4; \
    radix5_butterfly8_q15_256(s[0][K],s[1][K],s[2][K],s[3][K],s[4][K], \
                               &z0,&z1,&z2,&z3,&z4,dir); \
    DFT2160_STORE(B0,z0); DFT2160_STORE(B1,z1); DFT2160_STORE(B2,z2); \
    DFT2160_STORE(B3,z3); DFT2160_STORE(B4,z4); \
  } while (0)
    DFT2160_DFT5(0,0,6,12,18,24); DFT2160_DFT5(1,5,11,17,23,29);
    DFT2160_DFT5(2,10,16,22,28,4); DFT2160_DFT5(3,15,21,27,3,9);
    DFT2160_DFT5(4,20,26,2,8,14); DFT2160_DFT5(5,25,1,7,13,19);
#undef DFT2160_DFT5
#undef DFT2160_STORE
  }
}

static void dft2160_radix30_leaf72x8_direct(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  c16_t b[2160] __attribute__((aligned(64)));
  dft2160_radix30_stage(src, b, dir);
  dft2160_dft72x8_store(b, dst, 0, 8, dir);
  dft2160_dft72x8_store(b, dst, 8, 8, dir);
  dft2160_dft72x8_store(b, dst, 16, 8, dir);
  dft2160_dft72x8_store(b, dst, 24, 6, dir);
}

/* Size-specific dispatcher for the radix-15 family. N=480, 960 and 1920
 * use dedicated radix-30 PFA kernels; other supported sizes use the radix-15
 * parent followed by power-of-two leaves. */
static void dispatch_selected_radix15_radix30(const c16_t *src, c16_t *dst, int N, dft_dir_t dir)
{
  const int M = N / 15;
  AssertFatal(selected_radix15_radix30_size(N), "Invalid selected radix-15 N=%d\n", N);
  AssertFatal((M == 8 || M == 16 || M == 32 || M == 64 || M == 128) && (M & 3) == 0,
              "Invalid selected radix-15 leaf N=%d M=%d\n", N, M);

  if (N == 960) {
    dft960_radix30_pfa_avx2_selected(src, dst, dir);
    return;
  }

  if (N == 1920) {
    dft1920_radix30_pfa_avx2_selected(src, dst, dir);
    return;
  }

  const radix15_plan_t *plan = radix15_plan_get(N);
  AssertFatal(plan, "Missing radix-15 selected plan N=%d\n", N);

  if (N == 120) {
    dft120_radix15_pfa_avx2_selected(src, dst, dir);
    return;
  }

  if (N == 240) {
    dft240_radix15_pfa_avx2_selected(src, dst, dir);
    return;
  }

  if (N == 480) {
    dft480_radix30_pfa_avx2_selected(src, dst, dir);
    return;
  }

  c16_t work[2 * N] __attribute__((aligned(64)));
  c16_t *b = work;
  c16_t *y = work + N;

  radix15_35_stage_to_branch_major_q15_256(src, b, N, dir, r3_twiddle_slot(N), r5_twiddle_slot(N / 3));

  for (int br = 0; br < 15; br++)
    dft_power2_selected_child(b + br * M, y + br * M, M, dir, NULL);

  for (int k = 0; k < M; k += 4)
    radix15_scatter4_q15_128(y, dst, M, k);
}

static void radix3_pow2_selected(const c16_t *src, c16_t *dst, int N, dft_dir_t dir)
{
  const int size = N / 3;
  const size_t need = 2u * (size_t)N + dft_power2_mixed_large_work_len(size);
  c16_t *work = x86_dft_tls_work(need);
  if (!work)
    return;

  c16_t *sub = work;
  c16_t *tmp = work + N;
  c16_t *child_work = work + 2 * N;
  pack_radix3_selected(src, sub, size);
  dft_power2_selected_child(sub, tmp, size, dir, child_work);
  dft_power2_selected_child(sub + size, tmp + size, size, dir, child_work);
  dft_power2_selected_child(sub + 2 * size, tmp + 2 * size, size, dir, child_work);

  const radix3_selected_plan_t *r3plan = radix3_selected_plan_get(N);
  const r3_twiddle_t *tw = r3plan ? r3_twiddle_slot(N) : r3_twiddle_get(N);
  const __m128i *w1r = dir == DFT_DIR_FORWARD ? tw->r3_q15_w1_re : tw->r3_q15_w1_re_inv;
  const __m128i *w1i = dir == DFT_DIR_FORWARD ? tw->r3_q15_w1_im : tw->r3_q15_w1_im_inv;
  const __m128i *w2r = dir == DFT_DIR_FORWARD ? tw->r3_q15_w2_re : tw->r3_q15_w2_re_inv;
  const __m128i *w2i = dir == DFT_DIR_FORWARD ? tw->r3_q15_w2_im : tw->r3_q15_w2_im_inv;

  for (int k = 0; k < size; k += 4) {
    const int b = k >> 2;
    __m128i y0, y1, y2;
    radix3_combine4_q15_128_fast(_mm_load_si128((const __m128i *)(tmp + k)),
                                 _mm_load_si128((const __m128i *)(tmp + size + k)),
                                 _mm_load_si128((const __m128i *)(tmp + 2 * size + k)),
                                 w1r[b], w1i[b], w2r[b], w2i[b], &y0, &y1, &y2, dir);
    _mm_storeu_si128((__m128i *)(dst + k), y0);
    _mm_storeu_si128((__m128i *)(dst + size + k), y1);
    _mm_storeu_si128((__m128i *)(dst + 2 * size + k), y2);
  }

}

static void radix_3_fft_c16_scaled_strided(const c16_t *src, int stride, c16_t *dst, int N, dft_dir_t dir)
{
  if ((N % 3) != 0) {
    LOG_E(PHY, "Invalid radix-3 N=%d\n", N);
    return;
  }

  const int size = N / 3;

  if ((size & 3) != 0) {
    LOG_E(PHY, "Radix-3 child size=%d requires an unsupported scalar tail\n", size);
    return;
  }

  const r3_twiddle_t *tw = r3_twiddle_get(N);

  if (!tw || !tw->r3_q15_w1_re || !tw->r3_q15_w1_im || !tw->r3_q15_w2_re || !tw->r3_q15_w2_im || !tw->r3_q15_w1_re_inv
      || !tw->r3_q15_w1_im_inv || !tw->r3_q15_w2_re_inv || !tw->r3_q15_w2_im_inv) {
    LOG_E(PHY, "Missing radix-3 twiddles for N=%d\n", N);
    return;
  }

  c16_t tmp_stack[STACK_MAX_N] __attribute__((aligned(64)));
  const int use_recursive_tls = N > STACK_MAX_N;
  c16_t *tmp = use_recursive_tls ? x86_dft_recursive_work_acquire((size_t)N) : tmp_stack;
  if (!tmp) {
    LOG_E(PHY, "Radix-3 workspace allocation failed N=%d\n", N);
    return;
  }

  /*
   * Branch r:
   *   src[(3*n + r) * stride]
   *
   * Equivalent pointer:
   *   src + r*stride
   *
   * New stride:
   *   stride * 3
   */
  dft_mixed_radix_c16_scaled_strided(src + 0 * stride, stride * 3, tmp + 0 * size, size, dir);

  dft_mixed_radix_c16_scaled_strided(src + 1 * stride, stride * 3, tmp + 1 * size, size, dir);

  dft_mixed_radix_c16_scaled_strided(src + 2 * stride, stride * 3, tmp + 2 * size, size, dir);

  const __m128i *w1_re_tbl = (dir == DFT_DIR_FORWARD) ? tw->r3_q15_w1_re : tw->r3_q15_w1_re_inv;

  const __m128i *w1_im_tbl = (dir == DFT_DIR_FORWARD) ? tw->r3_q15_w1_im : tw->r3_q15_w1_im_inv;

  const __m128i *w2_re_tbl = (dir == DFT_DIR_FORWARD) ? tw->r3_q15_w2_re : tw->r3_q15_w2_re_inv;

  const __m128i *w2_im_tbl = (dir == DFT_DIR_FORWARD) ? tw->r3_q15_w2_im : tw->r3_q15_w2_im_inv;

  for (int k = 0; k < size; k += 4) {
    const int b = k >> 2;

    const __m128i A = _mm_load_si128((const __m128i *)(tmp + 0 * size + k));

    const __m128i X1 = _mm_load_si128((const __m128i *)(tmp + 1 * size + k));

    const __m128i X2 = _mm_load_si128((const __m128i *)(tmp + 2 * size + k));

    __m128i Y0, Y1, Y2;

    radix3_combine4_q15_128_fast(A, X1, X2, w1_re_tbl[b], w1_im_tbl[b], w2_re_tbl[b], w2_im_tbl[b], &Y0, &Y1, &Y2, dir);

    _mm_storeu_si128((__m128i *)(dst + 0 * size + k), Y0);
    _mm_storeu_si128((__m128i *)(dst + 1 * size + k), Y1);
    _mm_storeu_si128((__m128i *)(dst + 2 * size + k), Y2);
  }

  if (use_recursive_tls)
    x86_dft_recursive_work_release();
}

static inline void radix5_combine4_q15_128_fast(__m128i A,
                                                __m128i X1,
                                                __m128i X2,
                                                __m128i X3,
                                                __m128i X4,
                                                __m128i w1_re,
                                                __m128i w1_im,
                                                __m128i w2_re,
                                                __m128i w2_im,
                                                __m128i w3_re,
                                                __m128i w3_im,
                                                __m128i w4_re,
                                                __m128i w4_im,
                                                __m128i *Y0,
                                                __m128i *Y1,
                                                __m128i *Y2,
                                                __m128i *Y3,
                                                __m128i *Y4,
                                                dft_dir_t dir)
{
  const __m128i B = complex_mul4_prepack_q15_128(X1, w1_re, w1_im);
  const __m128i C = complex_mul4_prepack_q15_128(X2, w2_re, w2_im);
  const __m128i D = complex_mul4_prepack_q15_128(X3, w3_re, w3_im);
  const __m128i E = complex_mul4_prepack_q15_128(X4, w4_re, w4_im);

  const __m128i BE = _mm_adds_epi16(B, E);
  const __m128i BEminus = _mm_subs_epi16(B, E);
  const __m128i CD = _mm_adds_epi16(C, D);
  const __m128i CDminus = _mm_subs_epi16(C, D);

  /*
   * Y0 = (A + B + C + D + E) / sqrt(5)
   */
  const __m128i As = q15_mul_i16_128(A, 14654);
  *Y0 = _mm_adds_epi16(_mm_adds_epi16(BE, CD), As);

  /*
   * base1 = A/sqrt5 + c1/sqrt5*(B+E) + c2/sqrt5*(C+D)
   */

  const __m128i base1 = _mm_adds_epi16(_mm_adds_epi16(q15_mul_i16_128(BE, 10126), q15_mul_i16_128(CD, -26510)), As);

  /*
   * imag1 = s1/sqrt5*(B-E) + s2/sqrt5*(C-D)
   */
  const __m128i imag1 = _mm_adds_epi16(q15_mul_i16_128(BEminus, 31163), q15_mul_i16_128(CDminus, 19260));

  *Y1 = _mm_adds_epi16(base1, mul_minus_j_dir_i16_128(imag1, dir));
  *Y4 = _mm_adds_epi16(base1, mul_plus_j_dir_i16_128(imag1, dir));

  /*
   * base2 = A/sqrt5 + c2/sqrt5*(B+E) + c1/sqrt5*(C+D)
   */
  const __m128i base2 = _mm_adds_epi16(_mm_adds_epi16(q15_mul_i16_128(BE, -26510), q15_mul_i16_128(CD, 10126)), As);

  /*
   * imag2 = s2/sqrt5*(B-E) - s1/sqrt5*(C-D)
   */

  const __m128i imag2 = _mm_subs_epi16(q15_mul_i16_128(BEminus, 19260), q15_mul_i16_128(CDminus, 31163));

  *Y2 = _mm_adds_epi16(base2, mul_minus_j_dir_i16_128(imag2, dir));
  *Y3 = _mm_adds_epi16(base2, mul_plus_j_dir_i16_128(imag2, dir));
}

#define RADIX5_STACK_MAX_N 1024

static void radix_5_fft_c16_scaled_strided(const c16_t *src, int stride, c16_t *dst, int N, dft_dir_t dir)
{
  if ((N % 5) != 0) {
    LOG_E(PHY, "Invalid radix-5 N=%d\n", N);
    return;
  }

  const int size = N / 5;

  if ((size & 3) != 0) {
    LOG_E(PHY, "Radix-5 child size=%d requires an unsupported scalar tail\n", size);
    return;
  }

  const r5_twiddle_t *tw = r5_twiddle_get(N);

  if (!tw || !tw->r5_q15_w1_re || !tw->r5_q15_w1_im || !tw->r5_q15_w2_re || !tw->r5_q15_w2_im || !tw->r5_q15_w3_re
      || !tw->r5_q15_w3_im || !tw->r5_q15_w4_re || !tw->r5_q15_w4_im || !tw->r5_q15_w1_re_inv || !tw->r5_q15_w1_im_inv
      || !tw->r5_q15_w2_re_inv || !tw->r5_q15_w2_im_inv || !tw->r5_q15_w3_re_inv || !tw->r5_q15_w3_im_inv || !tw->r5_q15_w4_re_inv
      || !tw->r5_q15_w4_im_inv) {
    LOG_E(PHY, "Missing radix-5 twiddles for N=%d\n", N);
    return;
  }

  c16_t tmp_stack[RADIX5_STACK_MAX_N] __attribute__((aligned(64)));
  const int use_recursive_tls = N > RADIX5_STACK_MAX_N;
  c16_t *tmp = use_recursive_tls ? x86_dft_recursive_work_acquire((size_t)N) : tmp_stack;
  if (!tmp) {
    LOG_E(PHY, "Radix-5 workspace allocation failed N=%d\n", N);
    return;
  }

  dft_mixed_radix_c16_scaled_strided(src + 0 * stride, stride * 5, tmp + 0 * size, size, dir);

  dft_mixed_radix_c16_scaled_strided(src + 1 * stride, stride * 5, tmp + 1 * size, size, dir);

  dft_mixed_radix_c16_scaled_strided(src + 2 * stride, stride * 5, tmp + 2 * size, size, dir);

  dft_mixed_radix_c16_scaled_strided(src + 3 * stride, stride * 5, tmp + 3 * size, size, dir);

  dft_mixed_radix_c16_scaled_strided(src + 4 * stride, stride * 5, tmp + 4 * size, size, dir);

  const __m128i *w1_re_tbl = (dir == DFT_DIR_FORWARD) ? tw->r5_q15_w1_re : tw->r5_q15_w1_re_inv;
  const __m128i *w1_im_tbl = (dir == DFT_DIR_FORWARD) ? tw->r5_q15_w1_im : tw->r5_q15_w1_im_inv;
  const __m128i *w2_re_tbl = (dir == DFT_DIR_FORWARD) ? tw->r5_q15_w2_re : tw->r5_q15_w2_re_inv;
  const __m128i *w2_im_tbl = (dir == DFT_DIR_FORWARD) ? tw->r5_q15_w2_im : tw->r5_q15_w2_im_inv;
  const __m128i *w3_re_tbl = (dir == DFT_DIR_FORWARD) ? tw->r5_q15_w3_re : tw->r5_q15_w3_re_inv;
  const __m128i *w3_im_tbl = (dir == DFT_DIR_FORWARD) ? tw->r5_q15_w3_im : tw->r5_q15_w3_im_inv;
  const __m128i *w4_re_tbl = (dir == DFT_DIR_FORWARD) ? tw->r5_q15_w4_re : tw->r5_q15_w4_re_inv;
  const __m128i *w4_im_tbl = (dir == DFT_DIR_FORWARD) ? tw->r5_q15_w4_im : tw->r5_q15_w4_im_inv;

  for (int k = 0; k < size; k += 4) {
    const int b = k >> 2;

    const __m128i A = _mm_load_si128((const __m128i *)(tmp + 0 * size + k));

    const __m128i X1 = _mm_load_si128((const __m128i *)(tmp + 1 * size + k));

    const __m128i X2 = _mm_load_si128((const __m128i *)(tmp + 2 * size + k));

    const __m128i X3 = _mm_load_si128((const __m128i *)(tmp + 3 * size + k));

    const __m128i X4 = _mm_load_si128((const __m128i *)(tmp + 4 * size + k));

    __m128i Y0, Y1, Y2, Y3, Y4;

    radix5_combine4_q15_128_fast(A,
                                 X1,
                                 X2,
                                 X3,
                                 X4,
                                 w1_re_tbl[b],
                                 w1_im_tbl[b],
                                 w2_re_tbl[b],
                                 w2_im_tbl[b],
                                 w3_re_tbl[b],
                                 w3_im_tbl[b],
                                 w4_re_tbl[b],
                                 w4_im_tbl[b],
                                 &Y0,
                                 &Y1,
                                 &Y2,
                                 &Y3,
                                 &Y4,
                                 dir);

    _mm_storeu_si128((__m128i *)(dst + 0 * size + k), Y0);
    _mm_storeu_si128((__m128i *)(dst + 1 * size + k), Y1);
    _mm_storeu_si128((__m128i *)(dst + 2 * size + k), Y2);
    _mm_storeu_si128((__m128i *)(dst + 3 * size + k), Y3);
    _mm_storeu_si128((__m128i *)(dst + 4 * size + k), Y4);
  }

  if (use_recursive_tls)
    x86_dft_recursive_work_release();
}

/* DFT1440 = R5 x DFT288. Pack one radix-5 residue at a time so the five
 * children are contiguous and can use the R9 x DFT32 direct
 * kernel.  The final R5 combine is intentionally identical to the generic
 * path; only the slow strided child recursion is replaced. */
static void dft1440_r5_leaf288_direct(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { R = 5, M = 288, N = R * M };
  c16_t tmp[N] __attribute__((aligned(64)));

  for (int br = 0; br < R; br++) {
    for (int n = 0; n < M; n += 4)
      _mm_store_si128((__m128i *)(tmp + br * M + n), load4_complex_strided_c16(src + br, R, n));
    radix9_terminal_leaf32_direct(tmp + br * M, tmp + br * M, dir);
  }

  const r5_twiddle_t *tw = r5_twiddle_get(N);
  AssertFatal(tw, "Missing DFT1440 radix-5 twiddles\n");
  const __m128i *w1r = dir == DFT_DIR_FORWARD ? tw->r5_q15_w1_re : tw->r5_q15_w1_re_inv;
  const __m128i *w1i = dir == DFT_DIR_FORWARD ? tw->r5_q15_w1_im : tw->r5_q15_w1_im_inv;
  const __m128i *w2r = dir == DFT_DIR_FORWARD ? tw->r5_q15_w2_re : tw->r5_q15_w2_re_inv;
  const __m128i *w2i = dir == DFT_DIR_FORWARD ? tw->r5_q15_w2_im : tw->r5_q15_w2_im_inv;
  const __m128i *w3r = dir == DFT_DIR_FORWARD ? tw->r5_q15_w3_re : tw->r5_q15_w3_re_inv;
  const __m128i *w3i = dir == DFT_DIR_FORWARD ? tw->r5_q15_w3_im : tw->r5_q15_w3_im_inv;
  const __m128i *w4r = dir == DFT_DIR_FORWARD ? tw->r5_q15_w4_re : tw->r5_q15_w4_re_inv;
  const __m128i *w4i = dir == DFT_DIR_FORWARD ? tw->r5_q15_w4_im : tw->r5_q15_w4_im_inv;

  for (int k = 0; k < M; k += 4) {
    const int b = k >> 2;
    __m128i y0, y1, y2, y3, y4;
    radix5_combine4_q15_128_fast(_mm_load_si128((const __m128i *)(tmp + 0 * M + k)),
                                 _mm_load_si128((const __m128i *)(tmp + 1 * M + k)),
                                 _mm_load_si128((const __m128i *)(tmp + 2 * M + k)),
                                 _mm_load_si128((const __m128i *)(tmp + 3 * M + k)),
                                 _mm_load_si128((const __m128i *)(tmp + 4 * M + k)),
                                 w1r[b], w1i[b], w2r[b], w2i[b],
                                 w3r[b], w3i[b], w4r[b], w4i[b],
                                 &y0, &y1, &y2, &y3, &y4, dir);
    _mm_storeu_si128((__m128i *)(dst + 0 * M + k), y0);
    _mm_storeu_si128((__m128i *)(dst + 1 * M + k), y1);
    _mm_storeu_si128((__m128i *)(dst + 2 * M + k), y2);
    _mm_storeu_si128((__m128i *)(dst + 3 * M + k), y3);
    _mm_storeu_si128((__m128i *)(dst + 4 * M + k), y4);
  }
}

static inline int radix25_selected_size(int N)
{
  return N == 300 || N == 600;
}

static inline int radix3_r25_selected_size(int N)
{
  switch (N) {
    case 1200:
    case 2400:
      return 1;
    default:
      return 0;
  }
}

static inline void radix25_compute_leaves(const c16_t *src,
                                                  int stride,
                                                  c16_t *leaf,
                                                  int M,
                                                  dft_dir_t dir)
{
#define R25_LEAF_LOOP(FN)                                              \
  do {                                                                 \
    for (int r1 = 0; r1 < 5; r1++)                                   \
      for (int r0 = 0; r0 < 5; r0++) {                               \
        const int br = r0 + 5 * r1;                                  \
        FN(src + br * stride, stride * 25, leaf + br * M, dir);       \
      }                                                                \
  } while (0)

  switch (M) {
    case 8:
      R25_LEAF_LOOP(dft8_strided_q15_128);
      break;
    case 12:
      R25_LEAF_LOOP(dft12_q15_128_strided);
      break;
    case 16:
      R25_LEAF_LOOP(dft16_q15_128_strided);
      break;
    case 24:
      R25_LEAF_LOOP(dft24_q15_128_strided);
      break;
    case 32:
      R25_LEAF_LOOP(dft32_q15_128_strided);
      break;
    default:
      for (int r1 = 0; r1 < 5; r1++)
        for (int r0 = 0; r0 < 5; r0++) {
          const int br = r0 + 5 * r1;
          dft_mixed_radix_c16_scaled_strided(src + br * stride, stride * 25, leaf + br * M, M, dir);
        }
      break;
  }

#undef R25_LEAF_LOOP
}

/* Fuse the two radix-5 levels that the recursive x86 path would execute for
 * N=25*M.  The 25 terminal DFT_M results are materialized once.  For each
 * group of four frequency bins, the intermediate 5x5 radix result stays in a
 * small local vector tile and feeds the second radix immediately, avoiding a
 * full-N intermediate store/reload between the two radix-5 stages. */
static void radix25_selected_strided(const c16_t *src, int stride, c16_t *dst, int N, dft_dir_t dir)
{
  enum { RADIX25_MAX_N = 800 };
  const int M = N / 25;
  AssertFatal((N % 25) == 0 && (M & 3) == 0 && N <= RADIX25_MAX_N,
              "Invalid selected radix-25 N=%d M=%d\n",
              N,
              M);

  c16_t leaf[N] __attribute__((aligned(64)));

  /* Branch numbering matches the two recursive radix-5 calls exactly:
   * input index = r0 + 5*r1 + 25*n. */
  radix25_compute_leaves(src, stride, leaf, M, dir);

  const radix25_plan_t *plan = radix25_plan_get(N);
  AssertFatal(plan, "Missing selected radix-25 plan N=%d\n", N);
  const r5_twiddle_t *tw_inner = r5_twiddle_slot(N / 5);
  const r5_twiddle_t *tw_outer = r5_twiddle_slot(N);

#define R25_R5_TW(TW, PREFIX)                                                                 \
  const __m128i *PREFIX##1r = dir == DFT_DIR_FORWARD ? (TW)->r5_q15_w1_re : (TW)->r5_q15_w1_re_inv; \
  const __m128i *PREFIX##1i = dir == DFT_DIR_FORWARD ? (TW)->r5_q15_w1_im : (TW)->r5_q15_w1_im_inv; \
  const __m128i *PREFIX##2r = dir == DFT_DIR_FORWARD ? (TW)->r5_q15_w2_re : (TW)->r5_q15_w2_re_inv; \
  const __m128i *PREFIX##2i = dir == DFT_DIR_FORWARD ? (TW)->r5_q15_w2_im : (TW)->r5_q15_w2_im_inv; \
  const __m128i *PREFIX##3r = dir == DFT_DIR_FORWARD ? (TW)->r5_q15_w3_re : (TW)->r5_q15_w3_re_inv; \
  const __m128i *PREFIX##3i = dir == DFT_DIR_FORWARD ? (TW)->r5_q15_w3_im : (TW)->r5_q15_w3_im_inv; \
  const __m128i *PREFIX##4r = dir == DFT_DIR_FORWARD ? (TW)->r5_q15_w4_re : (TW)->r5_q15_w4_re_inv; \
  const __m128i *PREFIX##4i = dir == DFT_DIR_FORWARD ? (TW)->r5_q15_w4_im : (TW)->r5_q15_w4_im_inv

  R25_R5_TW(tw_inner, in_);
  R25_R5_TW(tw_outer, out_);

#undef R25_R5_TW

  for (int k = 0; k < M; k += 4) {
    __m128i stage1[5][5];
    const int ib = k >> 2;

    /* Inner child radix-5: for each outer residue r0, combine the five r1
     * leaves exactly as the recursive DFT_(5*M) child would do. */
    for (int r0 = 0; r0 < 5; r0++) {
      const __m128i x0 = _mm_load_si128((const __m128i *)(leaf + (r0 + 0 * 5) * M + k));
      const __m128i x1 = _mm_load_si128((const __m128i *)(leaf + (r0 + 1 * 5) * M + k));
      const __m128i x2 = _mm_load_si128((const __m128i *)(leaf + (r0 + 2 * 5) * M + k));
      const __m128i x3 = _mm_load_si128((const __m128i *)(leaf + (r0 + 3 * 5) * M + k));
      const __m128i x4 = _mm_load_si128((const __m128i *)(leaf + (r0 + 4 * 5) * M + k));

      radix5_combine4_q15_128_fast(x0,
                                   x1,
                                   x2,
                                   x3,
                                   x4,
                                   in_1r[ib],
                                   in_1i[ib],
                                   in_2r[ib],
                                   in_2i[ib],
                                   in_3r[ib],
                                   in_3i[ib],
                                   in_4r[ib],
                                   in_4i[ib],
                                   &stage1[r0][0],
                                   &stage1[r0][1],
                                   &stage1[r0][2],
                                   &stage1[r0][3],
                                   &stage1[r0][4],
                                   dir);
    }

    /* Outer radix-5.  Its frequency index is j1*M+k, exactly matching the
     * parent combine in the original recursive implementation. */
    for (int j1 = 0; j1 < 5; j1++) {
      const int outer_k = j1 * M + k;
      const int ob = outer_k >> 2;
      __m128i y0, y1, y2, y3, y4;

      radix5_combine4_q15_128_fast(stage1[0][j1],
                                   stage1[1][j1],
                                   stage1[2][j1],
                                   stage1[3][j1],
                                   stage1[4][j1],
                                   out_1r[ob],
                                   out_1i[ob],
                                   out_2r[ob],
                                   out_2i[ob],
                                   out_3r[ob],
                                   out_3i[ob],
                                   out_4r[ob],
                                   out_4i[ob],
                                   &y0,
                                   &y1,
                                   &y2,
                                   &y3,
                                   &y4,
                                   dir);

      _mm_storeu_si128((__m128i *)(dst + (0 * 5 + j1) * M + k), y0);
      _mm_storeu_si128((__m128i *)(dst + (1 * 5 + j1) * M + k), y1);
      _mm_storeu_si128((__m128i *)(dst + (2 * 5 + j1) * M + k), y2);
      _mm_storeu_si128((__m128i *)(dst + (3 * 5 + j1) * M + k), y3);
      _mm_storeu_si128((__m128i *)(dst + (4 * 5 + j1) * M + k), y4);
    }
  }
}

/* DFT1200/2400 use R3 -> R25 with a materialized outer radix-3 boundary.
 * DFT600 uses R25 x DFT24, avoiding the 75 DFT8 leaf calls required by
 * an R3 -> R25 x DFT8 decomposition. */
static void radix3_r25_selected(const c16_t *src, int stride, c16_t *dst, int N, dft_dir_t dir)
{
  enum { R3_R25_MAX_N = 2400 };
  const int size = N / 3;
  AssertFatal(radix3_r25_selected_size(N) && (size == 400 || size == 800),
              "Invalid R3->R25 selected N=%d child=%d\n",
              N,
              size);

  c16_t tmp[N] __attribute__((aligned(64)));

  radix25_selected_strided(src + 0 * stride, stride * 3, tmp + 0 * size, size, dir);
  radix25_selected_strided(src + 1 * stride, stride * 3, tmp + 1 * size, size, dir);
  radix25_selected_strided(src + 2 * stride, stride * 3, tmp + 2 * size, size, dir);

  const radix3_selected_plan_t *r3plan = radix3_selected_plan_get(N);
  const r3_twiddle_t *tw = r3plan ? r3_twiddle_slot(N) : r3_twiddle_get(N);
  AssertFatal(tw, "Missing R3->R25 parent twiddles N=%d\n", N);
  const __m128i *w1r = dir == DFT_DIR_FORWARD ? tw->r3_q15_w1_re : tw->r3_q15_w1_re_inv;
  const __m128i *w1i = dir == DFT_DIR_FORWARD ? tw->r3_q15_w1_im : tw->r3_q15_w1_im_inv;
  const __m128i *w2r = dir == DFT_DIR_FORWARD ? tw->r3_q15_w2_re : tw->r3_q15_w2_re_inv;
  const __m128i *w2i = dir == DFT_DIR_FORWARD ? tw->r3_q15_w2_im : tw->r3_q15_w2_im_inv;

  for (int k = 0; k < size; k += 4) {
    const int b = k >> 2;
    __m128i y0, y1, y2;
    radix3_combine4_q15_128_fast(_mm_load_si128((const __m128i *)(tmp + 0 * size + k)),
                                 _mm_load_si128((const __m128i *)(tmp + 1 * size + k)),
                                 _mm_load_si128((const __m128i *)(tmp + 2 * size + k)),
                                 w1r[b],
                                 w1i[b],
                                 w2r[b],
                                 w2i[b],
                                 &y0,
                                 &y1,
                                 &y2,
                                 dir);
    _mm_storeu_si128((__m128i *)(dst + 0 * size + k), y0);
    _mm_storeu_si128((__m128i *)(dst + 1 * size + k), y1);
    _mm_storeu_si128((__m128i *)(dst + 2 * size + k), y2);
  }
}


/* -------------------------------------------------------------------------
 * Normalized leaves with all normalization carried by twiddle coefficients.
 *
 * R30 parents for N=360/600/720 carry 1/sqrt(30) in W_N.
 * Their leaf transforms carry 1/sqrt(M) in the leaf's own Cooley-Tukey
 * twiddles.  No whole-transform 1/sqrt(N) is applied before the leaf.
 * ------------------------------------------------------------------------- */

static const int16_t dft12_parent_leaf_twiddle_re[12] = {9459, 8192, 4730, 0, -4730, -8192, -9459, -8192, -4730, 0, 4730, 8192};
static const int16_t dft12_parent_leaf_twiddle_im_fwd[12] = {0, -4730, -8192, -9459, -8192, -4730, 0, 4730, 8192, 9459, 8192, 4730};
static const int16_t dft20_parent_leaf_twiddle_re[20] = {7327, 6968, 5928, 4307, 2264, 0, -2264, -4307, -5928, -6968, -7327, -6968, -5928, -4307, -2264, 0, 2264, 4307, 5928, 6968};
static const int16_t dft20_parent_leaf_twiddle_im_fwd[20] = {0, -2264, -4307, -5928, -6968, -7327, -6968, -5928, -4307, -2264, 0, 2264, 4307, 5928, 6968, 7327, 6968, 5928, 4307, 2264};
static const int16_t dft24_parent_leaf_twiddle_re[24] = {6689, 6461, 5792, 4730, 3344, 1731, 0, -1731, -3344, -4730, -5792, -6461, -6689, -6461, -5792, -4730, -3344, -1731, 0, 1731, 3344, 4730, 5792, 6461};
static const int16_t dft24_parent_leaf_twiddle_im_fwd[24] = {0, -1731, -3344, -4730, -5792, -6461, -6689, -6461, -5792, -4730, -3344, -1731, 0, 1731, 3344, 4730, 5792, 6461, 6689, 6461, 5792, 4730, 3344, 1731};

static __attribute__((always_inline)) inline __m256i
parent_leaf_twiddle_mul_q15_256(__m256i x, const int16_t *wr, const int16_t *wi, int N, int power, dft_dir_t dir)
{
  power %= N;
  if (power < 0) power += N;
  return complex_mul8_bcast_q15_256_parent(x, wr[power],
                                            twiddle_im_scalar_dir_i16(wi[power], dir));
}

/* 12 = 4 x 3 Cooley-Tukey.
 * DFT3 is raw. W12 carries the complete 1/sqrt(12) leaf normalization.
 * The final DFT4 is raw. Output k = k2 + 3*k1. */
static __attribute__((always_inline)) inline void
dft12x8_parent_twiddle_scaled_store(const c16_t *b, int M, c16_t *dst,
                                  int R, int first, int valid, dft_dir_t dir)
{
  __m256i a[4][3];

  for (int n1 = 0; n1 < 4; n1++) {
    __m256i z0, z1, z2;
    radix3_butterfly8_q15_256(
        gather8_leaf_c16_n360(b + first * M, M, n1 + 0),
        gather8_leaf_c16_n360(b + first * M, M, n1 + 4),
        gather8_leaf_c16_n360(b + first * M, M, n1 + 8),
        &z0, &z1, &z2, dir);
    a[n1][0] = parent_leaf_twiddle_mul_q15_256(z0, dft12_parent_leaf_twiddle_re, dft12_parent_leaf_twiddle_im_fwd, 12, 0,       dir);
    a[n1][1] = parent_leaf_twiddle_mul_q15_256(z1, dft12_parent_leaf_twiddle_re, dft12_parent_leaf_twiddle_im_fwd, 12, n1,      dir);
    a[n1][2] = parent_leaf_twiddle_mul_q15_256(z2, dft12_parent_leaf_twiddle_re, dft12_parent_leaf_twiddle_im_fwd, 12, 2 * n1,  dir);
  }

  __m256i y[12];
  for (int k2 = 0; k2 < 3; k2++) {
    __m256i z[4];
    raw_dft4_lane8_n360(a[0][k2], a[1][k2], a[2][k2], a[3][k2], z, dir);
    for (int k1 = 0; k1 < 4; k1++)
      y[k2 + 3 * k1] = z[k1];
  }

  for (int k = 0; k < 12; k++)
    store8_leaf_row_n360(dst, R, k, first, valid, y[k]);
}

/* 20 = 5 x 4 Cooley-Tukey.
 * First DFT4 raw, W20/sqrt(20), final DFT5 raw.
 * Output k = k2 + 4*k1. */
static __attribute__((always_inline)) inline void
dft20x8_parent_twiddle_scaled_store(const c16_t *b, int M, c16_t *dst,
                                  int R, int first, int valid, dft_dir_t dir)
{
  __m256i a[5][4];

  for (int n1 = 0; n1 < 5; n1++) {
    __m256i z[4];
    raw_dft4_lane8_n600(
        gather8_leaf_c16_n600(b + first * M, M, n1 + 0),
        gather8_leaf_c16_n600(b + first * M, M, n1 + 5),
        gather8_leaf_c16_n600(b + first * M, M, n1 + 10),
        gather8_leaf_c16_n600(b + first * M, M, n1 + 15),
        z, dir);
    for (int k2 = 0; k2 < 4; k2++)
      a[n1][k2] = parent_leaf_twiddle_mul_q15_256(z[k2], dft20_parent_leaf_twiddle_re,
                                          dft20_parent_leaf_twiddle_im_fwd, 20, n1 * k2, dir);
  }

  __m256i y[20];
  for (int k2 = 0; k2 < 4; k2++) {
    __m256i z0, z1, z2, z3, z4;
    radix5_butterfly8_q15_256(a[0][k2], a[1][k2], a[2][k2], a[3][k2], a[4][k2],
                               &z0, &z1, &z2, &z3, &z4, dir);
    y[k2 + 4 * 0] = z0;
    y[k2 + 4 * 1] = z1;
    y[k2 + 4 * 2] = z2;
    y[k2 + 4 * 3] = z3;
    y[k2 + 4 * 4] = z4;
  }

  for (int k = 0; k < 20; k++)
    store8_leaf_row_n600(dst, R, k, first, valid, y[k]);
}

/* 24 = 8 x 3 Cooley-Tukey.
 * First DFT3 raw, W24/sqrt(24), final DFT8 raw.
 * Output k = k2 + 3*k1. */
static __attribute__((always_inline)) inline void
dft24x8_parent_twiddle_scaled_store(const c16_t *b, int M, c16_t *dst,
                                  int R, int first, int valid, dft_dir_t dir)
{
  __m256i a[8][3];

  for (int n1 = 0; n1 < 8; n1++) {
    __m256i z0, z1, z2;
    radix3_butterfly8_q15_256(
        gather8_leaf_c16_n720(b + first * M, M, n1 + 0),
        gather8_leaf_c16_n720(b + first * M, M, n1 + 8),
        gather8_leaf_c16_n720(b + first * M, M, n1 + 16),
        &z0, &z1, &z2, dir);
    a[n1][0] = parent_leaf_twiddle_mul_q15_256(z0, dft24_parent_leaf_twiddle_re, dft24_parent_leaf_twiddle_im_fwd, 24, 0,       dir);
    a[n1][1] = parent_leaf_twiddle_mul_q15_256(z1, dft24_parent_leaf_twiddle_re, dft24_parent_leaf_twiddle_im_fwd, 24, n1,      dir);
    a[n1][2] = parent_leaf_twiddle_mul_q15_256(z2, dft24_parent_leaf_twiddle_re, dft24_parent_leaf_twiddle_im_fwd, 24, 2 * n1,  dir);
  }

  __m256i y[24];
  for (int k2 = 0; k2 < 3; k2++) {
    __m256i z0,z1,z2,z3,z4,z5,z6,z7;
    dft8x8_q15_256_dir(a[0][k2], a[1][k2], a[2][k2], a[3][k2],
                        a[4][k2], a[5][k2], a[6][k2], a[7][k2],
                        &z0,&z1,&z2,&z3,&z4,&z5,&z6,&z7, dir);
    y[k2 + 3 * 0] = z0;
    y[k2 + 3 * 1] = z1;
    y[k2 + 3 * 2] = z2;
    y[k2 + 3 * 3] = z3;
    y[k2 + 3 * 4] = z4;
    y[k2 + 3 * 5] = z5;
    y[k2 + 3 * 6] = z6;
    y[k2 + 3 * 7] = z7;
  }

  for (int k = 0; k < 24; k++)
    store8_leaf_row_n720(dst, R, k, first, valid, y[k]);
}

static void dft360_radix30_leaf12_twscaled_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { R=30, M=12 };
  c16_t b[32*M] __attribute__((aligned(64)));
  memset(b, 0, sizeof(b));
  radix30_stage_360(src, b, dir);
  dft12x8_parent_twiddle_scaled_store(b,M,dst,R,0,8,dir);
  dft12x8_parent_twiddle_scaled_store(b,M,dst,R,8,8,dir);
  dft12x8_parent_twiddle_scaled_store(b,M,dst,R,16,8,dir);
  dft12x8_parent_twiddle_scaled_store(b,M,dst,R,24,6,dir);
}

static void dft600_radix30_leaf20_twscaled_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { R=30, M=20 };
  c16_t b[32*M] __attribute__((aligned(64)));
  memset(b, 0, sizeof(b));
  radix30_stage_600(src, b, dir);
  dft20x8_parent_twiddle_scaled_store(b,M,dst,R,0,8,dir);
  dft20x8_parent_twiddle_scaled_store(b,M,dst,R,8,8,dir);
  dft20x8_parent_twiddle_scaled_store(b,M,dst,R,16,8,dir);
  dft20x8_parent_twiddle_scaled_store(b,M,dst,R,24,6,dir);
}

static void dft720_radix30_leaf24_twscaled_avx2_selected(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { R=30, M=24 };
  c16_t b[32*M] __attribute__((aligned(64)));
  memset(b, 0, sizeof(b));
  radix30_stage_720(src, b, dir);
  dft24x8_parent_twiddle_scaled_store(b,M,dst,R,0,8,dir);
  dft24x8_parent_twiddle_scaled_store(b,M,dst,R,8,8,dir);
  dft24x8_parent_twiddle_scaled_store(b,M,dst,R,16,8,dir);
  dft24x8_parent_twiddle_scaled_store(b,M,dst,R,24,6,dir);
}

/* Large R24 transforms.
 *
 * 3072  = R24 x DFT128x8
 * 6144  = R24 x DFT256x8
 * 12288 = R24 x DFT512x8
 *
 * The complete R24 normalization is carried by the final parent W_N
 * coefficients (1/sqrt(24)); each child transform carries its own
 * normalization. No whole-N final scatter buffer is used. */

/* DFT3072 keeps only the six final DFT64x8 inputs. The R24 results at child
 * indices k and k+64 are paired in registers and immediately passed through
 * the DFT128 R2; no R24 parent result is materialized in the working sets. */
static pthread_mutex_t dft3072_r24_mtx = PTHREAD_MUTEX_INITIALIZER;
static int dft3072_r24_ready;
static __m256i dft3072_r24_re[16][23] __attribute__((aligned(64)));
static __m256i dft3072_r24_im[2][16][23] __attribute__((aligned(64)));
static uint32_t dft3072_r2_re[64] __attribute__((aligned(64)));
static uint32_t dft3072_r2_im[2][64] __attribute__((aligned(64)));

static __attribute__((noinline, cold)) void dft3072_r24_init_slow(void)
{
  pthread_mutex_lock(&dft3072_r24_mtx);
  if (__atomic_load_n(&dft3072_r24_ready, __ATOMIC_RELAXED)) {
    pthread_mutex_unlock(&dft3072_r24_mtx);
    return;
  }

  const float s24 = 1.0f / sqrtf(24.0f);
  for (int bl = 0; bl < 16; bl++) {
    const int off = 8 * bl;
    for (int br = 1; br < 24; br++) {
      dft3072_r24_re[bl][br - 1] = pack8_twiddle_q15_re_re_scaled(off, br, 3072, s24);
      dft3072_r24_im[0][bl][br - 1] =
          pack8_twiddle_q15_im_signed_scaled(off, br, 3072, s24, DFT_DIR_FORWARD);
      dft3072_r24_im[1][bl][br - 1] =
          pack8_twiddle_q15_im_signed_scaled(off, br, 3072, s24, DFT_DIR_INVERSE);
    }
  }

  const float s2 = sqrtf(2.0f);
  for (int k = 0; k < 64; k++) {
    const float theta = 2.0f * (float)M_PI * (float)k / 128.0f;
    const int16_t raw_re = q15_from_float(cosf(theta));
    const int16_t wr = (int16_t)(raw_re / s2);
    dft3072_r2_re[k] = (uint16_t)wr | ((uint32_t)(uint16_t)wr << 16);

    for (int ds = 0; ds < 2; ds++) {
      const dft_dir_t dir = ds == 0 ? DFT_DIR_FORWARD : DFT_DIR_INVERSE;
      const int16_t raw_im = q15_from_float((float)dir * sinf(theta));
      const int16_t wi = (int16_t)(raw_im / s2);
      dft3072_r2_im[ds][k] = (uint16_t)(-wi) | ((uint32_t)(uint16_t)wi << 16);
    }
  }

  __atomic_store_n(&dft3072_r24_ready, 1, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&dft3072_r24_mtx);
}

static __attribute__((always_inline)) inline void dft3072_r24_init(void)
{
  if (__builtin_expect(__atomic_load_n(&dft3072_r24_ready, __ATOMIC_ACQUIRE), 1))
    return;
  dft3072_r24_init_slow();
}

static inline void dft3072_r24_r2_to_batches(const c16_t *src,
                                              __m256i work[3][2][64],
                                              dft_dir_t dir)
{
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s24 = _mm256_set1_epi16(Q15_INV_SQRT24);
  const __m256i s2 = _mm256_set1_epi16(Q15_INV_SQRT2);
  dft3072_r24_init();

  for (int block = 0; block < 8; block++) {
    const int k0 = 8 * block;
    const __m256i r2_re = _mm256_load_si256((const __m256i *)(dft3072_r2_re + k0));
    const __m256i r2_im = _mm256_load_si256((const __m256i *)(dft3072_r2_im[ds] + k0));
    __m256i z[2][24];
    radix24_pfa_raw8_q15_256(src, 128, k0, z[0], dir);
    radix24_pfa_raw8_q15_256(src, 128, k0 + 64, z[1], dir);

    for (int group = 0; group < 3; group++) {
      __m256i row[2][8];
      const int first = 8 * group;
      for (int lane = 0; lane < 8; lane++) {
        const int br = first + lane;
        const __m256i x0 = br == 0
                               ? _mm256_mulhrs_epi16(z[0][0], s24)
                               : complex_mul8_prepack_q15_256(
                                     z[0][br], dft3072_r24_re[block][br - 1],
                                     dft3072_r24_im[ds][block][br - 1]);
        const __m256i x1 = br == 0
                               ? _mm256_mulhrs_epi16(z[1][0], s24)
                               : complex_mul8_prepack_q15_256(
                                     z[1][br], dft3072_r24_re[block + 8][br - 1],
                                     dft3072_r24_im[ds][block + 8][br - 1]);
        row[0][lane] = _mm256_mulhrs_epi16(_mm256_adds_epi16(x0, x1), s2);
        row[1][lane] = complex_mul8_prepack_q15_256(
            _mm256_subs_epi16(x0, x1), r2_re, r2_im);
      }
      transpose8_complex_i16_256(&row[0][0], &row[0][1], &row[0][2], &row[0][3],
                                 &row[0][4], &row[0][5], &row[0][6], &row[0][7]);
      transpose8_complex_i16_256(&row[1][0], &row[1][1], &row[1][2], &row[1][3],
                                 &row[1][4], &row[1][5], &row[1][6], &row[1][7]);
      for (int lane = 0; lane < 8; lane++) {
        work[group][0][k0 + lane] = row[0][lane];
        work[group][1][k0 + lane] = row[1][lane];
      }
    }
  }
}

static inline void dft3072_dft64x8_pair_store(__m256i x[2][64],
                                               c16_t *dst,
                                               int first,
                                               dft_dir_t dir)
{
  const __m256i even_dc = dft64x8_dc_q15_256(x[0]);
  const __m256i odd_dc = dft64x8_dc_q15_256(x[1]);
  dft64x8_selected_store(x[0], dst, 48, first, dir);
  dft64x8_selected_store(x[1], dst, 48, 24 + first, dir);
  _mm256_storeu_si256((__m256i *)(dst + first), even_dc);
  _mm256_storeu_si256((__m256i *)(dst + 24 + first), odd_dc);
}

static void dft3072_r24_dft128x8_direct(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  __m256i work[3][2][64] __attribute__((aligned(64)));
  selected_q15_twiddles_init();
  dft3072_r24_r2_to_batches(src, work, dir);
  for (int group = 0; group < 3; group++)
    dft3072_dft64x8_pair_store(work[group], dst, 8 * group, dir);
}

static __attribute__((always_inline)) inline void
r24_store8_child_outputs(const c16_t *const p[8], int M, c16_t *dst, int first)
{
  for (int k = 0; k < M; k += 8) {
    __m256i z0 = _mm256_loadu_si256((const __m256i *)(p[0] + k));
    __m256i z1 = _mm256_loadu_si256((const __m256i *)(p[1] + k));
    __m256i z2 = _mm256_loadu_si256((const __m256i *)(p[2] + k));
    __m256i z3 = _mm256_loadu_si256((const __m256i *)(p[3] + k));
    __m256i z4 = _mm256_loadu_si256((const __m256i *)(p[4] + k));
    __m256i z5 = _mm256_loadu_si256((const __m256i *)(p[5] + k));
    __m256i z6 = _mm256_loadu_si256((const __m256i *)(p[6] + k));
    __m256i z7 = _mm256_loadu_si256((const __m256i *)(p[7] + k));
    transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
    const __m256i z[8] = {z0,z1,z2,z3,z4,z5,z6,z7};
    for (int lane = 0; lane < 8; lane++)
      _mm256_storeu_si256((__m256i *)(dst + 24 * (k + lane) + first), z[lane]);
  }
}

#define DEFINE_R24_PARENT_TABLES(TAG,NVAL,BLOCKS) \
static pthread_mutex_t TAG##_mtx = PTHREAD_MUTEX_INITIALIZER; \
static int TAG##_ready; \
static __m256i TAG##_re[2][BLOCKS][24] __attribute__((aligned(64))); \
static __m256i TAG##_im[2][BLOCKS][24] __attribute__((aligned(64))); \
static void TAG##_init(void) { \
  if (__builtin_expect(__atomic_load_n(&TAG##_ready, __ATOMIC_ACQUIRE), 1)) return; \
  pthread_mutex_lock(&TAG##_mtx); \
  if (__atomic_load_n(&TAG##_ready, __ATOMIC_RELAXED)) { pthread_mutex_unlock(&TAG##_mtx); return; } \
  const float sc = 1.0f / sqrtf(24.0f); \
  for (int ds = 0; ds < 2; ds++) { \
    const dft_dir_t d = ds == 0 ? DFT_DIR_FORWARD : DFT_DIR_INVERSE; \
    for (int bl = 0; bl < BLOCKS; bl++) { \
      const int off = 8 * bl; \
      for (int br = 0; br < 24; br++) { \
        TAG##_re[ds][bl][br] = pack8_twiddle_q15_re_re_scaled(off, br, NVAL, sc); \
        TAG##_im[ds][bl][br] = pack8_twiddle_q15_im_signed_scaled(off, br, NVAL, sc, d); \
      } \
    } \
  } \
  __atomic_store_n(&TAG##_ready, 1, __ATOMIC_RELEASE); \
  pthread_mutex_unlock(&TAG##_mtx); \
}

#define DEFINE_R24_PARENT_STAGE(TAG,MVAL) \
static void TAG##_stage(const c16_t *src, c16_t *b, dft_dir_t dir) { \
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1; \
  TAG##_init(); \
  for (int off = 0; off < MVAL; off += 8) { \
    __m256i z[24]; \
    radix24_pfa_raw8_q15_256(src, MVAL, off, z, dir); \
    for (int br = 0; br < 24; br++) { \
      const __m256i v = br == 0 \
        ? _mm256_mulhrs_epi16(z[br], _mm256_set1_epi16(Q15_INV_SQRT24)) \
        : complex_mul8_prepack_q15_256(z[br], TAG##_re[ds][off >> 3][br], TAG##_im[ds][off >> 3][br]); \
      _mm256_store_si256((__m256i *)(b + br * (MVAL) + off), v); \
    } \
  } \
}

DEFINE_R24_PARENT_TABLES(r24_6144_parent, 6144, 32)
DEFINE_R24_PARENT_TABLES(r24_12288_parent, 12288, 64)
DEFINE_R24_PARENT_STAGE(r24_12288_parent, 512)

/* DFT6144-specific true DFT256x8 child.  A vector lane is one R24
 * branch, so the R16 parent and DFT16 leaves below advance eight
 * independent DFT256 transforms together. */
static pthread_mutex_t dft6144_leaf256_mtx = PTHREAD_MUTEX_INITIALIZER;
static int dft6144_leaf256_ready;
static uint32_t dft6144_leaf256_re[16][16] __attribute__((aligned(64)));
static uint32_t dft6144_leaf256_im[2][16][16] __attribute__((aligned(64)));

static __attribute__((noinline, cold)) void dft6144_leaf256_init_slow(void)
{
  pthread_mutex_lock(&dft6144_leaf256_mtx);
  if (__atomic_load_n(&dft6144_leaf256_ready, __ATOMIC_RELAXED)) {
    pthread_mutex_unlock(&dft6144_leaf256_mtx);
    return;
  }

  const float s8 = 1.0f / sqrtf(8.0f);
  for (int r = 0; r < 16; r++) {
    for (int n = 0; n < 16; n++) {
      const float theta = 2.0f * (float)M_PI * (float)(r * n) / 256.0f;
      const int16_t wr = q15_from_float(cosf(theta) * s8);
      dft6144_leaf256_re[r][n] = (uint16_t)wr | ((uint32_t)(uint16_t)wr << 16);
      for (int ds = 0; ds < 2; ds++) {
        const dft_dir_t d = ds == 0 ? DFT_DIR_FORWARD : DFT_DIR_INVERSE;
        const int16_t wi = q15_from_float((float)d * sinf(theta) * s8);
        dft6144_leaf256_im[ds][r][n] =
            (uint16_t)(-wi) | ((uint32_t)(uint16_t)wi << 16);
      }
    }
  }

  __atomic_store_n(&dft6144_leaf256_ready, 1, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&dft6144_leaf256_mtx);
}

static __attribute__((always_inline)) inline void dft6144_leaf256_init(void)
{
  if (__builtin_expect(__atomic_load_n(&dft6144_leaf256_ready, __ATOMIC_ACQUIRE), 1))
    return;
  dft6144_leaf256_init_slow();
}

static inline void dft6144_r24_parent_to_batches(const c16_t *src,
                                                  __m256i work[3][16][16],
                                                  dft_dir_t dir)
{
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s24 = _mm256_set1_epi16(Q15_INV_SQRT24);
  r24_6144_parent_init();

  for (int off = 0; off < 256; off += 8) {
    __m256i z[24];
    radix24_pfa_raw8_q15_256(src, 256, off, z, dir);

    for (int group = 0; group < 3; group++) {
      const int first = 8 * group;
      __m256i row[8];
      for (int lane = 0; lane < 8; lane++) {
        const int br = first + lane;
        row[lane] = br == 0
                        ? _mm256_mulhrs_epi16(z[0], s24)
                        : complex_mul8_prepack_q15_256(z[br],
                                                       r24_6144_parent_re[ds][off >> 3][br],
                                                       r24_6144_parent_im[ds][off >> 3][br]);
      }
      transpose8_complex_i16_256(&row[0], &row[1], &row[2], &row[3],
                                  &row[4], &row[5], &row[6], &row[7]);
      for (int lane = 0; lane < 8; lane++) {
        const int n = off + lane;
        work[group][n >> 4][n & 15] = row[lane];
      }
    }
  }
}

static inline void dft6144_dft256x8_store(__m256i stage[16][16],
                                           c16_t *dst,
                                           int first,
                                           dft_dir_t dir)
{
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  const __m256i s8 = _mm256_set1_epi16(Q15_INV_SQRT8);

  /* R16 parent.  This is the lane-parallel form of
   * dft256_radix16_selected(), including its exact Q15 rounding order. */
  for (int n = 0; n < 16; n++) {
    __m256i x[16], y[16];
    for (int q = 0; q < 16; q++)
      x[q] = stage[q][n];
    dft16x8_selected(x, y, dir);
    stage[0][n] = _mm256_mulhrs_epi16(y[0], s8);
    for (int r = 1; r < 16; r++) {
      stage[r][n] = complex_mul8_prepack_q15_256(
          y[r],
          _mm256_set1_epi32((int32_t)dft6144_leaf256_re[r][n]),
          _mm256_set1_epi32((int32_t)dft6144_leaf256_im[ds][r][n]));
    }
  }

  /* Sixteen DFT16 children; each YMM is still eight independent DFT256s. */
  for (int r = 0; r < 16; r++) {
    __m256i y[16];
    dft16x8_selected(stage[r], y, dir);
    for (int k = 0; k < 16; k++) {
      y[k] = _mm256_mulhrs_epi16(y[k], s8);
      _mm256_storeu_si256((__m256i *)(dst + 24 * (16 * k + r) + first), y[k]);
    }
  }
}

static void dft6144_r24_leaf256_direct(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  __m256i work[3][16][16] __attribute__((aligned(64)));
  selected_q15_twiddles_init();
  dft6144_leaf256_init();
  dft6144_r24_parent_to_batches(src, work, dir);
  for (int group = 0; group < 3; group++)
    dft6144_dft256x8_store(work[group], dst, 8 * group, dir);
}

/* A vector lane holds one of eight independent R24 children throughout the
 * R8 parent and its DFT64 leaves. Keeping that layout between the stages
 * avoids sequential DFT512 calls and a final 8x8 output transpose. */
static pthread_mutex_t dft12288_leaf512x8_mtx = PTHREAD_MUTEX_INITIALIZER;
static int dft12288_leaf512x8_ready;
static uint32_t dft12288_leaf512x8_re[8][64] __attribute__((aligned(64)));
static uint32_t dft12288_leaf512x8_im[2][8][64] __attribute__((aligned(64)));

static __attribute__((noinline, cold)) void dft12288_leaf512x8_init_slow(void)
{
  pthread_mutex_lock(&dft12288_leaf512x8_mtx);
  if (__atomic_load_n(&dft12288_leaf512x8_ready, __ATOMIC_RELAXED)) {
    pthread_mutex_unlock(&dft12288_leaf512x8_mtx);
    return;
  }

  const float s8 = 1.0f / sqrtf(8.0f);
  for (int r = 0; r < 8; r++) {
    for (int n = 0; n < 64; n++) {
      const float theta = 2.0f * (float)M_PI * (float)(r * n) / 512.0f;
      const int16_t wr = q15_from_float(cosf(theta) * s8);
      dft12288_leaf512x8_re[r][n] =
          (uint16_t)wr | ((uint32_t)(uint16_t)wr << 16);
      for (int ds = 0; ds < 2; ds++) {
        const dft_dir_t d = ds == 0 ? DFT_DIR_FORWARD : DFT_DIR_INVERSE;
        const int16_t wi = q15_from_float((float)d * sinf(theta) * s8);
        dft12288_leaf512x8_im[ds][r][n] =
            (uint16_t)(-wi) | ((uint32_t)(uint16_t)wi << 16);
      }
    }
  }

  __atomic_store_n(&dft12288_leaf512x8_ready, 1, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&dft12288_leaf512x8_mtx);
}

static __attribute__((always_inline)) inline void dft12288_leaf512x8_init(void)
{
  if (__builtin_expect(__atomic_load_n(&dft12288_leaf512x8_ready, __ATOMIC_ACQUIRE), 1))
    return;
  dft12288_leaf512x8_init_slow();
}

static void dft12288_leaf512x8_store(const c16_t *b,
                                     c16_t *dst,
                                     int first,
                                     dft_dir_t dir)
{
  enum { R = 24, M = 512, LEAF_M = 64 };
  const int ds = dir == DFT_DIR_FORWARD ? 0 : 1;
  __m256i leaf[8][LEAF_M] __attribute__((aligned(64)));

  selected_q15_twiddles_init();
  dft12288_leaf512x8_init();

  for (int n0 = 0; n0 < LEAF_M; n0 += 8) {
    __m256i input[8][8] __attribute__((aligned(64)));

    for (int q = 0; q < 8; q++) {
      __m256i z0 = _mm256_loadu_si256((const __m256i *)(b + (first + 0) * M + q * LEAF_M + n0));
      __m256i z1 = _mm256_loadu_si256((const __m256i *)(b + (first + 1) * M + q * LEAF_M + n0));
      __m256i z2 = _mm256_loadu_si256((const __m256i *)(b + (first + 2) * M + q * LEAF_M + n0));
      __m256i z3 = _mm256_loadu_si256((const __m256i *)(b + (first + 3) * M + q * LEAF_M + n0));
      __m256i z4 = _mm256_loadu_si256((const __m256i *)(b + (first + 4) * M + q * LEAF_M + n0));
      __m256i z5 = _mm256_loadu_si256((const __m256i *)(b + (first + 5) * M + q * LEAF_M + n0));
      __m256i z6 = _mm256_loadu_si256((const __m256i *)(b + (first + 6) * M + q * LEAF_M + n0));
      __m256i z7 = _mm256_loadu_si256((const __m256i *)(b + (first + 7) * M + q * LEAF_M + n0));
      transpose8_complex_i16_256(&z0, &z1, &z2, &z3, &z4, &z5, &z6, &z7);
      input[q][0] = z0; input[q][1] = z1; input[q][2] = z2; input[q][3] = z3;
      input[q][4] = z4; input[q][5] = z5; input[q][6] = z6; input[q][7] = z7;
    }

    for (int lane = 0; lane < 8; lane++) {
      const int n = n0 + lane;
      __m256i h[8];
      dft8x8_q15_256_dir(input[0][lane], input[1][lane], input[2][lane], input[3][lane],
                         input[4][lane], input[5][lane], input[6][lane], input[7][lane],
                         &h[0], &h[1], &h[2], &h[3], &h[4], &h[5], &h[6], &h[7], dir);
      for (int r = 0; r < 8; r++)
        leaf[r][n] = complex_mul8_prepack_q15_256(
            h[r],
            _mm256_set1_epi32((int32_t)dft12288_leaf512x8_re[r][n]),
            _mm256_set1_epi32((int32_t)dft12288_leaf512x8_im[ds][r][n]));
    }
  }

  for (int r = 0; r < 8; r++)
    dft64x8_selected_store(leaf[r], dst, R * 8, first + R * r, dir);
}

static void dft12288_r24_leaf512_direct(const c16_t *src, c16_t *dst, dft_dir_t dir)
{
  enum { R = 24, M = 512, N = R * M };
  c16_t b[N] __attribute__((aligned(64)));
  r24_12288_parent_stage(src, b, dir);
  for (int first = 0; first < R; first += 8)
    dft12288_leaf512x8_store(b, dst, first, dir);
}

#undef DEFINE_R24_PARENT_STAGE
#undef DEFINE_R24_PARENT_TABLES

static void dft_mixed_radix_c16_scaled_strided(const c16_t *src, int stride, c16_t *dst, int N, dft_dir_t dir)
{
  if (N == 1) {
    dst[0] = src[0];
    return;
  }

  if (N == 4) {
    dft4_void(src, dst, dir);
    return;
  }

  if (N == 8) {
    dft8_strided_q15_128(src, stride, dst, dir);
    return;
  }

  if (N == 12) {
    dft12_q15_128_strided(src, stride, dst, dir);
    return;
  }

  if (N == 16) {
    dft16_q15_128_strided(src, stride, dst, dir);
    return;
  }

  if (N == 20) {
    dft20_q15_128_strided(src, stride, dst, dir);
    return;
  }

  if (N == 24) {
    if (stride == 1)
      radix3_pow2_selected(src, dst, N, dir);
    else
      dft24_q15_128_strided(src, stride, dst, dir);
    return;
  }

  if (N == 32) {
    dft32_q15_128_strided(src, stride, dst, dir);
    return;
  }

  if (N == 64) {
    dft64_q15_128_strided(src, stride, dst, dir);
    return;
  }

  if (N == 128) {
    dft128_q15_128_strided(src, stride, dst, dir);
    return;
  }

  /* Contiguous size-specific kernels precede the generic family dispatch
   * because their parent/leaf layouts are specialized for the full size. */
  if (stride == 1 && N == 600) {
    dft600_radix30_leaf20_twscaled_avx2_selected(src, dst, dir);
    return;
  }


  if (stride == 1 && N == 1536) {
    dft1536_radix24_pfa_leaf64(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 3072) {
    dft3072_r24_dft128x8_direct(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 1440) {
    dft1440_r5_leaf288_direct(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 6144) {
    dft6144_r24_leaf256_direct(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 12288 && src != dst) {
    dft12288_r24_leaf512_direct(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 108) {
    radix9_terminal_leaf12_direct(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 72) {
    radix9_terminal_leaf8_direct(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 144) {
    dft144_radix18_leaf8_avx2_selected(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 648) {
    radix81_terminal_leaf8_direct(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 1296) {
    radix81_terminal_leaf16_direct_w16folded(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 2592) {
    radix81_terminal_leaf32_direct_2592(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 972) {
    radix81_terminal_leaf12_direct_972(src, dst, dir);
    return;
  }

  if (stride == 1 && radix81_selected_size(N)) {
    radix81_selected(src, dst, N, dir);
    return;
  }

  if (stride == 1 && selected_radix9_radix18_size(N)) {
    dispatch_selected_radix9_radix18(src, dst, N, dir);
    return;
  }

  if (stride == 1 && selected_radix15_radix30_size(N)) {
    dispatch_selected_radix15_radix30(src, dst, N, dir);
    return;
  }

  if (stride == 1 && radix3_r25_selected_size(N)) {
    radix3_r25_selected(src, stride, dst, N, dir);
    return;
  }

  if (stride == 1 && radix25_selected_size(N)) {
    radix25_selected_strided(src, stride, dst, N, dir);
    return;
  }

  if (stride == 1 && N == 96) {
    dft96_radix12_pfa_leaf8(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 192) {
    dft192_radix12_pfa_leaf16(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 216) {
    dft216_radix27_leaf8_avx2_selected(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 360) {
    dft360_radix30_leaf12_twscaled_avx2_selected(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 384) {
    dft384_radix24_pfa_avx2_selected(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 432) {
    dft432_radix27_leaf16_avx2_selected(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 720) {
    dft720_radix30_leaf24_twscaled_avx2_selected(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 864) {
    dft864_radix27_leaf32_avx2_selected(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 1728) {
    dft1728_radix27_leaf64_avx2_selected(src, dst, dir);
    return;
  }

  if (stride == 1 && N == 2160) {
    dft2160_radix30_leaf72x8_direct(src, dst, dir);
    return;
  }

  if (N % 5 == 0) {
    radix_5_fft_c16_scaled_strided(src, stride, dst, N, dir);
    return;
  }

  if (N % 3 == 0) {
    if (stride == 1 && is_power_of_two_int(N / 3))
      radix3_pow2_selected(src, dst, N, dir);
    else
      radix_3_fft_c16_scaled_strided(src, stride, dst, N, dir);
    return;
  }

  if (is_power_of_two_int(N) && N >= 256) {
    dft_power2_mixed_large_q15_strided(src, stride, dst, N, dir);
    return;
  }

  c16_t *tmp = x86_dft_recursive_work_acquire((size_t)N);
  if (!tmp) {
    LOG_E(PHY, "Mixed-radix workspace allocation failed N=%d\n", N);
    return;
  }

  for (int i = 0; i < N; i++)
    tmp[i] = src[i * stride];

  dft_mixed_radix_c16_scaled(tmp, dst, N, dir);
  x86_dft_recursive_work_release();
}

static void dft_mixed_radix_c16_scaled(const c16_t *src, c16_t *dst, int N, dft_dir_t dir)
{
  if (is_power_of_two_int(N) && N >= 256) {
    dft_power2_mixed_large_q15(src, dst, N, dir);
    return;
  }
  dft_mixed_radix_c16_scaled_strided(src, 1, dst, N, dir);
}
#define DEFINE_MIXED_DFT_ONLY(N)                                                     \
  void dft##N(int16_t *input, int16_t *output, uint8_t scale_flag)                   \
  {                                                                                  \
    (void)scale_flag;                                                                \
                                                                                     \
    dft_mixed_radix_c16_scaled((c16_t *)input, (c16_t *)output, N, DFT_DIR_FORWARD); \
  }

#define DEFINE_MIXED_IDFT_ONLY(N)                                                    \
  void idft##N(int16_t *input, int16_t *output, uint8_t scale_flag)                  \
  {                                                                                  \
    (void)scale_flag;                                                                \
                                                                                     \
    dft_mixed_radix_c16_scaled((c16_t *)input, (c16_t *)output, N, DFT_DIR_INVERSE); \
  }

DEFINE_MIXED_IDFT_ONLY(4)
DEFINE_MIXED_IDFT_ONLY(8)
OAI_DFT_SMALL_HOT void idft12(int16_t *input, int16_t *output, uint8_t scale_flag)
{
  (void)scale_flag;
  dft12_q15_128((const c16_t *)input, (c16_t *)output, DFT_DIR_INVERSE);
}

OAI_DFT_SMALL_HOT void idft16(int16_t *input, int16_t *output, uint8_t scale_flag)
{
  (void)scale_flag;
  dft16_q15_128((const c16_t *)input, (c16_t *)output, DFT_DIR_INVERSE);
}
DEFINE_MIXED_IDFT_ONLY(20)
DEFINE_MIXED_IDFT_ONLY(24)
DEFINE_MIXED_IDFT_ONLY(32)

DEFINE_MIXED_DFT_ONLY(4)
DEFINE_MIXED_DFT_ONLY(8)

DEFINE_MIXED_DFT_ONLY(192)
DEFINE_MIXED_DFT_ONLY(384)
DEFINE_MIXED_DFT_ONLY(768)
DEFINE_MIXED_DFT_ONLY(1536)
DEFINE_MIXED_DFT_ONLY(3072)
DEFINE_MIXED_DFT_ONLY(6144)
DEFINE_MIXED_DFT_ONLY(12288)
DEFINE_MIXED_DFT_ONLY(64)
DEFINE_MIXED_DFT_ONLY(128)
DEFINE_MIXED_DFT_ONLY(256)
DEFINE_MIXED_DFT_ONLY(512)
DEFINE_MIXED_DFT_ONLY(1024)
DEFINE_MIXED_DFT_ONLY(2048)
DEFINE_MIXED_DFT_ONLY(4096)
DEFINE_MIXED_DFT_ONLY(8192)
DEFINE_MIXED_DFT_ONLY(16384)

DEFINE_MIXED_IDFT_ONLY(64)
DEFINE_MIXED_IDFT_ONLY(128)
DEFINE_MIXED_IDFT_ONLY(256)
DEFINE_MIXED_IDFT_ONLY(512)
DEFINE_MIXED_IDFT_ONLY(1024)
DEFINE_MIXED_IDFT_ONLY(2048)
DEFINE_MIXED_IDFT_ONLY(4096)
DEFINE_MIXED_IDFT_ONLY(8192)
DEFINE_MIXED_IDFT_ONLY(16384)
DEFINE_MIXED_IDFT_ONLY(192)
DEFINE_MIXED_IDFT_ONLY(384)
DEFINE_MIXED_IDFT_ONLY(768)
DEFINE_MIXED_IDFT_ONLY(1536)
DEFINE_MIXED_IDFT_ONLY(3072)
DEFINE_MIXED_IDFT_ONLY(6144)
DEFINE_MIXED_IDFT_ONLY(12288)

DEFINE_MIXED_DFT_ONLY(32768)

DEFINE_MIXED_IDFT_ONLY(32768)

DEFINE_MIXED_DFT_ONLY(18432)

DEFINE_MIXED_IDFT_ONLY(18432)

DEFINE_MIXED_DFT_ONLY(24576)

DEFINE_MIXED_IDFT_ONLY(24576)

DEFINE_MIXED_DFT_ONLY(36864)

DEFINE_MIXED_IDFT_ONLY(36864)

DEFINE_MIXED_DFT_ONLY(49152)

DEFINE_MIXED_IDFT_ONLY(49152)

DEFINE_MIXED_DFT_ONLY(65536)

DEFINE_MIXED_IDFT_ONLY(65536)

DEFINE_MIXED_DFT_ONLY(98304)

DEFINE_MIXED_IDFT_ONLY(98304)

DEFINE_MIXED_DFT_ONLY(36)

DEFINE_MIXED_IDFT_ONLY(36)

DEFINE_MIXED_DFT_ONLY(48)

DEFINE_MIXED_IDFT_ONLY(48)
DEFINE_MIXED_DFT_ONLY(60)

DEFINE_MIXED_IDFT_ONLY(60)

DEFINE_MIXED_DFT_ONLY(72)

DEFINE_MIXED_IDFT_ONLY(72)

DEFINE_MIXED_DFT_ONLY(96)

DEFINE_MIXED_IDFT_ONLY(96)

DEFINE_MIXED_DFT_ONLY(108)

DEFINE_MIXED_IDFT_ONLY(108)

DEFINE_MIXED_DFT_ONLY(120)

DEFINE_MIXED_IDFT_ONLY(120)

DEFINE_MIXED_DFT_ONLY(144)

DEFINE_MIXED_IDFT_ONLY(144)

DEFINE_MIXED_DFT_ONLY(180)

DEFINE_MIXED_IDFT_ONLY(180)

DEFINE_MIXED_DFT_ONLY(216)

DEFINE_MIXED_IDFT_ONLY(216)

DEFINE_MIXED_DFT_ONLY(240)

DEFINE_MIXED_IDFT_ONLY(240)

void dft288(int16_t *input, int16_t *output, uint8_t scale_flag)
{
  (void)scale_flag;
  radix9_terminal_leaf32_direct((const c16_t *)input, (c16_t *)output, DFT_DIR_FORWARD);
}

void idft288(int16_t *input, int16_t *output, uint8_t scale_flag)
{
  (void)scale_flag;
  radix9_terminal_leaf32_direct((const c16_t *)input, (c16_t *)output, DFT_DIR_INVERSE);
}

DEFINE_MIXED_DFT_ONLY(300)

DEFINE_MIXED_IDFT_ONLY(300)

DEFINE_MIXED_DFT_ONLY(324)

DEFINE_MIXED_IDFT_ONLY(324)

DEFINE_MIXED_DFT_ONLY(360)

DEFINE_MIXED_IDFT_ONLY(360)

DEFINE_MIXED_DFT_ONLY(432)

DEFINE_MIXED_IDFT_ONLY(432)

DEFINE_MIXED_DFT_ONLY(480)

DEFINE_MIXED_IDFT_ONLY(480)

DEFINE_MIXED_DFT_ONLY(540)

DEFINE_MIXED_IDFT_ONLY(540)

DEFINE_MIXED_DFT_ONLY(576)

DEFINE_MIXED_IDFT_ONLY(576)

DEFINE_MIXED_DFT_ONLY(600)

DEFINE_MIXED_IDFT_ONLY(600)

DEFINE_MIXED_DFT_ONLY(648)

DEFINE_MIXED_IDFT_ONLY(648)

DEFINE_MIXED_DFT_ONLY(720)

DEFINE_MIXED_IDFT_ONLY(720)

DEFINE_MIXED_DFT_ONLY(864)

DEFINE_MIXED_IDFT_ONLY(864)

DEFINE_MIXED_DFT_ONLY(900)

DEFINE_MIXED_IDFT_ONLY(900)

DEFINE_MIXED_DFT_ONLY(960)

DEFINE_MIXED_IDFT_ONLY(960)

DEFINE_MIXED_DFT_ONLY(972)

DEFINE_MIXED_IDFT_ONLY(972)

DEFINE_MIXED_DFT_ONLY(1080)

DEFINE_MIXED_IDFT_ONLY(1080)

DEFINE_MIXED_DFT_ONLY(1152)

DEFINE_MIXED_IDFT_ONLY(1152)

DEFINE_MIXED_DFT_ONLY(1200)

DEFINE_MIXED_IDFT_ONLY(1200)

DEFINE_MIXED_DFT_ONLY(1296)

DEFINE_MIXED_IDFT_ONLY(1296)

DEFINE_MIXED_DFT_ONLY(1440)

DEFINE_MIXED_IDFT_ONLY(1440)

DEFINE_MIXED_DFT_ONLY(1500)

DEFINE_MIXED_IDFT_ONLY(1500)

DEFINE_MIXED_DFT_ONLY(1620)

DEFINE_MIXED_IDFT_ONLY(1620)

DEFINE_MIXED_DFT_ONLY(1728)

DEFINE_MIXED_IDFT_ONLY(1728)

DEFINE_MIXED_DFT_ONLY(1800)

DEFINE_MIXED_IDFT_ONLY(1800)

DEFINE_MIXED_DFT_ONLY(1920)

DEFINE_MIXED_IDFT_ONLY(1920)

DEFINE_MIXED_DFT_ONLY(1944)

DEFINE_MIXED_IDFT_ONLY(1944)

DEFINE_MIXED_DFT_ONLY(2160)

DEFINE_MIXED_IDFT_ONLY(2160)

DEFINE_MIXED_DFT_ONLY(2304)

DEFINE_MIXED_IDFT_ONLY(2304)

DEFINE_MIXED_DFT_ONLY(2400)

DEFINE_MIXED_IDFT_ONLY(2400)

DEFINE_MIXED_DFT_ONLY(2592)

DEFINE_MIXED_IDFT_ONLY(2592)

DEFINE_MIXED_DFT_ONLY(2700)

DEFINE_MIXED_IDFT_ONLY(2700)

DEFINE_MIXED_DFT_ONLY(2880)

DEFINE_MIXED_IDFT_ONLY(2880)

DEFINE_MIXED_DFT_ONLY(2916)

DEFINE_MIXED_IDFT_ONLY(2916)

DEFINE_MIXED_DFT_ONLY(3000)

DEFINE_MIXED_IDFT_ONLY(3000)

DEFINE_MIXED_DFT_ONLY(3240)

DEFINE_MIXED_IDFT_ONLY(3240)

DEFINE_MIXED_DFT_ONLY(1048576)
DEFINE_MIXED_IDFT_ONLY(1048576)

DEFINE_MIXED_DFT_ONLY(1572864)
DEFINE_MIXED_IDFT_ONLY(1572864)

void dft_implementation(uint8_t sizeidx, int16_t *input, int16_t *output, unsigned char scale_flag)
{
  AssertFatal(sizeidx < DFT_SIZE_IDXTABLESIZE, "Invalid dft size index %i\n", sizeidx);
  const int algn = (dft_ftab[sizeidx].size % 3) != 0 ? 0x1F : 0xF;
  AssertFatal(((intptr_t)output & algn) == 0, "Buffers should be aligned %p", (void *)output);
  if (((intptr_t)input) & algn) {
    LOG_D(PHY, "DFT called with input not aligned, add a memcpy, size %d\n", sizeidx);
    int sz = dft_ftab[sizeidx].size;
    if (sizeidx == DFT_12)
      sz *= 8;
    int16_t tmp[sz * 2] __attribute__((aligned(32)));
    memcpy(tmp, input, sizeof tmp);
    dft_ftab[sizeidx].func(tmp, output, scale_flag);
  } else
    dft_ftab[sizeidx].func(input, output, scale_flag);
}

void idft_implementation(uint8_t sizeidx, int16_t *input, int16_t *output, unsigned char scale_flag)
{
  AssertFatal(sizeidx < DFT_SIZE_IDXTABLESIZE, "Invalid idft size index %i\n", sizeidx);
  const int algn = 0x1F;
  AssertFatal(((intptr_t)output & algn) == 0, "Buffers should be 32-byte aligned %p", (void *)output);
  if (((intptr_t)input) & algn) {
    LOG_D(PHY, "DFT called with input not aligned, add a memcpy\n");
    int sz = idft_ftab[sizeidx].size;
    int16_t tmp[sz * 2] __attribute__((aligned(32)));
    memcpy(tmp, input, sizeof tmp);
    idft_ftab[sizeidx].func(tmp, output, scale_flag);
  } else
    idft_ftab[sizeidx].func(input, output, scale_flag);
}

#endif
