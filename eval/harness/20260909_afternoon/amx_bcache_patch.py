#!/usr/bin/env python3
"""IDE_046: GemmKernel224Int4 AMX 경로 — M 블록마다 반복되던 INT4→INT8 B 타일 언팩을 첫 M 블록에서 스레드 로컬 L2 캐시에 두고 재사용.
env KT_AMX_BCACHE=1 (m > M_STEP 일 때만 발동), KT_AMX_A_LOADD=1 (A 타일 non-temporal → 일반 load)."""
P = "/sgl-workspace/ktransformers/kt-kernel/operators/amx/la/amx_kernels.hpp"
s = open(P).read()
if "amx_kernel_bcached" in s:
    print("already"); raise SystemExit

# (0) include
s = s.replace("#include <mutex>\n", "#include <mutex>\n#include <type_traits>\n#include <cstdlib>\n", 1)

# (1) GemmKernel224Int4 안에 캐시형 커널 추가 (apply_scale 앞)
i0 = s.index("struct GemmKernel224Int4 {")
i1 = s.index("  static void apply_scale(int m, int n, int m_begin, int n_begin, float* c, BufferA* ba, BufferB* bb) {", i0)
kernel = r'''  // ---- IDE_046: B 타일 언팩 캐시. scratch 레이아웃: [(n_step_idx * KB_STEPS + k_step_idx)] × 4 KB
  //      (lo tile2 1KB | lo tile3 1KB | hi tile2 1KB | hi tile3 1KB). fill=true 면 언팩+저장, false 면 저장분을 tileload.
  static constexpr int BC_KB_STEPS = K_BLOCK / (2 * K_STEP);            // 28
  static constexpr int BC_N_STEPS = N_BLOCK / N_STEP;                    // 4
  static constexpr size_t BC_TILE_BYTES = (size_t)TILE_N * TILE_K;      // 1024
  static constexpr size_t BC_SCRATCH_BYTES = (size_t)BC_N_STEPS * BC_KB_STEPS * 4 * BC_TILE_BYTES;  // 458752
  static bool bcache_on() {
    static bool v = std::getenv("KT_AMX_BCACHE") != nullptr;
    return v;
  }
  static bool a_loadd_on() {
    static bool v = std::getenv("KT_AMX_A_LOADD") != nullptr;
    return v;
  }
  static int8_t* bcache_scratch() {
    static thread_local int8_t* buf = nullptr;
    if (!buf) buf = (int8_t*)std::aligned_alloc(64, BC_SCRATCH_BYTES);
    return buf;
  }
  static void amx_kernel_bcached(int m, int n, int k, int m_begin, int n_begin, int n_start, int k_block_begin, float* c,
                                 BufferA* ba, BufferB* bb, int8_t* scratch, bool fill) {
#ifdef HAVE_AMX
    using K = GemmKernel224Int4;
    if (k_block_begin == 0) {
      K::clean_c();
    } else {
      K::load_c((int32_t*)c, K::N_STEP * sizeof(int32_t));
    }
    const bool a_ld = a_loadd_on();
    const int n_idx = (n_begin - n_start) / N_STEP;
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      const int kb_idx = k_begin / K::BufferB::B_K_STEP;
      __m512i* sc = (__m512i*)(scratch + ((size_t)n_idx * BC_KB_STEPS + kb_idx) * 4 * BC_TILE_BYTES);
      if (fill) {
        const __m512i* b = (const __m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);  // 32 행 × 64 B
        const __m512i lo = lo_mask(), hi = hi_mask();
        for (int i = 0; i < 2 * TILE_N; i++) {
          __m512i raw = b[i];
          sc[i] = _mm512_slli_epi32(_mm512_and_si512(lo, raw), 4);   // lo: 행 0..31 → tile2 (0..15) / tile3 (16..31)
          sc[2 * TILE_N + i] = _mm512_and_si512(hi, raw);            // hi
        }
        asm volatile("" ::: "memory");
      }
      int8_t* a0 = (int8_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      int8_t* a1 = (int8_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP);
      if (a_ld) {
        _tile_loadd(0, a0, K::K_STEP);
        _tile_loadd(1, a0 + (size_t)K::K_STEP * TILE_M, K::K_STEP);
      } else {
        K::load_a(a0, K::K_STEP * sizeof(int8_t));
      }
      _tile_loadd(2, sc, TILE_K);
      _tile_loadd(3, sc + TILE_N, TILE_K);
      K::run_tile();
      if (a_ld) {
        _tile_loadd(0, a1, K::K_STEP);
        _tile_loadd(1, a1 + (size_t)K::K_STEP * TILE_M, K::K_STEP);
      } else {
        K::load_a(a1, K::K_STEP * sizeof(int8_t));
      }
      _tile_loadd(2, sc + 2 * TILE_N, TILE_K);
      _tile_loadd(3, sc + 3 * TILE_N, TILE_K);
      K::run_tile();
    }
    K::store_c((int32_t*)c, K::N_STEP * sizeof(int32_t));
#else
    (void)m; (void)n; (void)k; (void)m_begin; (void)n_begin; (void)n_start; (void)k_block_begin; (void)c; (void)ba; (void)bb; (void)scratch; (void)fill;
#endif
  }

'''
s = s[:i1] + kernel + s[i1:]

# (2) integer_mat_mul: Int4 + AMX + m > M_STEP + env → 캐시 경로
j0 = s.index("void integer_mat_mul(int m, int n, int k, typename K::BufferA* ba, typename K::BufferB* bb, typename K::BufferC* bc,")
anchor = "  auto [n_start, n_end] = K::split_range_n(n, ith, nth);\n"
j1 = s.index(anchor, j0) + len(anchor)
branch = r'''
  if constexpr (amx_or_avx && AMX_AVAILABLE && std::is_same_v<K, GemmKernel224Int4>) {
    if (m > K::M_STEP && (n_end - n_start) <= K::N_BLOCK && K::bcache_on()) {
      int8_t* scratch = K::bcache_scratch();
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
        for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
          for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
            float* c = bc->get_submat(m, n, m_begin, n_begin);
            K::amx_kernel_bcached(m, n, k, m_begin, n_begin, n_start, k_block_begin, c, ba, bb, scratch, m_begin == 0);
            if (k_block_begin + K::K_BLOCK >= k) {
              K::apply_scale(m, n, m_begin, n_begin, c, ba, bb);
            }
          }
        }
      }
      return;
    }
  }
'''
s = s[:j1] + branch + s[j1:]
open(P, "w").write(s)
print("IDE_046 bcache patch applied")
