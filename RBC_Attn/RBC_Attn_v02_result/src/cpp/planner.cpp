// RBC-Attn v0.1 — exact-edge compiler (C++ / OpenMP)
//
// 입력: 한 KV head 의 query-major CSR (선택된 KV block 목록)
// 출력: GPU task descriptor 집합 + query 별 reduction CSR
//
// 계약 (설계 §2.2):
//   - 원래 sparse edge (query, KV block) 를 중복 없이 정확히 한 번 배정한다.
//   - 병합 직사각형의 빈 칸은 mask 로 제외한다 (계산은 하되 결과에 반영하지 않는다).
//   - 한 계획 그룹의 query 위치 수는 최대 min(64, max_m/G).
//
// 정책 (policy):
//   0 = RBC            signature rectangle + 비용 기반 bounded greedy 병합
//   1 = SIGNATURE_ONLY 동일 signature 만 묶음 (병합 없음)
//   2 = Q_OUTER        query 그룹의 KV 합집합을 하나의 직사각형으로
//   3 = KV_OUTER       KV block 하나마다 그것을 고른 query 를 모아 하나씩
//
// C ABI 로 노출해 ctypes 로 부른다 (pybind11 의존 없음).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kMaxGroupQ = 64;          // membership bitset = uint64_t 1개
constexpr int kMergeAtomLimit = 32;     // 설계 §2.1: 최대 32 atom
constexpr int kMergeTrialLimit = 32;    // 최대 32 회 탐색

struct Cost {
    double byte_weight = 1.0;
    double flop_weight = 0.015;
    double task_weight = 0.0;
    double partial_weight = 1.0;
};

struct Params {
    int bk = 128;
    int d = 128;
    int g = 8;
    int max_m = 128;        // GPU 타일의 query-row 상한 (q*G 기준)
    int policy = 0;
    // v0.2/P1: 기본 0. 과거 2*SM 강제 분할은 legacy_min_tasks 옵션으로만 남긴다.
    // 분할은 계획 후보의 일부이며, 선택 뒤에 다시 나누지 않는다.
    int min_tasks = 0;
    int q_outer_reorder = 1;
    Cost cost;
};

inline uint32_t round_pow2_at_least16(uint32_t x) {
    uint32_t v = 16;
    while (v < x) v <<= 1;
    return v;
}

// 설계 §2.1 의 비용 모형. 시간이 아니라 결정 휴리스틱이다.
inline double task_cost(uint32_t q, uint32_t k, const Params& p) {
    if (q == 0 || k == 0) return 0.0;
    const double G = p.g, D = p.d, BK = p.bk;
    const double q_bytes = 2.0 * q * G * D;
    const double kv_bytes = 4.0 * k * BK * D;
    const double partial = 8.0 * q * G * (D + 2.0);
    const double flops = 4.0 * round_pow2_at_least16((uint32_t)(q * G)) * k * BK * D;
    return p.cost.byte_weight * (q_bytes + kv_bytes)
         + p.cost.partial_weight * partial
         + p.cost.flop_weight * flops
         + p.cost.task_weight;
}

struct Atom {                 // 같은 signature 를 가진 KV block 묶음
    uint64_t sig;
    // uniq 안의 **인덱스** 를 담는다. uniq 는 정렬돼 있으므로 인덱스 순서가 block id
    // 순서와 같고, kmask 를 채울 때 block->위치 이진탐색이 필요 없다 (§8.6 항목 3).
    std::vector<int32_t> idx;
};

struct Task {
    std::vector<int32_t> queries;   // 그룹 내 지역 인덱스
    std::vector<int32_t> blocks;
    // blocks[j] 가 실제로 연결된 query 를 task 의 query 슬롯 순서로 표시한 비트마스크.
    // 병합 직사각형의 빈 칸은 여기서 0 이 되어 커널이 제외한다.
    std::vector<uint64_t> kmask;
};

// worker 별 스크래치. 호출 간 capacity 를 재사용해 hot path 할당을 없앤다 (§8.6 항목 1).
struct Scratch {
    std::vector<int32_t> uniq;
    std::vector<uint64_t> memb;
    std::vector<size_t> order;
    std::vector<Atom> atoms, cur, next;
    std::vector<std::pair<int32_t, int32_t>> pairs;   // (block id, query slot)
};

// Task 풀에서 다음 task 를 꺼낸다. 내부 vector 의 capacity 가 유지된다 (§8.6 항목 1·4).
inline Task& next_task(std::vector<Task>& out, size_t& used) {
    if (used >= out.size()) out.resize(used + 1);
    Task& t = out[used++];
    t.queries.clear();
    t.blocks.clear();
    t.kmask.clear();
    return t;
}

// 한 그룹(최대 64 query)에 대한 계획.
// CSR (ptr, ids) 을 직접 읽는다. 그룹마다 행을 vector 로 복사하지 않는다 (§8.6).
void plan_group(const int64_t* ptr, const int32_t* ids, int q0, int q1,
                const Params& p, std::vector<Task>& out, size_t& used, Scratch& sc) {
    const int nq = q1 - q0;
    if (nq <= 0) return;

    // --- KV block -> query membership bitset ---
    // (block, query slot) 쌍을 정렬해 한 번에 훑는다. 이진탐색을 쓰지 않는다.
    sc.pairs.clear();
    for (int i = 0; i < nq; ++i)
        for (int64_t a = ptr[q0 + i]; a < ptr[q0 + i + 1]; ++a)
            sc.pairs.push_back({ids[a], i});
    if (sc.pairs.empty()) return;
    std::sort(sc.pairs.begin(), sc.pairs.end());

    sc.uniq.clear();
    sc.memb.clear();
    for (size_t j = 0; j < sc.pairs.size();) {
        const int32_t b = sc.pairs[j].first;
        uint64_t m = 0;
        while (j < sc.pairs.size() && sc.pairs[j].first == b) {
            m |= (uint64_t(1) << sc.pairs[j].second);
            ++j;
        }
        sc.uniq.push_back(b);
        sc.memb.push_back(m);
    }
    const std::vector<int32_t>& uniq = sc.uniq;
    const std::vector<uint64_t>& memb = sc.memb;

    // task 의 query 슬롯 순서에 맞춰 block 별 마스크를 만든다.
    auto fill_from_idx = [&](Task& t, const std::vector<int32_t>& idx, uint64_t sig) {
        t.blocks.resize(idx.size());
        t.kmask.resize(idx.size());
        for (size_t j = 0; j < idx.size(); ++j) {
            const uint64_t full = memb[idx[j]];
            uint64_t m = 0;
            for (size_t si = 0; si < t.queries.size(); ++si)
                if (full & (uint64_t(1) << t.queries[si])) m |= (uint64_t(1) << si);
            t.blocks[j] = uniq[idx[j]];
            t.kmask[j] = m;
        }
        (void)sig;
    };
    auto queries_from_sig = [&](Task& t, uint64_t sig) {
        for (int i = 0; i < nq; ++i)
            if (sig & (uint64_t(1) << i)) t.queries.push_back(i);
    };
    auto row_empty = [&](int i) { return ptr[q0 + i + 1] == ptr[q0 + i]; };

    if (p.policy == 3) {                      // KV_OUTER: block 하나 = task 하나
        for (size_t j = 0; j < uniq.size(); ++j) {
            Task& t = next_task(out, used);
            queries_from_sig(t, memb[j]);
            const int32_t one = (int32_t)j;
            std::vector<int32_t> idx1(1, one);
            fill_from_idx(t, idx1, memb[j]);
            if (t.queries.empty() || t.blocks.empty()) --used;
        }
        return;
    }
    if (p.policy == 2) {                      // Q_OUTER: 그룹 합집합 하나
        Task& t = next_task(out, used);
        for (int i = 0; i < nq; ++i)
            if (!row_empty(i)) t.queries.push_back(i);   // 빈 행은 슬롯을 주지 않는다
        if (t.queries.empty()) { --used; return; }
        std::vector<int32_t> all(uniq.size());
        std::iota(all.begin(), all.end(), 0);
        uint64_t sig = 0;
        for (int32_t lq : t.queries) sig |= (uint64_t(1) << lq);
        fill_from_idx(t, all, sig);
        return;
    }

    // --- signature rectangle: 동일 membership 인 block 을 모은다 ---
    sc.order.resize(uniq.size());
    std::iota(sc.order.begin(), sc.order.end(), 0);
    std::sort(sc.order.begin(), sc.order.end(),
              [&](size_t a, size_t b) { return memb[a] < memb[b]; });

    sc.atoms.clear();
    for (size_t oi = 0; oi < sc.order.size();) {
        const uint64_t sig = memb[sc.order[oi]];
        sc.atoms.emplace_back();
        Atom& a = sc.atoms.back();
        a.sig = sig;
        a.idx.clear();
        while (oi < sc.order.size() && memb[sc.order[oi]] == sig) {
            a.idx.push_back((int32_t)sc.order[oi]);
            ++oi;
        }
        std::sort(a.idx.begin(), a.idx.end());
    }

    auto emit_atoms = [&](const std::vector<Atom>& as) {
        for (const auto& a : as) {
            Task& t = next_task(out, used);
            queries_from_sig(t, a.sig);
            if (t.queries.empty() || a.idx.empty()) { --used; continue; }
            fill_from_idx(t, a.idx, a.sig);
        }
    };

    if (p.policy == 1) { emit_atoms(sc.atoms); return; }   // SIGNATURE_ONLY

    // --- RBC: bounded greedy 병합 ---
    // 같은 query 를 공유하는 두 atom 을 합치면 Q·partial bytes 를 아끼고
    // masked FLOPs 를 더 낸다. 순비용이 음수일 때만 합친다.
    sc.cur = sc.atoms;
    if (sc.cur.size() <= (size_t)kMergeAtomLimit) {
        int trials = 0;
        bool improved = true;
        while (improved && trials < kMergeTrialLimit && sc.cur.size() > 1) {
            improved = false;
            double best_gain = 0.0;
            size_t bi = 0, bj = 0;
            for (size_t i = 0; i < sc.cur.size() && trials < kMergeTrialLimit; ++i) {
                for (size_t j = i + 1; j < sc.cur.size() && trials < kMergeTrialLimit; ++j) {
                    ++trials;
                    const uint64_t su = sc.cur[i].sig | sc.cur[j].sig;
                    if (su == 0) continue;
                    const uint32_t qi = (uint32_t)__builtin_popcountll(sc.cur[i].sig);
                    const uint32_t qj = (uint32_t)__builtin_popcountll(sc.cur[j].sig);
                    const uint32_t qu = (uint32_t)__builtin_popcountll(su);
                    const uint32_t ki = (uint32_t)sc.cur[i].idx.size();
                    const uint32_t kj = (uint32_t)sc.cur[j].idx.size();
                    const double before = task_cost(qi, ki, p) + task_cost(qj, kj, p);
                    const double after = task_cost(qu, ki + kj, p);
                    const double gain = before - after;
                    // max_m 제약: 병합 결과의 query-row 수가 타일 상한을 넘으면 불가
                    if ((int)(qu * p.g) > p.max_m) continue;
                    if (gain > best_gain) { best_gain = gain; bi = i; bj = j; }
                }
            }
            if (best_gain > 0.0) {
                sc.next.clear();
                for (size_t t = 0; t < sc.cur.size(); ++t)
                    if (t != bi && t != bj) sc.next.push_back(sc.cur[t]);
                sc.next.emplace_back();
                Atom& m = sc.next.back();
                m.sig = sc.cur[bi].sig | sc.cur[bj].sig;
                m.idx = sc.cur[bi].idx;
                m.idx.insert(m.idx.end(), sc.cur[bj].idx.begin(), sc.cur[bj].idx.end());
                std::sort(m.idx.begin(), m.idx.end());
                sc.cur.swap(sc.next);
                improved = true;
            }
        }
    }

    // Q-outer 합집합 경로가 전체로 더 싸면 그쪽을 쓴다.
    double rbc_cost = 0.0;
    for (const auto& a : sc.cur)
        rbc_cost += task_cost((uint32_t)__builtin_popcountll(a.sig),
                              (uint32_t)a.idx.size(), p);
    int nq_nonempty = 0;
    for (int i = 0; i < nq; ++i) if (!row_empty(i)) ++nq_nonempty;
    const double qouter_cost = task_cost((uint32_t)nq_nonempty, (uint32_t)uniq.size(), p);
    if (qouter_cost < rbc_cost && (int)(nq_nonempty * p.g) <= p.max_m) {
        Task& t = next_task(out, used);
        for (int i = 0; i < nq; ++i)
            if (!row_empty(i)) t.queries.push_back(i);
        if (t.queries.empty()) { --used; return; }
        std::vector<int32_t> all(uniq.size());
        std::iota(all.begin(), all.end(), 0);
        uint64_t sig = 0;
        for (int32_t lq : t.queries) sig |= (uint64_t(1) << lq);
        fill_from_idx(t, all, sig);
        return;
    }
    emit_atoms(sc.cur);
}

}  // namespace

extern "C" {

// 계획 결과를 담는 평탄 버퍼. 호출자가 크기를 먼저 물어본 뒤 할당한다.
struct RbcPlan {
    int32_t n_tasks;
    int32_t n_task_q;        // task_q 총 길이
    int32_t n_task_k;        // task_k 총 길이
    int32_t n_red;           // reduction CSR 총 길이 (multi-owner 슬롯만)
    int32_t n_multi_rows;    // degree>=2 인 query 행 수
    int32_t direct_rows;     // degree==1 (main 이 직접 기록)
    int32_t empty_rows;      // degree==0
};

// 내부 보관소 (한 번에 하나의 계획만 들고 있는다 — 벤치 용도)
static thread_local std::vector<int32_t> g_task_qptr, g_task_q, g_task_kptr, g_task_k;
static thread_local std::vector<uint64_t> g_task_kmask;
static thread_local std::vector<int32_t> g_red_ptr, g_red_slot;
static thread_local std::vector<int32_t> g_slot_mode;   // 0=partial, 1=direct
static thread_local std::vector<int32_t> g_multi_rows;  // degree>=2 인 query 행
static thread_local std::vector<int32_t> g_task_qbase;   // task 의 그룹 시작 query (전역 인덱스)
// reduction 평탄화용 스크래치. 호출 간 capacity 를 재사용한다 (§8.6 항목 1).
static thread_local std::vector<int32_t> g_deg, g_first_slot, g_row_pos, g_fill_cur;
// 그룹별 task 풀·스크래치. 호출 간 재사용한다 (§8.6 항목 1).
static thread_local std::vector<std::vector<Task>> g_per_group;
static thread_local std::vector<size_t> g_used;
static thread_local std::vector<Scratch> g_scratch;

int rbc_plan(const int64_t* ptr, const int32_t* block_ids, int nq,
             int bk, int d, int g, int max_m, int policy, int min_tasks,
             double byte_w, double flop_w, double task_w, double partial_w,
             int n_threads, RbcPlan* out_sizes) {
    Params p;
    p.bk = bk; p.d = d; p.g = g; p.max_m = max_m; p.policy = policy;
    p.min_tasks = min_tasks;
    p.cost.byte_weight = byte_w; p.cost.flop_weight = flop_w;
    p.cost.task_weight = task_w; p.cost.partial_weight = partial_w;

    const int group_q = std::min(kMaxGroupQ, std::max(1, max_m / std::max(1, g)));
    const int n_groups = (nq + group_q - 1) / group_q;

    // per_group 과 그 안의 Task 는 호출 간 재사용한다 (capacity 유지).
    // g_used[gi] 가 이번 호출에서 실제로 쓴 task 수다.
    if ((int)g_per_group.size() < n_groups) g_per_group.resize(n_groups);
    g_used.assign(n_groups, 0);
    if ((int)g_scratch.size() < n_groups) g_scratch.resize(n_groups);

    // OpenMP worker 는 자기 thread_local 인스턴스를 보므로, 마스터의 것을 포인터로
    // 넘겨야 한다. 그룹 인덱스가 유일하므로 worker 간 충돌은 없다.
    auto* pg = &g_per_group;
    auto* sc = &g_scratch;
    auto* us = &g_used;

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(dynamic)
#endif
    for (int gi = 0; gi < n_groups; ++gi) {
        const int q0 = gi * group_q;
        const int q1 = std::min(nq, q0 + group_q);
        size_t used = 0;
        plan_group(ptr, block_ids, q0, q1, p, (*pg)[gi], used, (*sc)[gi]);
        (*us)[gi] = used;
    }
    auto& per_group = g_per_group;

    // --- min_tasks 보강: 남는 CTA 가 적으면 긴 rectangle 을 KV 방향으로 분할 ---
    if (min_tasks > 0) {
        int total = 0;
        for (int gi = 0; gi < n_groups; ++gi) total += (int)g_used[gi];
        while (total < min_tasks) {
            // 가장 KV block 이 많은 task 를 반으로 쪼갠다
            int best_g = -1; size_t best_t = 0; size_t best_k = 1;
            for (int gi = 0; gi < n_groups; ++gi)
                for (size_t t = 0; t < g_used[gi]; ++t)
                    if (per_group[gi][t].blocks.size() > best_k) {
                        best_k = per_group[gi][t].blocks.size(); best_g = gi; best_t = t;
                    }
            if (best_g < 0) break;
            Task& src = per_group[best_g][best_t];
            const size_t half = src.blocks.size() / 2;
            Task nt; nt.queries = src.queries;
            nt.blocks.assign(src.blocks.begin() + half, src.blocks.end());
            nt.kmask.assign(src.kmask.begin() + half, src.kmask.end());
            src.blocks.resize(half);
            src.kmask.resize(half);
            if (g_used[best_g] >= per_group[best_g].size())
                per_group[best_g].resize(g_used[best_g] + 1);
            per_group[best_g][g_used[best_g]] = std::move(nt);
            ++g_used[best_g];
            ++total;
        }
    }

    // --- 평탄화 ---
    g_task_qptr.clear(); g_task_q.clear(); g_task_kptr.clear(); g_task_k.clear();
    g_task_kmask.clear(); g_task_qbase.clear();
    g_red_ptr.clear(); g_red_slot.clear();
    g_task_qptr.push_back(0); g_task_kptr.push_back(0);
    // reduction 은 slot 번호를 담는다. query 마다 vector 를 만들지 않고
    // count -> prefix sum -> fill 로 평탄 버퍼에 쓴다 (§8.6 항목 5).
    g_deg.assign(nq, 0);
    g_first_slot.assign(nq, -1);
    int tid = 0;
    for (int gi = 0; gi < n_groups; ++gi) {
        const int q0 = gi * group_q;
        for (size_t ti = 0; ti < g_used[gi]; ++ti) {
            auto& t = per_group[gi][ti];
            for (int32_t lq : t.queries) g_task_q.push_back(q0 + lq);
            for (size_t j = 0; j < t.blocks.size(); ++j) {
                g_task_k.push_back(t.blocks[j]);
                g_task_kmask.push_back(t.kmask[j]);
            }
            g_task_qptr.push_back((int32_t)g_task_q.size());
            g_task_kptr.push_back((int32_t)g_task_k.size());
            g_task_qbase.push_back(q0);
            ++tid;
        }
    }
    for (size_t sl = 0; sl < g_task_q.size(); ++sl) {
        const int32_t qq = g_task_q[sl];
        if (g_deg[qq] == 0) g_first_slot[qq] = (int32_t)sl;
        ++g_deg[qq];
    }

    // --- P2: 출력 소유권 (degree) 계산 ---
    // degree[row] = 그 query 의 결과를 쓰는 task 수.
    //   0      -> 빈 행 (out=0, LSE=-inf)
    //   1      -> main 이 정규화한 최종 출력을 직접 기록 (partial/merge 없음)
    //   2 이상 -> partial workspace 에 저장하고 compact merge
    g_slot_mode.assign(g_task_q.size(), 0);
    g_multi_rows.clear();
    g_row_pos.assign(nq, -1);
    int32_t direct_rows = 0, multi_rows = 0, empty_rows = 0;
    for (int i = 0; i < nq; ++i) {
        if (g_deg[i] == 0) { ++empty_rows; continue; }
        if (g_deg[i] == 1) { g_slot_mode[g_first_slot[i]] = 1; ++direct_rows; continue; }
        g_row_pos[i] = (int32_t)g_multi_rows.size();
        g_multi_rows.push_back(i);
        ++multi_rows;
    }
    g_red_ptr.assign(g_multi_rows.size() + 1, 0);
    for (size_t m = 0; m < g_multi_rows.size(); ++m)
        g_red_ptr[m + 1] = g_red_ptr[m] + g_deg[g_multi_rows[m]];
    g_red_slot.assign(g_red_ptr.empty() ? 0 : g_red_ptr.back(), 0);
    g_fill_cur.assign(g_multi_rows.size(), 0);
    for (size_t m = 0; m < g_multi_rows.size(); ++m) g_fill_cur[m] = g_red_ptr[m];
    for (size_t sl = 0; sl < g_task_q.size(); ++sl) {
        const int32_t rp = g_row_pos[g_task_q[sl]];
        if (rp >= 0) g_red_slot[g_fill_cur[rp]++] = (int32_t)sl;
    }

    out_sizes->n_tasks = tid;
    out_sizes->n_task_q = (int32_t)g_task_q.size();
    out_sizes->n_task_k = (int32_t)g_task_k.size();
    out_sizes->n_red = (int32_t)g_red_slot.size();
    out_sizes->n_multi_rows = multi_rows;
    out_sizes->direct_rows = direct_rows;
    out_sizes->empty_rows = empty_rows;
    return 0;
}

// 앞서 만든 계획을 호출자 버퍼로 복사한다.
int rbc_fetch(int32_t* task_qptr, int32_t* task_q, int32_t* task_kptr, int32_t* task_k,
              int32_t* red_ptr, int32_t* red_slot, uint64_t* task_kmask,
              int32_t* slot_mode, int32_t* multi_rows) {
    std::memcpy(task_qptr, g_task_qptr.data(), g_task_qptr.size() * sizeof(int32_t));
    std::memcpy(task_q, g_task_q.data(), g_task_q.size() * sizeof(int32_t));
    std::memcpy(task_kptr, g_task_kptr.data(), g_task_kptr.size() * sizeof(int32_t));
    std::memcpy(task_k, g_task_k.data(), g_task_k.size() * sizeof(int32_t));
    std::memcpy(red_ptr, g_red_ptr.data(), g_red_ptr.size() * sizeof(int32_t));
    std::memcpy(red_slot, g_red_slot.data(), g_red_slot.size() * sizeof(int32_t));
    std::memcpy(task_kmask, g_task_kmask.data(), g_task_kmask.size() * sizeof(uint64_t));
    std::memcpy(slot_mode, g_slot_mode.data(), g_slot_mode.size() * sizeof(int32_t));
    std::memcpy(multi_rows, g_multi_rows.data(), g_multi_rows.size() * sizeof(int32_t));
    return 0;
}


// --- v0.2/P4: 계획 생성 없이 후보의 예측 특징만 뽑는다 (§7.7 "빠른 우회") ---
//
// Q-outer 와 signature 두 후보는 descriptor 를 만들지 않고도 특징이 정확히 결정된다.
// (Q-outer = 그룹당 union 1개, signature = 그룹당 동일 membership atom)
// bounded_merge 만 실제 탐색이 필요하다. 이 함수는 CSR 을 한 번 순회하고 그룹별로
// membership 해시를 세는 것까지만 하므로 rbc_plan 보다 훨씬 싸다.
//
// out 레이아웃은 rbc/select.py 의 PROBE_FIELDS 와 1:1 로 맞춘다.
static const int kProbeLen = 56;
static const int kBmCand[8] = {1, 2, 4, 8, 16, 32, 64, 128};

int rbc_probe(const int64_t* ptr, const int32_t* block_ids, int nq,
              int bk, int d, int g, int max_m, int n_threads, double* out) {
    (void)bk; (void)d;
    for (int i = 0; i < kProbeLen; ++i) out[i] = 0.0;
    const int group_q = std::min(kMaxGroupQ, std::max(1, max_m / std::max(1, g)));
    const int n_groups = (nq + group_q - 1) / group_q;

    // 그룹별 요약을 모아 두 번째 패스에서 BM 별 방문량을 누적한다.
    struct GSum {
        int q_nonempty = 0;
        int union_sz = 0;
        std::vector<std::pair<int, int>> atoms;   // (popcount, block 수)
        int max_atoms = 0;
        long long full_blocks_q = 0;     // Q-outer task 에서 membership 이 full 인 block
        long long slots_sig = 0;
        long long multi_rows = 0, direct_rows = 0;
        long long uniq_blocks = 0, issued_blocks = 0;
    };
    std::vector<GSum> gs(n_groups);
    long long empty_rows = 0, edges = 0;
    std::vector<int> degs(nq, 0);

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(dynamic)
#endif
    for (int gi = 0; gi < n_groups; ++gi) {
        const int q0 = gi * group_q;
        const int q1 = std::min(nq, q0 + group_q);
        // block -> membership bitset
        std::vector<int32_t> uniq;
        std::vector<uint64_t> memb;
        {
            std::vector<std::pair<int32_t, int>> pairs;
            for (int i = q0; i < q1; ++i)
                for (int64_t a = ptr[i]; a < ptr[i + 1]; ++a)
                    pairs.push_back({block_ids[a], i - q0});
            std::sort(pairs.begin(), pairs.end());
            for (size_t j = 0; j < pairs.size();) {
                const int32_t b = pairs[j].first;
                uint64_t m = 0;
                while (j < pairs.size() && pairs[j].first == b) {
                    m |= (uint64_t(1) << pairs[j].second); ++j;
                }
                uniq.push_back(b); memb.push_back(m);
            }
        }
        GSum& S = gs[gi];
        S.union_sz = (int)uniq.size();
        S.uniq_blocks = (long long)uniq.size();
        for (int i = q0; i < q1; ++i) {
            const int dgr = (int)(ptr[i + 1] - ptr[i]);
            if (dgr > 0) ++S.q_nonempty;
            S.issued_blocks += dgr;
        }
        // atom = 동일 membership
        std::vector<uint64_t> sorted_m = memb;
        std::vector<size_t> ord(memb.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(),
                  [&](size_t x, size_t y) { return memb[x] < memb[y]; });
        std::vector<int> deg_local(q1 - q0, 0);
        for (size_t oi = 0; oi < ord.size();) {
            const uint64_t sig = memb[ord[oi]];
            int cnt = 0;
            while (oi < ord.size() && memb[ord[oi]] == sig) { ++cnt; ++oi; }
            const int pc = __builtin_popcountll(sig);
            S.atoms.push_back({pc, cnt});
            S.slots_sig += pc;
            for (int b = 0; b < q1 - q0; ++b)
                if (sig & (uint64_t(1) << b)) ++deg_local[b];
        }
        S.max_atoms = (int)S.atoms.size();
        {
            const uint64_t want_q = (S.q_nonempty >= 64)
                ? ~uint64_t(0) : ((uint64_t(1) << S.q_nonempty) - 1);
            for (size_t j = 0; j < memb.size(); ++j)
                if (memb[j] == want_q) ++S.full_blocks_q;
        }
        for (int b = 0; b < q1 - q0; ++b) {
            if (deg_local[b] == 0) continue;
            if (deg_local[b] == 1) ++S.direct_rows; else ++S.multi_rows;
        }
    }

    // 전역 집계
    for (int i = 0; i < nq; ++i) {
        const int dgr = (int)(ptr[i + 1] - ptr[i]);
        degs[i] = dgr; edges += dgr;
        if (dgr == 0) ++empty_rows;
    }
    int maxq_q = 0, maxq_s = 0, maxk_q = 0, maxk_s = 0, max_atoms = 0;
    long long ntask_q = 0, ntask_s = 0, sumk_q = 0, sumk_s = 0;
    long long slots_q = 0, slots_s = 0, multi_q = 0, multi_s = 0;
    long long direct_q = 0, direct_s = 0, uniq_b = 0, issued_b = 0;
    double padded_q = 0.0, padded_s = 0.0;
    for (int gi = 0; gi < n_groups; ++gi) {
        const GSum& S = gs[gi];
        if (S.q_nonempty > 0) {
            ++ntask_q; sumk_q += S.union_sz;
            maxq_q = std::max(maxq_q, S.q_nonempty);
            maxk_q = std::max(maxk_q, S.union_sz);
            slots_q += S.q_nonempty;
            padded_q += (double)round_pow2_at_least16((uint32_t)(S.q_nonempty * g)) * S.union_sz;
        }
        ntask_s += (long long)S.atoms.size();
        for (const auto& a : S.atoms) {
            sumk_s += a.second;
            maxq_s = std::max(maxq_s, a.first);
            maxk_s = std::max(maxk_s, a.second);
            padded_s += (double)round_pow2_at_least16((uint32_t)(a.first * g)) * a.second;
        }
        slots_s += S.slots_sig;
        multi_s += S.multi_rows; direct_s += S.direct_rows;
        max_atoms = std::max(max_atoms, S.max_atoms);
        uniq_b += S.uniq_blocks; issued_b += S.issued_blocks;
    }
    // Q-outer 는 그룹당 task 하나라 모든 비어있지 않은 행이 degree 1 (direct)
    direct_q = slots_q; multi_q = 0;

    // BM 별 방문량: sum ceil(q/BM) * k. 실제 BM 은 next_pow2(max_q) 이지만
    // 선택기가 후보 BM 을 바꿔볼 수 있도록 8개 후보 전부 누적한다.
    double vis_q[8] = {0}, vis_s[8] = {0};
    for (int gi = 0; gi < n_groups; ++gi) {
        const GSum& S = gs[gi];
        for (int bi = 0; bi < 8; ++bi) {
            const int BM = kBmCand[bi];
            if (S.q_nonempty > 0)
                vis_q[bi] += (double)((S.q_nonempty + BM - 1) / BM) * S.union_sz;
            for (const auto& a : S.atoms)
                vis_s[bi] += (double)((a.first + BM - 1) / BM) * a.second;
        }
    }
    // KV 길이 분포 (§7.3: 합·p50·p90·max). task 당 KV block 수를 모아 분위수를 낸다.
    std::vector<int> kq, kszs;
    long long full_q = 0;
    for (int gi = 0; gi < n_groups; ++gi) {
        const GSum& S = gs[gi];
        if (S.q_nonempty > 0) kq.push_back(S.union_sz);
        for (const auto& a : S.atoms) kszs.push_back(a.second);
        full_q += S.full_blocks_q;
    }
    std::sort(kq.begin(), kq.end());
    std::sort(kszs.begin(), kszs.end());
    auto quant = [](const std::vector<int>& v, double p) -> double {
        if (v.empty()) return 0.0;
        size_t i = (size_t)(v.size() * p);
        if (i >= v.size()) i = v.size() - 1;
        return (double)v[i];
    };
    const double q_k_p50 = quant(kq, 0.5), q_k_p90 = quant(kq, 0.9);
    const double s_k_p50 = quant(kszs, 0.5), s_k_p90 = quant(kszs, 0.9);
    const double q_full_ratio = uniq_b > 0 ? (double)full_q / (double)uniq_b : 1.0;

    std::vector<int> dsort;
    dsort.reserve(nq);
    for (int i = 0; i < nq; ++i) if (degs[i] > 0) dsort.push_back(degs[i]);
    std::sort(dsort.begin(), dsort.end());
    const double p50 = dsort.empty() ? 0.0 : dsort[dsort.size() / 2];
    const double p90 = dsort.empty() ? 0.0 : dsort[(size_t)(dsort.size() * 0.9)];
    const double dmax = dsort.empty() ? 0.0 : dsort.back();

    int o = 0;
    out[o++] = n_groups; out[o++] = group_q; out[o++] = (double)edges;
    out[o++] = (double)empty_rows; out[o++] = (double)dsort.size();
    out[o++] = (double)ntask_q; out[o++] = maxq_q; out[o++] = maxk_q; out[o++] = (double)sumk_q;
    out[o++] = (double)ntask_s; out[o++] = maxq_s; out[o++] = maxk_s; out[o++] = (double)sumk_s;
    for (int bi = 0; bi < 8; ++bi) out[o++] = vis_q[bi];
    for (int bi = 0; bi < 8; ++bi) out[o++] = vis_s[bi];
    out[o++] = padded_q; out[o++] = padded_s;
    out[o++] = (double)slots_q; out[o++] = (double)slots_s;
    out[o++] = (double)direct_q; out[o++] = (double)direct_s;
    out[o++] = (double)multi_q; out[o++] = (double)multi_s;
    out[o++] = (double)max_atoms;
    out[o++] = p50; out[o++] = p90; out[o++] = dmax;
    out[o++] = (double)uniq_b; out[o++] = (double)issued_b;
    out[o++] = q_k_p50; out[o++] = q_k_p90;
    out[o++] = s_k_p50; out[o++] = s_k_p90;
    out[o++] = q_full_ratio; out[o++] = 1.0;   // signature atom 은 정의상 full membership
    return o;
}

int rbc_probe_len() { return kProbeLen; }

int rbc_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // extern "C"
