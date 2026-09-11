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
    int min_tasks = 0;      // 0 이면 CTA 보강 분할 없음
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
    std::vector<int32_t> blocks;
};

struct Task {
    std::vector<int32_t> queries;   // 그룹 내 지역 인덱스
    std::vector<int32_t> blocks;
    // blocks[j] 가 실제로 연결된 query 를 task 의 query 슬롯 순서로 표시한 비트마스크.
    // 병합 직사각형의 빈 칸은 여기서 0 이 되어 커널이 제외한다.
    std::vector<uint64_t> kmask;
};

// 한 그룹(최대 64 query)에 대한 계획
void plan_group(const std::vector<std::vector<int32_t>>& rows,  // rows[i] = query i 의 block 목록
                const Params& p, std::vector<Task>& out) {
    const int nq = (int)rows.size();
    if (nq == 0) return;

    // --- KV block -> query membership bitset ---
    // block id 는 희소하므로 정렬된 고유 목록을 만들고 그 위에서 센다.
    std::vector<int32_t> uniq;
    for (const auto& r : rows) uniq.insert(uniq.end(), r.begin(), r.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    if (uniq.empty()) return;

    std::vector<uint64_t> memb(uniq.size(), 0);
    for (int i = 0; i < nq; ++i) {
        for (int32_t b : rows[i]) {
            const size_t pos = std::lower_bound(uniq.begin(), uniq.end(), b) - uniq.begin();
            memb[pos] |= (uint64_t(1) << i);
        }
    }

    // task 의 query 슬롯 순서에 맞춰 block 별 마스크를 만든다.
    auto fill_kmask = [&](Task& t) {
        t.kmask.resize(t.blocks.size(), 0);
        for (size_t j = 0; j < t.blocks.size(); ++j) {
            const size_t pos = std::lower_bound(uniq.begin(), uniq.end(), t.blocks[j]) - uniq.begin();
            const uint64_t full = memb[pos];
            uint64_t m = 0;
            for (size_t si = 0; si < t.queries.size(); ++si)
                if (full & (uint64_t(1) << t.queries[si])) m |= (uint64_t(1) << si);
            t.kmask[j] = m;
        }
    };

    if (p.policy == 3) {                      // KV_OUTER: block 하나 = task 하나
        for (size_t j = 0; j < uniq.size(); ++j) {
            Task t;
            for (int i = 0; i < nq; ++i)
                if (memb[j] & (uint64_t(1) << i)) t.queries.push_back(i);
            t.blocks.push_back(uniq[j]);
            fill_kmask(t);
            out.push_back(std::move(t));
        }
        return;
    }
    if (p.policy == 2) {                      // Q_OUTER: 그룹 전체 합집합 하나
        Task t;
        t.queries.resize(nq);
        std::iota(t.queries.begin(), t.queries.end(), 0);
        t.blocks = uniq;
        fill_kmask(t);
        out.push_back(std::move(t));
        return;
    }

    // --- signature rectangle: 동일 membership 인 block 을 모은다 ---
    std::vector<size_t> order(uniq.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return memb[a] < memb[b]; });

    std::vector<Atom> atoms;
    for (size_t oi = 0; oi < order.size();) {
        const uint64_t sig = memb[order[oi]];
        Atom a; a.sig = sig;
        while (oi < order.size() && memb[order[oi]] == sig) {
            a.blocks.push_back(uniq[order[oi]]);
            ++oi;
        }
        std::sort(a.blocks.begin(), a.blocks.end());
        atoms.push_back(std::move(a));
    }

    // --- Q-outer 합집합이 더 싼지 먼저 본다 (설계 §2 단계 4) ---
    auto emit_atoms = [&](const std::vector<Atom>& as) {
        for (const auto& a : as) {
            Task t;
            for (int i = 0; i < nq; ++i)
                if (a.sig & (uint64_t(1) << i)) t.queries.push_back(i);
            t.blocks = a.blocks;
            fill_kmask(t);
            if (!t.queries.empty() && !t.blocks.empty()) out.push_back(std::move(t));
        }
    };

    if (p.policy == 1) { emit_atoms(atoms); return; }   // SIGNATURE_ONLY

    // --- RBC: bounded greedy 병합 ---
    // 같은 query 를 공유하는 두 atom 을 합치면 Q·partial bytes 를 아끼고
    // masked FLOPs 를 더 낸다. 순비용이 음수일 때만 합친다.
    std::vector<Atom> cur = atoms;
    if (cur.size() <= (size_t)kMergeAtomLimit) {
        int trials = 0;
        bool improved = true;
        while (improved && trials < kMergeTrialLimit && cur.size() > 1) {
            improved = false;
            double best_gain = 0.0;
            size_t bi = 0, bj = 0;
            for (size_t i = 0; i < cur.size() && trials < kMergeTrialLimit; ++i) {
                for (size_t j = i + 1; j < cur.size() && trials < kMergeTrialLimit; ++j) {
                    ++trials;
                    const uint64_t su = cur[i].sig | cur[j].sig;
                    if (su == 0) continue;
                    const uint32_t qi = (uint32_t)__builtin_popcountll(cur[i].sig);
                    const uint32_t qj = (uint32_t)__builtin_popcountll(cur[j].sig);
                    const uint32_t qu = (uint32_t)__builtin_popcountll(su);
                    const uint32_t ki = (uint32_t)cur[i].blocks.size();
                    const uint32_t kj = (uint32_t)cur[j].blocks.size();
                    const double before = task_cost(qi, ki, p) + task_cost(qj, kj, p);
                    const double after = task_cost(qu, ki + kj, p);
                    const double gain = before - after;
                    // max_m 제약: 병합 결과의 query-row 수가 타일 상한을 넘으면 불가
                    if ((int)(qu * p.g) > p.max_m) continue;
                    if (gain > best_gain) { best_gain = gain; bi = i; bj = j; }
                }
            }
            if (best_gain > 0.0) {
                Atom m;
                m.sig = cur[bi].sig | cur[bj].sig;
                m.blocks = cur[bi].blocks;
                m.blocks.insert(m.blocks.end(), cur[bj].blocks.begin(), cur[bj].blocks.end());
                std::sort(m.blocks.begin(), m.blocks.end());
                std::vector<Atom> next;
                for (size_t t = 0; t < cur.size(); ++t)
                    if (t != bi && t != bj) next.push_back(cur[t]);
                next.push_back(std::move(m));
                cur.swap(next);
                improved = true;
            }
        }
    }

    // Q-outer 합집합 경로가 전체로 더 싸면 그쪽을 쓴다.
    double rbc_cost = 0.0;
    for (const auto& a : cur)
        rbc_cost += task_cost((uint32_t)__builtin_popcountll(a.sig),
                              (uint32_t)a.blocks.size(), p);
    const double qouter_cost = task_cost((uint32_t)nq, (uint32_t)uniq.size(), p);
    if (qouter_cost < rbc_cost && (int)(nq * p.g) <= p.max_m) {
        Task t;
        t.queries.resize(nq);
        std::iota(t.queries.begin(), t.queries.end(), 0);
        t.blocks = uniq;
        fill_kmask(t);
        out.push_back(std::move(t));
        return;
    }
    emit_atoms(cur);
}

}  // namespace

extern "C" {

// 계획 결과를 담는 평탄 버퍼. 호출자가 크기를 먼저 물어본 뒤 할당한다.
struct RbcPlan {
    int32_t n_tasks;
    int32_t n_task_q;        // task_q 총 길이
    int32_t n_task_k;        // task_k 총 길이
    int32_t n_red;           // reduction CSR 총 길이
};

// 내부 보관소 (한 번에 하나의 계획만 들고 있는다 — 벤치 용도)
static thread_local std::vector<int32_t> g_task_qptr, g_task_q, g_task_kptr, g_task_k;
static thread_local std::vector<uint64_t> g_task_kmask;
static thread_local std::vector<int32_t> g_red_ptr, g_red_task;
static thread_local std::vector<int32_t> g_task_qbase;   // task 의 그룹 시작 query (전역 인덱스)

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

    std::vector<std::vector<Task>> per_group(n_groups);

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(dynamic)
#endif
    for (int gi = 0; gi < n_groups; ++gi) {
        const int q0 = gi * group_q;
        const int q1 = std::min(nq, q0 + group_q);
        std::vector<std::vector<int32_t>> rows(q1 - q0);
        for (int i = q0; i < q1; ++i) {
            const int64_t a = ptr[i], b = ptr[i + 1];
            rows[i - q0].assign(block_ids + a, block_ids + b);
        }
        plan_group(rows, p, per_group[gi]);
    }

    // --- min_tasks 보강: 남는 CTA 가 적으면 긴 rectangle 을 KV 방향으로 분할 ---
    if (min_tasks > 0) {
        int total = 0;
        for (auto& v : per_group) total += (int)v.size();
        while (total < min_tasks) {
            // 가장 KV block 이 많은 task 를 반으로 쪼갠다
            int best_g = -1; size_t best_t = 0; size_t best_k = 1;
            for (int gi = 0; gi < n_groups; ++gi)
                for (size_t t = 0; t < per_group[gi].size(); ++t)
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
            per_group[best_g].push_back(std::move(nt));
            ++total;
        }
    }

    // --- 평탄화 ---
    g_task_qptr.clear(); g_task_q.clear(); g_task_kptr.clear(); g_task_k.clear();
    g_task_kmask.clear(); g_task_qbase.clear();
    g_task_qptr.push_back(0); g_task_kptr.push_back(0);
    std::vector<std::vector<int32_t>> red(nq);   // query -> task id 목록
    int tid = 0;
    for (int gi = 0; gi < n_groups; ++gi) {
        const int q0 = gi * group_q;
        for (auto& t : per_group[gi]) {
            for (int32_t lq : t.queries) {
                // reduction 은 partial 슬롯 번호(= task_q 안의 위치)를 가리킨다.
                red[q0 + lq].push_back((int32_t)g_task_q.size());
                g_task_q.push_back(q0 + lq);
            }
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
    g_red_ptr.clear(); g_red_task.clear();
    g_red_ptr.push_back(0);
    for (int i = 0; i < nq; ++i) {
        for (int32_t t : red[i]) g_red_task.push_back(t);
        g_red_ptr.push_back((int32_t)g_red_task.size());
    }

    out_sizes->n_tasks = tid;
    out_sizes->n_task_q = (int32_t)g_task_q.size();
    out_sizes->n_task_k = (int32_t)g_task_k.size();
    out_sizes->n_red = (int32_t)g_red_task.size();
    return 0;
}

// 앞서 만든 계획을 호출자 버퍼로 복사한다.
int rbc_fetch(int32_t* task_qptr, int32_t* task_q, int32_t* task_kptr, int32_t* task_k,
              int32_t* red_ptr, int32_t* red_task, uint64_t* task_kmask) {
    std::memcpy(task_qptr, g_task_qptr.data(), g_task_qptr.size() * sizeof(int32_t));
    std::memcpy(task_q, g_task_q.data(), g_task_q.size() * sizeof(int32_t));
    std::memcpy(task_kptr, g_task_kptr.data(), g_task_kptr.size() * sizeof(int32_t));
    std::memcpy(task_k, g_task_k.data(), g_task_k.size() * sizeof(int32_t));
    std::memcpy(red_ptr, g_red_ptr.data(), g_red_ptr.size() * sizeof(int32_t));
    std::memcpy(red_task, g_red_task.data(), g_red_task.size() * sizeof(int32_t));
    std::memcpy(task_kmask, g_task_kmask.data(), g_task_kmask.size() * sizeof(uint64_t));
    return 0;
}

int rbc_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // extern "C"
