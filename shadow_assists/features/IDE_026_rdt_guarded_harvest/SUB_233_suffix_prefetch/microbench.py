import importlib.util, sys, time, random

def load(so):
    spec = importlib.util.spec_from_file_location("_C", so)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def build_tree(m, n_seqs, seq_len, vocab, seed):
    rng = random.Random(seed)
    t = m.SuffixTree(64)
    seqs = []
    for sid in range(n_seqs):
        # realistic-ish: some repeated motifs to create branching + matches
        s = [rng.randint(0, vocab) for _ in range(seq_len)]
        # inject repeats so the tree has real depth/branches
        for k in range(0, seq_len-20, 40):
            motif = s[k:k+12]
            j = rng.randint(0, seq_len-12)
            s[j:j+12] = motif
        t.extend(sid, s); seqs.append(s)
    return t, seqs

def bench(m, n_seqs, seq_len, vocab, n_calls, ctx_len, K, seed):
    t, seqs = build_tree(m, n_seqs, seq_len, vocab, seed)
    rng = random.Random(seed+1)
    # pre-generate contexts drawn from the sequences (so they match → real walks)
    ctxs = []
    for _ in range(n_calls):
        s = seqs[rng.randrange(len(seqs))]
        p = rng.randrange(0, len(s)-ctx_len)
        ctxs.append(s[p:p+ctx_len])
    # warmup
    for c in ctxs[:1000]: t.speculate(c, K, 1, 1.0, 0.1, False)
    t0 = time.perf_counter()
    tot = 0
    for c in ctxs:
        d = t.speculate(c, K, 1, 1.0, 0.1, False)
        tot += len(d.token_ids)
    dt = time.perf_counter() - t0
    return dt, n_calls, tot

if __name__ == "__main__":
    so = sys.argv[1]
    m = load(so)
    # large tree to stress memory hierarchy (many seqs → cache misses on walk)
    dt, n, tot = bench(m, n_seqs=400, seq_len=400, vocab=32000,
                       n_calls=200000, ctx_len=24, K=6, seed=12345)
    print(f"{so.split('/')[-1]}: {dt*1e9/n:.1f} ns/call  ({n} calls, {dt:.3f}s, drafted {tot} toks)")
