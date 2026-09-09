P="/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"; s=open(P).read()
if "KT_DUMP_A" in s: print("already"); raise SystemExit
anchor="""    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();"""
assert s.count(anchor)==2
dump="""    {
      static bool dump_on = std::getenv("KT_DUMP_A") != nullptr; static int dump_ct = 0;
      if (dump_on && qlen >= 10 && dump_ct < 4 && tp_part_idx == 0) {
        int e0 = m_expert_id_map_[0]; char fn[128]; snprintf(fn, sizeof fn, "/tmp/dumpA_q%d_%d_%s.bin", qlen, dump_ct, fuse_qin_ ? "fuse" : "base");
        FILE* f = fopen(fn, "wb"); if (f) {
          int m = m_local_num_[e0]; int mm = gate_up_ba_[e0]->max_m; int kk = config_.hidden_size;
          fwrite(&m, 4, 1, f); fwrite(&mm, 4, 1, f); fwrite(&kk, 4, 1, f);
          fwrite(gate_up_ba_[e0]->a, 1, (size_t)mm * kk, f); fwrite(gate_up_ba_[e0]->d, 4, mm, f);
          // 토큰→행 매핑도 기록
          for (int i = 0; i < qlen; i++) for (int j = 0; j < k; j++) if (expert_ids[i * k + j] == e0) { int rec[2] = {i, m_local_pos_[i][j]}; fwrite(rec, 4, 2, f); }
          fclose(f); }
        dump_ct++;
      }
    }
"""
i=s.index(anchor); s=s[:i]+dump+s[i:]; open(P,"w").write(s); print("dump patch applied (prefill path)")
