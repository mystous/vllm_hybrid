# VALIDATION_REPORT — IDE_075 (2026-09-17T15:58:37+09:00)

## parser/단위/기록기 합성 시험

```
{
 "t_kst": "2026-09-17T15:24:27.108663+09:00",
 "tsparse_selftest": "PASS (T01-T07, W01-W06)",
 "gpu_union_selftest": "PASS (IDE_074 §4.3 8건, 재사용)",
 "recorder_synthetic": {
  "Q11_ring_overflow": "ok=8 overwritten_detected=2 (cap 8 / 10 tasks), 작업 정지 없음",
  "Q12_snapshot": "PASS",
  "order": "enq_b<=enq_a<=dequeue<=exec_start<=exec_end PASS",
  "spsc_stale": "PASS",
  "Q01_Q02_fifo_reducer_definition": "PASS",
  "kt_evt_now_cost_ns": 27.2
 },
 "kt_kernel_so_v2_sha256": "a4e9045a038bc7af2ec81b1cf251b007399d7e998a5d58c86f579c1bd1bd6ed3",
 "so_bytes": 8319008
}
```

## 세션별 항목 유효성

### S2_CORE (CORE) all_pass=True

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": true,
  "in": 65886,
  "out": 16384,
  "expected": "65886/16384 (PROBE128, 지시서 §2.2)"
 },
 "config_valid": {
  "pass": true,
  "so": "a4e9045a038bc7af",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE_153555/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_process_begin",
   "client_process_end",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "last_related_deferred_end",
   "log_flush_end",
   "session_begin"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24490
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 5,
  "fwd_missing_share": 0.0002041649652919559,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24490,
  "def_tasks_expected": 24490,
  "fwd_recorded": 24485,
  "done_store_recorded": 24490,
  "imm_tasks": 395,
  "def_task_seq_found_in_tasks": 24490,
  "def_task_kinds": {
   "3": 24490
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 395,
   "2": 24490,
   "3": 24490
  },
  "expected": {
   "3(deferred)": 24490,
   "2(done)": 24490,
   "1(imm)": 395
  }
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S3_CORR (CORR) all_pass=True

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": true,
  "in": 65886,
  "out": 16384,
  "expected": "65886/16384 (PROBE128, 지시서 §2.2)"
 },
 "config_valid": {
  "pass": true,
  "so": "a4e9045a038bc7af",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE_153555/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_process_begin",
   "client_process_end",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "last_related_deferred_end",
   "log_flush_end",
   "session_begin"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24490
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24490,
  "def_tasks_expected": 24490,
  "fwd_recorded": 24490,
  "done_store_recorded": 24490,
  "imm_tasks": 395,
  "def_task_seq_found_in_tasks": 24490,
  "def_task_kinds": {
   "3": 24490
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 395,
   "2": 24490,
   "3": 24490
  },
  "expected": {
   "3(deferred)": 24490,
   "2(done)": 24490,
   "1(imm)": 395
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "166-TP-0.trace.json.gz": {
    "kernel": 695850,
    "memcpy": 124894
   },
   "166-TP-1.trace.json.gz": {
    "kernel": 598558,
    "memcpy": 1762
   },
   "166-TP-2.trace.json.gz": {
    "kernel": 598558,
    "memcpy": 1762
   },
   "166-TP-3.trace.json.gz": {
    "kernel": 598558,
    "memcpy": 1762
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 23808
  },
  "matched": 23808,
  "method": "VALIDATED_HEURISTIC_MAPPING"
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 23808
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.9002436209770484,
    "late_share_thr_1us": 0.8992819592255418,
    "late_share_thr_2us": 0.8987690729580715,
    "late_share_thr_5us": 0.8965893063213233,
    "clock_indeterminate_share_5us": 0.008334401846390564,
    "n": 15598,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.014864864864864866,
    "late_share_thr_1us": 0.014864864864864866,
    "late_share_thr_2us": 0.014864864864864866,
    "late_share_thr_5us": 0.014414414414414415,
    "clock_indeterminate_share_5us": 0.0013513513513513514,
    "n": 2220,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15598,
    15598
   ],
   "32:cold_present_nonempty": [
    2220,
    2220
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15598,
   "2:cold_path_empty": 18,
   "2:no_cold_consumed": 256,
   "32:cold_present_nonempty": 2220,
   "32:cold_path_empty": 5588,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S4_CORR (CORR) all_pass=True

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": true,
  "in": 65886,
  "out": 16384,
  "expected": "65886/16384 (PROBE128, 지시서 §2.2)"
 },
 "config_valid": {
  "pass": true,
  "so": "a4e9045a038bc7af",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE_153555/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_process_begin",
   "client_process_end",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "last_related_deferred_end",
   "log_flush_end",
   "session_begin"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24490
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 2,
  "fwd_missing_share": 8.166598611678236e-05,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24490,
  "def_tasks_expected": 24490,
  "fwd_recorded": 24488,
  "done_store_recorded": 24490,
  "imm_tasks": 395,
  "def_task_seq_found_in_tasks": 24490,
  "def_task_kinds": {
   "3": 24490
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 395,
   "2": 24490,
   "3": 24490
  },
  "expected": {
   "3(deferred)": 24490,
   "2(done)": 24490,
   "1(imm)": 395
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "290-TP-0.trace.json.gz": {
    "kernel": 695785,
    "memcpy": 124900
   },
   "290-TP-1.trace.json.gz": {
    "kernel": 598493,
    "memcpy": 1768
   },
   "290-TP-2.trace.json.gz": {
    "kernel": 598493,
    "memcpy": 1768
   },
   "290-TP-3.trace.json.gz": {
    "kernel": 598493,
    "memcpy": 1768
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 23808
  },
  "matched": 23808,
  "method": "VALIDATED_HEURISTIC_MAPPING"
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 23808
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.8990125673249552,
    "late_share_thr_1us": 0.8986278532957168,
    "late_share_thr_2us": 0.8979866632469864,
    "late_share_thr_5us": 0.8956142600666838,
    "clock_indeterminate_share_5us": 0.006860733521415747,
    "n": 15596,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.01697866782760122,
    "late_share_thr_1us": 0.01697866782760122,
    "late_share_thr_2us": 0.01697866782760122,
    "late_share_thr_5us": 0.01697866782760122,
    "clock_indeterminate_share_5us": 0.0017414018284719198,
    "n": 2297,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15596,
    15596
   ],
   "32:cold_present_nonempty": [
    2295,
    2297
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15596,
   "2:cold_path_empty": 20,
   "2:no_cold_consumed": 256,
   "32:cold_present_nonempty": 2297,
   "32:cold_path_empty": 5511,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S5_RESOURCE (RESOURCE) all_pass=True

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": true,
  "in": 65886,
  "out": 16384,
  "expected": "65886/16384 (PROBE128, 지시서 §2.2)"
 },
 "config_valid": {
  "pass": true,
  "so": "a4e9045a038bc7af",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE_153555/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_process_begin",
   "client_process_end",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "last_related_deferred_end",
   "log_flush_end",
   "session_begin"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24490
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 1,
  "fwd_missing_share": 4.083299305839118e-05,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24490,
  "def_tasks_expected": 24490,
  "fwd_recorded": 24489,
  "done_store_recorded": 24490,
  "imm_tasks": 395,
  "def_task_seq_found_in_tasks": 24490,
  "def_task_kinds": {
   "3": 24490
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 395,
   "2": 24490,
   "3": 24490
  },
  "expected": {
   "3(deferred)": 24490,
   "2(done)": 24490,
   "1(imm)": 395
  }
 },
 "counter_running_valid": {
  "pass": true,
  "interval_rows": 228,
  "not_counted_rows": 0
 },
 "counter_scope_valid": {
  "pass": true,
  "target": "3757104,3757200,3757361,3757517",
  "note": "process-level -p, 스레드 상속 (perf 문서); TID 별 coverage 는 thread_role_map 와 대조"
 },
 "pcm_artifact_valid": {
  "pass": true,
  "bytes": 30653
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S1_OFF_A (OFF) all_pass=True

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": true,
  "in": 65886,
  "out": 16384,
  "expected": "65886/16384 (PROBE128, 지시서 §2.2)"
 },
 "config_valid": {
  "pass": true,
  "so": "a4e9045a038bc7af",
  "mode": "OFF",
  "env_kt_evt": null
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_process_begin",
   "client_process_end",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "log_flush_end",
   "session_begin"
  ],
  "note": "OFF 는 서버측 창 없음(NOT_COLLECTED)"
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S6_OFF_B (OFF) all_pass=True

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": true,
  "in": 65886,
  "out": 16384,
  "expected": "65886/16384 (PROBE128, 지시서 §2.2)"
 },
 "config_valid": {
  "pass": true,
  "so": "a4e9045a038bc7af",
  "mode": "OFF",
  "env_kt_evt": null
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_process_begin",
   "client_process_end",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "log_flush_end",
   "session_begin"
  ],
  "note": "OFF 는 서버측 창 없음(NOT_COLLECTED)"
 },
 "artifact_valid": {
  "pass": true
 }
}
```

## clock

profiles/clock_alignment_summary.csv. 방법: CPU CLOCK_REALTIME ↔ kineto host 시계 순서검사 (go ≥ dtoh_end). 일정 offset 은 검출 불가 → 5 µs 이내 차이는 CLOCK_INDETERMINATE 로 분류, 민감도 0/1/2/5 µs 전부 보고.

## observer

profiles/observer_overhead_v2.csv (기준 PLAN_RESOLVED §6). 기록 전용 task 0 (validation task_count_valid), 레코드 누락 0 (sampling_valid footer).
