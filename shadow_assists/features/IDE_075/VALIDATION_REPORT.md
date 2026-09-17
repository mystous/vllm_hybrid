# VALIDATION_REPORT — IDE_075 (2026-09-17T20:53:53+09:00)

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

### S10_CORR (CORR) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE2_162053/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24428
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24428,
  "def_tasks_expected": 24428,
  "fwd_recorded": 24428,
  "done_store_recorded": 24428,
  "imm_tasks": 394,
  "def_task_seq_found_in_tasks": 24428,
  "def_task_kinds": {
   "3": 24428
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 394,
   "2": 24428,
   "3": 24428
  },
  "expected": {
   "3(deferred)": 24428,
   "2(done)": 24428,
   "1(imm)": 394
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "087-TP-0.trace.json.gz": {
    "kernel": 690868,
    "memcpy": 124834
   },
   "087-TP-1.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   },
   "087-TP-2.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   },
   "087-TP-3.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 23870,
   "method:count_match_zip": 23808,
   "count_mismatch_pairs": 62,
   "count_mismatch_rows": 62,
   "resync_unmatched_gpu": 62,
   "resync_unmatched_cpu": 0,
   "resync_pairs": 0
  },
  "matched": 23808,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 0
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 23808
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.9493965016459046,
    "late_share_thr_1us": 0.9492674110888788,
    "late_share_thr_2us": 0.949202865810366,
    "late_share_thr_5us": 0.9482992319111857,
    "clock_indeterminate_share_5us": 0.00180726779836055,
    "n": 15493,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.09800718719372754,
    "late_share_thr_1us": 0.09637373407383208,
    "late_share_thr_2us": 0.09539366220189481,
    "late_share_thr_5us": 0.09212675596210389,
    "clock_indeterminate_share_5us": 0.00914733747141457,
    "n": 3061,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15493,
    15493
   ],
   "32:cold_present_nonempty": [
    3061,
    3061
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15493,
   "2:cold_path_empty": 123,
   "2:no_cold_consumed": 256,
   "32:cold_present_nonempty": 3061,
   "32:cold_path_empty": 4747,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S11_RESOURCE (RESOURCE) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE2_162053/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24428
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24428,
  "def_tasks_expected": 24428,
  "fwd_recorded": 24428,
  "done_store_recorded": 24428,
  "imm_tasks": 394,
  "def_task_seq_found_in_tasks": 24428,
  "def_task_kinds": {
   "3": 24428
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 394,
   "2": 24428,
   "3": 24428
  },
  "expected": {
   "3(deferred)": 24428,
   "2(done)": 24428,
   "1(imm)": 394
  }
 },
 "counter_running_valid": {
  "pass": true,
  "interval_rows": 234,
  "not_counted_rows": 0
 },
 "counter_scope_valid": {
  "pass": true,
  "target": "4013165,4013235,4013320,4013424",
  "note": "process-level -p, 스레드 상속 (perf 문서); TID 별 coverage 는 thread_role_map 와 대조"
 },
 "pcm_artifact_valid": {
  "pass": true,
  "bytes": 31502
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S12_C1_CORR (CORR) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 16,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE2_162053/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 127968
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 127968,
  "def_tasks_expected": 127968,
  "fwd_recorded": 127968,
  "done_store_recorded": 127968,
  "imm_tasks": 2064,
  "def_task_seq_found_in_tasks": 127968,
  "def_task_kinds": {
   "3": 127968
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 2064,
   "2": 127968,
   "3": 127968
  },
  "expected": {
   "3(deferred)": 127968,
   "2(done)": 127968,
   "1(imm)": 2064
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "257-TP-0.trace.json.gz": {
    "kernel": 3507035,
    "memcpy": 645910
   },
   "257-TP-1.trace.json.gz": {
    "kernel": 2995691,
    "memcpy": 4396
   },
   "257-TP-2.trace.json.gz": {
    "kernel": 2995691,
    "memcpy": 4396
   },
   "257-TP-3.trace.json.gz": {
    "kernel": 2995691,
    "memcpy": 4396
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 127038,
   "count_mismatch_pairs": 62,
   "count_mismatch_rows": 62,
   "resync_unmatched_gpu": 62,
   "resync_unmatched_cpu": 0,
   "resync_pairs": 126976,
   "method:ANCHOR_RESYNC(go>=dtoh_end<next)": 126976
  },
  "matched": 126976,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 0
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 126976
  },
  "sensitivity": {
   "35:cold_present_nonempty": {
    "late_share_thr_0us": 0.18571888066902542,
    "late_share_thr_1us": 0.18404631714377614,
    "late_share_thr_2us": 0.18245952610700117,
    "late_share_thr_5us": 0.17823523104964084,
    "clock_indeterminate_share_5us": 0.017883563846896107,
    "n": 46635,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "35:cold_present_nonempty": [
    46635,
    46635
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "35:cold_present_nonempty": 46635,
   "35:cold_path_empty": 78293,
   "35:no_cold_consumed": 2048
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S13_LONG_CORR (CORR) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 32,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE2_162053/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 32922
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 32922,
  "def_tasks_expected": 32922,
  "fwd_recorded": 32922,
  "done_store_recorded": 32922,
  "imm_tasks": 531,
  "def_task_seq_found_in_tasks": 32922,
  "def_task_kinds": {
   "3": 32922
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 531,
   "2": 32922,
   "3": 32922
  },
  "expected": {
   "3(deferred)": 32922,
   "2(done)": 32922,
   "1(imm)": 531
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "642-TP-0.trace.json.gz": {
    "kernel": 907209,
    "memcpy": 167909
   },
   "642-TP-1.trace.json.gz": {
    "kernel": 776241,
    "memcpy": 1439
   },
   "642-TP-2.trace.json.gz": {
    "kernel": 776241,
    "memcpy": 1439
   },
   "642-TP-3.trace.json.gz": {
    "kernel": 776241,
    "memcpy": 1439
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 31806,
   "method:count_match_zip": 31744,
   "count_mismatch_pairs": 62,
   "count_mismatch_rows": 62,
   "resync_unmatched_gpu": 62,
   "resync_unmatched_cpu": 0,
   "resync_pairs": 0
  },
  "matched": 31744,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 0
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 31744
  },
  "sensitivity": {
   "26:cold_present_nonempty": {
    "late_share_thr_0us": 0.5277708857618828,
    "late_share_thr_1us": 0.5264239889412682,
    "late_share_thr_2us": 0.5251479814270017,
    "late_share_thr_5us": 0.5205756211675469,
    "clock_indeterminate_share_5us": 0.013185410980753553,
    "n": 28213,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "26:cold_present_nonempty": [
    28213,
    28213
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "26:cold_present_nonempty": 28213,
   "26:cold_path_empty": 3019,
   "26:no_cold_consumed": 512
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S14_FOCUS_sched (FOCUS) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE2_162053/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24428
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24428,
  "def_tasks_expected": 24428,
  "fwd_recorded": 24428,
  "done_store_recorded": 24428,
  "imm_tasks": 394,
  "def_task_seq_found_in_tasks": 24428,
  "def_task_kinds": {
   "3": 24428
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 394,
   "2": 24428,
   "3": 24428
  },
  "expected": {
   "3(deferred)": 24428,
   "2(done)": 24428,
   "1(imm)": 394
  }
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S9_CORR (CORR) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE2_162053/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24428
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24428,
  "def_tasks_expected": 24428,
  "fwd_recorded": 24428,
  "done_store_recorded": 24428,
  "imm_tasks": 394,
  "def_task_seq_found_in_tasks": 24428,
  "def_task_kinds": {
   "3": 24428
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 394,
   "2": 24428,
   "3": 24428
  },
  "expected": {
   "3(deferred)": 24428,
   "2(done)": 24428,
   "1(imm)": 394
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "816-TP-0.trace.json.gz": {
    "kernel": 1411364,
    "memcpy": 255482
   },
   "816-TP-1.trace.json.gz": {
    "kernel": 1212506,
    "memcpy": 2894
   },
   "816-TP-2.trace.json.gz": {
    "kernel": 1212506,
    "memcpy": 2894
   },
   "816-TP-3.trace.json.gz": {
    "kernel": 1212506,
    "memcpy": 2894
   }
  }
 },
 "mapping_valid": {
  "pass": false,
  "coverage": {
   "gpu_layer_rows_total": 48360,
   "count_mismatch_pairs": 124,
   "count_mismatch_rows": 24552,
   "resync_unmatched_gpu": 24552,
   "resync_unmatched_cpu": 0,
   "resync_pairs": 0,
   "method:count_match_zip": 23808
  },
  "matched": 23808,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 0
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 23808
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.9627437101859597,
    "late_share_thr_1us": 0.9625506724149026,
    "late_share_thr_2us": 0.9625506724149026,
    "late_share_thr_5us": 0.9621002509491023,
    "clock_indeterminate_share_5us": 0.0012869184737146903,
    "n": 15541,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.09702711532179026,
    "late_share_thr_1us": 0.09572035282587389,
    "late_share_thr_2us": 0.09572035282587389,
    "late_share_thr_5us": 0.09441359032995753,
    "clock_indeterminate_share_5us": 0.006207121855602744,
    "n": 3061,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15541,
    15541
   ],
   "32:cold_present_nonempty": [
    3061,
    3061
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15541,
   "2:cold_path_empty": 75,
   "2:no_cold_consumed": 256,
   "32:cold_present_nonempty": 3061,
   "32:cold_path_empty": 4747,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S17_CORR_pinned (CORR) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE3_171618/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24428
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24428,
  "def_tasks_expected": 24428,
  "fwd_recorded": 24428,
  "done_store_recorded": 24428,
  "imm_tasks": 394,
  "def_task_seq_found_in_tasks": 24428,
  "def_task_kinds": {
   "3": 24428
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 394,
   "2": 24428,
   "3": 24428
  },
  "expected": {
   "3(deferred)": 24428,
   "2(done)": 24428,
   "1(imm)": 394
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "145-TP-0.trace.json.gz": {
    "kernel": 690868,
    "memcpy": 124834
   },
   "145-TP-1.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   },
   "145-TP-2.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   },
   "145-TP-3.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 23870,
   "method:count_match_zip": 23808,
   "count_mismatch_pairs": 62,
   "count_mismatch_rows": 62,
   "resync_unmatched_gpu": 62,
   "resync_unmatched_cpu": 0,
   "resync_pairs": 0
  },
  "matched": 23808,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 0
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 23808
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.9526883108500613,
    "late_share_thr_1us": 0.9525592202930355,
    "late_share_thr_2us": 0.9524301297360098,
    "late_share_thr_5us": 0.9517846769508811,
    "clock_indeterminate_share_5us": 0.0014199961272832891,
    "n": 15493,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.09702711532179026,
    "late_share_thr_1us": 0.09637373407383208,
    "late_share_thr_2us": 0.09539366220189481,
    "late_share_thr_5us": 0.09408689970597844,
    "clock_indeterminate_share_5us": 0.006207121855602744,
    "n": 3061,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15493,
    15493
   ],
   "32:cold_present_nonempty": [
    3061,
    3061
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15493,
   "2:cold_path_empty": 123,
   "2:no_cold_consumed": 256,
   "32:cold_present_nonempty": 3061,
   "32:cold_path_empty": 4747,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S18_CORR_pinned (CORR) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE3_171618/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24428
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24428,
  "def_tasks_expected": 24428,
  "fwd_recorded": 24428,
  "done_store_recorded": 24428,
  "imm_tasks": 394,
  "def_task_seq_found_in_tasks": 24428,
  "def_task_kinds": {
   "3": 24428
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 394,
   "2": 24428,
   "3": 24428
  },
  "expected": {
   "3(deferred)": 24428,
   "2(done)": 24428,
   "1(imm)": 394
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "268-TP-0.trace.json.gz": {
    "kernel": 690868,
    "memcpy": 124834
   },
   "268-TP-1.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   },
   "268-TP-2.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   },
   "268-TP-3.trace.json.gz": {
    "kernel": 593330,
    "memcpy": 1392
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 23870,
   "method:count_match_zip": 23808,
   "count_mismatch_pairs": 62,
   "count_mismatch_rows": 62,
   "resync_unmatched_gpu": 62,
   "resync_unmatched_cpu": 0,
   "resync_pairs": 0
  },
  "matched": 23808,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 0
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 23808
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.9524946750145227,
    "late_share_thr_1us": 0.9521719486219583,
    "late_share_thr_2us": 0.9521719486219583,
    "late_share_thr_5us": 0.9516555863938553,
    "clock_indeterminate_share_5us": 0.0016136319628219196,
    "n": 15493,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.1035609278013721,
    "late_share_thr_1us": 0.10290754655341391,
    "late_share_thr_2us": 0.10160078405749755,
    "late_share_thr_5us": 0.099640640313623,
    "clock_indeterminate_share_5us": 0.007840574975498203,
    "n": 3061,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15493,
    15493
   ],
   "32:cold_present_nonempty": [
    3061,
    3061
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15493,
   "2:cold_path_empty": 123,
   "2:no_cold_consumed": 256,
   "32:cold_present_nonempty": 3061,
   "32:cold_path_empty": 4747,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S19_FOCUS_pinned (FOCUS) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE3_171618/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
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
 "artifact_valid": {
  "pass": false
 }
}
```

### S20_RESOURCE_pinned (RESOURCE) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE3_171618/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 24428
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 24428,
  "def_tasks_expected": 24428,
  "fwd_recorded": 24428,
  "done_store_recorded": 24428,
  "imm_tasks": 394,
  "def_task_seq_found_in_tasks": 24428,
  "def_task_kinds": {
   "3": 24428
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 394,
   "2": 24428,
   "3": 24428
  },
  "expected": {
   "3(deferred)": 24428,
   "2(done)": 24428,
   "1(imm)": 394
  }
 },
 "counter_running_valid": {
  "pass": true,
  "interval_rows": 240,
  "not_counted_rows": 0
 },
 "counter_scope_valid": {
  "pass": true,
  "target": "171924,172101,172168,172332",
  "note": "process-level -p, 스레드 상속 (perf 문서); TID 별 coverage 는 thread_role_map 와 대조"
 },
 "pcm_artifact_valid": {
  "pass": true,
  "bytes": 31502
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S22_CORR_vllmw (CORR) all_pass=False

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
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE4_174930/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
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
   "133-TP-0.trace.json.gz": {
    "kernel": 699064,
    "memcpy": 125602
   },
   "133-TP-1.trace.json.gz": {
    "kernel": 601338,
    "memcpy": 1788
   },
   "133-TP-2.trace.json.gz": {
    "kernel": 601338,
    "memcpy": 1788
   },
   "133-TP-3.trace.json.gz": {
    "kernel": 601338,
    "memcpy": 1788
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 23870,
   "count_mismatch_pairs": 86,
   "count_mismatch_rows": 86,
   "resync_unmatched_gpu": 216,
   "resync_gpu_rows_without_dtoh": 1,
   "resync_unmatched_cpu": 155,
   "resync_pairs": 15717,
   "method:ANCHOR_RESYNC(go>=dtoh_end<next)": 6084,
   "zip_order_violation_groups": 38,
   "method:ANCHOR_RESYNC(order_violation)": 9633,
   "method:count_match_zip": 7936
  },
  "matched": 23653,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 38
 },
 "clock_valid_by_pair": {
  "pass": false,
  "counts": {
   "OK": 22668,
   "CLOCK_INDETERMINATE": 966,
   "CLOCK_ORDER_VIOLATION": 19
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.9162881245944192,
    "late_share_thr_1us": 0.9159636599610642,
    "late_share_thr_2us": 0.9152498377676833,
    "late_share_thr_5us": 0.9136924075275795,
    "clock_indeterminate_share_5us": 0.006554185593770279,
    "n": 15410,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.01989443767762891,
    "late_share_thr_1us": 0.01989443767762891,
    "late_share_thr_2us": 0.01989443767762891,
    "late_share_thr_5us": 0.0194884287454324,
    "clock_indeterminate_share_5us": 0.0016240357287860333,
    "n": 2463,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15410,
    15410
   ],
   "32:cold_present_nonempty": [
    2463,
    2463
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15410,
   "2:cold_path_empty": 16,
   "2:no_cold_consumed": 255,
   "2:unknown_path_or_mapping": 36,
   "32:cold_present_nonempty": 2463,
   "32:cold_path_empty": 5345,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S23_CORR_vllmw (CORR) all_pass=True

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
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE4_174930/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
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
   "258-TP-0.trace.json.gz": {
    "kernel": 699003,
    "memcpy": 125600
   },
   "258-TP-1.trace.json.gz": {
    "kernel": 601277,
    "memcpy": 1786
   },
   "258-TP-2.trace.json.gz": {
    "kernel": 601277,
    "memcpy": 1786
   },
   "258-TP-3.trace.json.gz": {
    "kernel": 601277,
    "memcpy": 1786
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 23870,
   "count_mismatch_pairs": 78,
   "count_mismatch_rows": 78,
   "resync_gpu_rows_without_dtoh": 1,
   "resync_unmatched_cpu": 54,
   "resync_pairs": 15818,
   "method:ANCHOR_RESYNC(go>=dtoh_end<next)": 4088,
   "resync_unmatched_gpu": 115,
   "zip_order_violation_groups": 46,
   "method:ANCHOR_RESYNC(order_violation)": 11730,
   "method:count_match_zip": 7936
  },
  "matched": 23754,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 46
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 23754
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.910816549771572,
    "late_share_thr_1us": 0.9101730905347146,
    "late_share_thr_2us": 0.9094652853741716,
    "late_share_thr_5us": 0.9072131780451709,
    "clock_indeterminate_share_5us": 0.006691976063316389,
    "n": 15541,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.0385894876912841,
    "late_share_thr_1us": 0.03792415169660679,
    "late_share_thr_2us": 0.036593479707252165,
    "late_share_thr_5us": 0.03592814371257485,
    "clock_indeterminate_share_5us": 0.005988023952095809,
    "n": 3006,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15541,
    15541
   ],
   "32:cold_present_nonempty": [
    3006,
    3006
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15541,
   "2:cold_path_empty": 21,
   "2:no_cold_consumed": 256,
   "32:cold_present_nonempty": 3006,
   "32:cold_path_empty": 4802,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S24_RESOURCE_vllmw (RESOURCE) all_pass=True

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
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE4_174930/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
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
 "counter_running_valid": {
  "pass": true,
  "interval_rows": 234,
  "not_counted_rows": 0
 },
 "counter_scope_valid": {
  "pass": true,
  "target": "415714,415784,415871,416077",
  "note": "process-level -p, 스레드 상속 (perf 문서); TID 별 coverage 는 thread_role_map 와 대조"
 },
 "pcm_artifact_valid": {
  "pass": true,
  "bytes": 31466
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S25_FOCUS_vllmw (FOCUS) all_pass=True

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
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE4_174930/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
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
 "artifact_valid": {
  "pass": true
 }
}
```

### S26_C1_CORR_vllmw (CORR) all_pass=True

```
{
 "execution_valid": {
  "pass": true,
  "completed": 16,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE4_174930/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 127968
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 127968,
  "def_tasks_expected": 127968,
  "fwd_recorded": 127968,
  "done_store_recorded": 127968,
  "imm_tasks": 2064,
  "def_task_seq_found_in_tasks": 127968,
  "def_task_kinds": {
   "3": 127968
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 2064,
   "2": 127968,
   "3": 127968
  },
  "expected": {
   "3(deferred)": 127968,
   "2(done)": 127968,
   "1(imm)": 2064
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "506-TP-0.trace.json.gz": {
    "kernel": 3534734,
    "memcpy": 649338
   },
   "506-TP-1.trace.json.gz": {
    "kernel": 3023824,
    "memcpy": 8506
   },
   "506-TP-2.trace.json.gz": {
    "kernel": 3023824,
    "memcpy": 8506
   },
   "506-TP-3.trace.json.gz": {
    "kernel": 3023824,
    "memcpy": 8506
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 126976,
   "method:count_match_zip": 126976
  },
  "matched": 126976,
  "method": "VALIDATED_HEURISTIC_MAPPING",
  "zip_order_violation_groups": 0
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 126976
  },
  "sensitivity": {
   "35:cold_present_nonempty": {
    "late_share_thr_0us": 0.01816332461493752,
    "late_share_thr_1us": 0.017485227162646517,
    "late_share_thr_2us": 0.017049307371888017,
    "late_share_thr_5us": 0.015789983531919016,
    "clock_indeterminate_share_5us": 0.005666957279860506,
    "n": 20646,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "35:cold_present_nonempty": [
    20646,
    20646
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "35:cold_present_nonempty": 20646,
   "35:cold_path_empty": 104282,
   "35:no_cold_consumed": 2048
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S27_LONG_CORR_vllmw (CORR) all_pass=True

```
{
 "execution_valid": {
  "pass": true,
  "completed": 32,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/CORE4_174930/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": null
 },
 "sampling_valid": {
  "pass": true,
  "footer": [],
  "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음",
  "records_in_session": 32984
 },
 "lifecycle_valid": {
  "pass": true,
  "fwd_missing": 0,
  "fwd_missing_share": 0.0,
  "criterion": "미기록 ≤ 0.1 % (건수 공개)",
  "packets": 32984,
  "def_tasks_expected": 32984,
  "fwd_recorded": 32984,
  "done_store_recorded": 32984,
  "imm_tasks": 532,
  "def_task_seq_found_in_tasks": 32984,
  "def_task_kinds": {
   "3": 32984
  },
  "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"
 },
 "task_count_valid": {
  "pass": true,
  "tasks_in_window_by_kind": {
   "1": 532,
   "2": 32984,
   "3": 32984
  },
  "expected": {
   "3(deferred)": 32984,
   "2(done)": 32984,
   "1(imm)": 532
  }
 },
 "gpu_activity_valid": {
  "pass": true,
  "per_trace": {
   "910-TP-0.trace.json.gz": {
    "kernel": 914313,
    "memcpy": 168135
   },
   "910-TP-1.trace.json.gz": {
    "kernel": 783593,
    "memcpy": 1975
   },
   "910-TP-2.trace.json.gz": {
    "kernel": 783593,
    "memcpy": 1975
   },
   "910-TP-3.trace.json.gz": {
    "kernel": 783593,
    "memcpy": 1975
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 31744,
   "count_mismatch_pairs": 12,
   "count_mismatch_rows": 12,
   "resync_gpu_rows_without_dtoh": 1,
   "resync_unmatched_cpu": 56,
   "resync_pairs": 31688,
   "method:ANCHOR_RESYNC(go>=dtoh_end<next)": 6138,
   "resync_unmatched_gpu": 55,
   "zip_order_violation_groups": 50,
   "method:ANCHOR_RESYNC(order_violation)": 25550
  },
  "matched": 31688,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 50
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 30618,
   "CLOCK_INDETERMINATE": 1070
  },
  "sensitivity": {
   "26:cold_present_nonempty": {
    "late_share_thr_0us": 0.030434966394724604,
    "late_share_thr_1us": 0.030181341674768568,
    "late_share_thr_2us": 0.02984317538149385,
    "late_share_thr_5us": 0.029082301221625733,
    "clock_indeterminate_share_5us": 0.00312803821279114,
    "n": 23657,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "26:cold_present_nonempty": [
    23657,
    23657
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "26:cold_present_nonempty": 23657,
   "26:cold_path_empty": 7519,
   "26:no_cold_consumed": 512
  },
  "residual_max": 0
 },
 "artifact_valid": {
  "pass": true
 }
}
```

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
  "method": "VALIDATED_HEURISTIC_MAPPING",
  "zip_order_violation_groups": 0
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
  "method": "VALIDATED_HEURISTIC_MAPPING",
  "zip_order_violation_groups": 0
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

### S28_FOCUS2_vllmw_syswide (FOCUS) all_pass=True

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
  "so": "12926df2c30b7246",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/FOCUS2_183951/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
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
 "artifact_valid": {
  "pass": true
 }
}
```

### S29_CORR_vllmw_fp8fixso (CORR) all_pass=True

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
  "so": "12926df2c30b7246",
  "mode": "CORE",
  "env_kt_evt": "/models/kt/ide075/FOCUS2_183951/kt_evt.csv"
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_go",
   "first_request_sent",
   "last_related_deferred_end",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
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
   "259-TP-0.trace.json.gz": {
    "kernel": 699127,
    "memcpy": 125599
   },
   "259-TP-1.trace.json.gz": {
    "kernel": 601401,
    "memcpy": 1785
   },
   "259-TP-2.trace.json.gz": {
    "kernel": 601401,
    "memcpy": 1785
   },
   "259-TP-3.trace.json.gz": {
    "kernel": 601401,
    "memcpy": 1785
   }
  }
 },
 "mapping_valid": {
  "pass": true,
  "coverage": {
   "gpu_layer_rows_total": 23870,
   "method:count_match_zip": 23808,
   "count_mismatch_pairs": 62,
   "count_mismatch_rows": 62,
   "resync_unmatched_gpu": 62,
   "resync_unmatched_cpu": 0,
   "resync_pairs": 0
  },
  "matched": 23808,
  "method": "VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)",
  "zip_order_violation_groups": 0
 },
 "clock_valid_by_pair": {
  "pass": true,
  "counts": {
   "OK": 23808
  },
  "sensitivity": {
   "2:cold_present_nonempty": {
    "late_share_thr_0us": 0.911025641025641,
    "late_share_thr_1us": 0.9103205128205129,
    "late_share_thr_2us": 0.9095512820512821,
    "late_share_thr_5us": 0.9076923076923077,
    "clock_indeterminate_share_5us": 0.006346153846153846,
    "n": 15600,
    "cold_gpu_ready_late_share": 1.0
   },
   "32:cold_present_nonempty": {
    "late_share_thr_0us": 0.01715794104707435,
    "late_share_thr_1us": 0.01715794104707435,
    "late_share_thr_2us": 0.01715794104707435,
    "late_share_thr_5us": 0.01627804663440387,
    "clock_indeterminate_share_5us": 0.0017597888253409592,
    "n": 2273,
    "cold_gpu_ready_late_share": 1.0
   }
  }
 },
 "producer_consumer_valid": {
  "pass": true,
  "linked_over_n": {
   "2:cold_present_nonempty": [
    15600,
    15600
   ],
   "32:cold_present_nonempty": [
    2273,
    2273
   ]
  },
  "criterion": "연결 ≥ 99.9 % (건수 공개)",
  "cohorts": {
   "2:cold_present_nonempty": 15600,
   "2:cold_path_empty": 16,
   "2:no_cold_consumed": 256,
   "32:cold_present_nonempty": 2273,
   "32:cold_path_empty": 5535,
   "32:no_cold_consumed": 128
  },
  "residual_max": 0
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

### S15_OFF_CLOSE2 (OFF) all_pass=True

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
  "so": "d659ca0b274ca469",
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

### S15b_OFF_CLOSE2_probe (OFF) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "OFF",
  "env_kt_evt": null
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_request_sent",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": "OFF 는 서버측 창 없음(NOT_COLLECTED)"
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S8_OFF_OPEN2 (OFF) all_pass=True

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
  "so": "d659ca0b274ca469",
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

### S8b_OFF_OPEN2_probe (OFF) all_pass=False

```
{
 "execution_valid": {
  "pass": false,
  "completed": null,
  "failed": null,
  "rc": 1
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "OFF",
  "env_kt_evt": null
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": "OFF 는 서버측 창 없음(NOT_COLLECTED)"
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S16_OFF_X_vllm (OFF) all_pass=True

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
  "so": "d659ca0b274ca469",
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

### S16b_OFF_X_probe_pinned (OFF) all_pass=False

```
{
 "execution_valid": {
  "pass": true,
  "completed": 128,
  "failed": 0,
  "rc": 0
 },
 "workload_valid": {
  "pass": null,
  "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"
 },
 "config_valid": {
  "pass": true,
  "so": "d659ca0b274ca469",
  "mode": "OFF",
  "env_kt_evt": null
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_request_sent",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": "OFF 는 서버측 창 없음(NOT_COLLECTED)"
 },
 "artifact_valid": {
  "pass": false
 }
}
```

### S16c_OFF_X_vllm2 (OFF) all_pass=True

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
  "so": "d659ca0b274ca469",
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

### S21_OFF_Y_vllm (OFF) all_pass=True

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
  "so": "d659ca0b274ca469",
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

### S21b_OFF_Y_vllmw (OFF) all_pass=True

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
  "so": "d659ca0b274ca469",
  "mode": "OFF",
  "env_kt_evt": null
 },
 "window_valid": {
  "pass": true,
  "events": [
   "client_init_begin",
   "client_process_end",
   "client_ready",
   "collector_arm",
   "collector_stop",
   "drain_confirmed",
   "engine_cache_flushed",
   "first_request_sent",
   "last_response_received",
   "log_flush_end",
   "session_begin",
   "start_signal_send"
  ],
  "note": "OFF 는 서버측 창 없음(NOT_COLLECTED)"
 },
 "artifact_valid": {
  "pass": true
 }
}
```

### S21c_OFF_Y_vllm2 (OFF) all_pass=True

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
  "so": "d659ca0b274ca469",
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
