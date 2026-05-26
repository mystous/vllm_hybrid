// IDE_018 — Python bindings for phase-burst scheduler
//
// pybind11 ABI: vllm/v1/worker/gpu_model_runner.py 의 forward path 에서
// (1) phase signal update, (2) scheduler 의 task enqueue, (3) stats query.
//
// import path: `phase_burst._core` (CMake target name).

#include "phase_detector.h"
#include "scheduler.h"
#include "task_pool.h"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

namespace py = pybind11;
using namespace vllm_hybrid_phase;

PYBIND11_MODULE(_core, m) {
    m.doc() = "IDE_018 phase-burst scheduler C++ core";

    // ── Phase enum ────────────────────────────────────────────────
    py::enum_<Phase>(m, "Phase")
        .value("IDLE", PHASE_IDLE)
        .value("ATTENTION", PHASE_ATTENTION)
        .value("LINEAR", PHASE_LINEAR)
        .value("SAMPLE", PHASE_SAMPLE)
        .value("TP_ALLRED", PHASE_TP_ALLRED)
        .value("POST_STEP", PHASE_POST_STEP)
        .export_values();

    // ── TaskKind enum ─────────────────────────────────────────────
    py::enum_<TaskKind>(m, "TaskKind")
        .value("A_SCHEDULE",    TASK_A_SCHEDULE)
        .value("B_DETOKENIZE",  TASK_B_DETOKENIZE)
        .value("C_GRAMMAR",     TASK_C_GRAMMAR)
        .value("D_CLASSIFY",    TASK_D_CLASSIFY)
        .value("E_KV_PREFETCH", TASK_E_KV_PREFETCH)
        .value("F_DRAFT",       TASK_F_DRAFT)
        .value("G_COLDKV",      TASK_G_COLDKV)
        .value("H_SAMPLE",      TASK_H_SAMPLE)
        .value("I_LOGITS",      TASK_I_LOGITS)
        .value("J_PRECOMPUTE",  TASK_J_PRECOMPUTE)
        .export_values();

    // ── PhaseSignal ────────────────────────────────────────────────
    py::class_<PhaseSignal, std::unique_ptr<PhaseSignal, py::nodelete>>(m, "PhaseSignal")
        .def("update",
             [](PhaseSignal& self, uint8_t phase, uint64_t step_id) {
                 self.update(phase, step_id);
             },
             py::arg("phase"), py::arg("step_id") = 0)
        .def("wait_next",
             [](PhaseSignal& self, int timeout_us) {
                 py::gil_scoped_release rel;
                 return self.wait_next(timeout_us);
             },
             py::arg("timeout_us") = -1)
        .def("current", &PhaseSignal::current)
        .def("current_step", &PhaseSignal::current_step)
        .def("ns_in_phase", &PhaseSignal::ns_in_phase)
        .def_readonly("total_updates", &PhaseSignal::total_updates)
        .def_readonly("signal_drops", &PhaseSignal::signal_drops);

    m.def("get_global_signal", &get_or_create_global_signal,
          py::return_value_policy::reference);
    m.def("release_global_signal", &release_global_signal);

    // ── PhaseBurstStats ────────────────────────────────────────────
    py::class_<PhaseBurstStats>(m, "PhaseBurstStats")
        .def_readonly("num_workers", &PhaseBurstStats::num_workers)
        .def_readonly("pending_tasks", &PhaseBurstStats::pending_tasks)
        .def_property_readonly("tasks_executed",
            [](const PhaseBurstStats& s) {
                return py::make_tuple(
                    s.tasks_executed[0], s.tasks_executed[1], s.tasks_executed[2],
                    s.tasks_executed[3], s.tasks_executed[4], s.tasks_executed[5]);
            })
        .def_property_readonly("tasks_skipped",
            [](const PhaseBurstStats& s) {
                return py::make_tuple(
                    s.tasks_skipped[0], s.tasks_skipped[1], s.tasks_skipped[2],
                    s.tasks_skipped[3], s.tasks_skipped[4], s.tasks_skipped[5]);
            })
        .def_property_readonly("avg_dispatch_latency_ns",
            [](const PhaseBurstStats& s) {
                return py::make_tuple(
                    s.avg_dispatch_latency_ns[0], s.avg_dispatch_latency_ns[1],
                    s.avg_dispatch_latency_ns[2], s.avg_dispatch_latency_ns[3],
                    s.avg_dispatch_latency_ns[4], s.avg_dispatch_latency_ns[5]);
            });

    // ── Scheduler ──────────────────────────────────────────────────
    py::class_<PhaseBurstScheduler>(m, "PhaseBurstScheduler")
        .def(py::init([](PhaseSignal* sig, int num_workers, int cpu_base) {
                 return std::make_unique<PhaseBurstScheduler>(sig, num_workers, cpu_base);
             }),
             py::arg("signal"), py::arg("num_workers") = 20, py::arg("cpu_base") = 80)
        .def("start", &PhaseBurstScheduler::start)
        .def("stop",  &PhaseBurstScheduler::stop)
        .def("snapshot_stats", &PhaseBurstScheduler::snapshot_stats)
        .def("enqueue_python_callable",
             [](PhaseBurstScheduler& self, TaskKind kind, uint64_t step_id,
                uint8_t applicable_phases, py::function fn) {
                 // GIL: Python callable invocation reacquires GIL on worker thread.
                 auto pyfn = std::make_shared<py::function>(std::move(fn));
                 Task t{
                     kind,
                     step_id,
                     applicable_phases,
                     /*fn=*/ [pyfn]() {
                         py::gil_scoped_acquire acq;
                         try { (*pyfn)(); } catch (...) { /* swallow */ }
                     },
                     /*enqueued_ns=*/ 0,
                 };
                 self.enqueue(std::move(t));
             },
             py::arg("kind"), py::arg("step_id"),
             py::arg("applicable_phases"), py::arg("fn"));

    // ── Stub-handle accessors (microbench / test) ─────────────────
    m.def("attention_pool_stub_invocation_count",
          &attention_pool::stub_invocation_count);
    m.def("linear_pool_stub_invocation_count",
          &linear_pool::stub_invocation_count);

    // ── Phase mask constants exposed to Python ─────────────────────
    m.attr("MASK_ATTN")   = uint8_t(MASK_ATTN);
    m.attr("MASK_LINEAR") = uint8_t(MASK_LINEAR);
    m.attr("MASK_SAMPLE") = uint8_t(MASK_SAMPLE);
    m.attr("MASK_TP_AR")  = uint8_t(MASK_TP_AR);
    m.attr("MASK_IDLE")   = uint8_t(MASK_IDLE);
    m.attr("MASK_POST")   = uint8_t(MASK_POST);
    m.attr("MASK_ANY")    = uint8_t(MASK_ANY);
}
