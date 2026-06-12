#!/usr/bin/env python3
"""IDE_026 / TSK_048 T0 — resctrl 제어·판독 도구.

파일 I/O 만 사용, 외부 의존 0. resctrl 루트 경로를 인자로 받으므로
호스트에서 직접 실행하든 bind-mount 된 컨테이너에서 실행하든 동일 동작.

사용 예 (root 필요):
    # HW capability 출력
    rdt_ctl.py info
    # CLOS 그룹 생성 + schemata 기록 (serving 상위 way / harvest 하위 4 way)
    rdt_ctl.py setup --group serving --l3-ways top   --mb 100
    rdt_ctl.py setup --group harvest --l3-ways low4  --mb 20
    # 코어/스레드 등록
    rdt_ctl.py assign --group harvest --cpus 16-55
    rdt_ctl.py assign --group harvest --tids 1234,1235
    # 주기 판독 (mbm 은 delta→GB/s 환산, llc_occupancy 는 절대 MB)
    rdt_ctl.py mon --groups serving,harvest --interval 1 --duration 30 --csv mon.csv
    # 정합 self-test (G0)
    rdt_ctl.py selftest
    # 정리
    rdt_ctl.py teardown --group serving --group harvest
"""

import argparse
import os
import sys
import time


class Resctrl:
    def __init__(self, root="/sys/fs/resctrl"):
        self.root = root.rstrip("/")

    # ---------- 기본 ----------

    def mounted(self):
        return os.path.isdir(os.path.join(self.root, "info"))

    def require_mounted(self):
        if not self.mounted():
            sys.exit(
                f"[rdt_ctl] {self.root} 가 resctrl 마운트가 아님.\n"
                f"  호스트에서: sudo mount -t resctrl resctrl /sys/fs/resctrl\n"
                f"  (컨테이너 내 mount 는 CAP_SYS_ADMIN 부재로 불가 — HOST_RUNBOOK.md 참조)"
            )

    def _read(self, *parts):
        with open(os.path.join(self.root, *parts)) as f:
            return f.read().strip()

    def _write(self, value, *parts):
        path = os.path.join(self.root, *parts)
        with open(path, "w") as f:
            f.write(value)

    def _try_read(self, *parts):
        try:
            return self._read(*parts)
        except OSError:
            return None

    # ---------- info ----------

    def info(self):
        out = {}
        for res, keys in (
            ("L3", ["cbm_mask", "min_cbm_bits", "num_closids", "shareable_bits"]),
            ("MB", ["min_bandwidth", "bandwidth_gran", "num_closids", "delay_linear"]),
            ("L3_MON", ["mon_features", "num_rmids", "max_threshold_occupancy"]),
        ):
            d = {}
            for k in keys:
                v = self._try_read("info", res, k)
                if v is not None:
                    d[k] = v
            if d:
                out[res] = d
        return out

    def cbm_width(self):
        mask = self._read("info", "L3", "cbm_mask")
        return int(mask, 16).bit_length()

    def domains(self, resource="L3"):
        """루트 그룹 schemata 에서 도메인 id 목록 추출 (dual-socket → [0, 1])."""
        for line in self._read("schemata").splitlines():
            line = line.strip()
            if line.startswith(resource + ":"):
                return [tok.split("=")[0] for tok in line.split(":", 1)[1].split(";")]
        return []

    # ---------- CLOS 그룹 ----------

    def group_path(self, group):
        return self.root if group in ("", ".", "root") else os.path.join(self.root, group)

    def create(self, group):
        path = self.group_path(group)
        if not os.path.isdir(path):
            os.mkdir(path)

    def delete(self, group):
        path = self.group_path(group)
        if path != self.root and os.path.isdir(path):
            os.rmdir(path)  # resctrl 그룹 삭제는 rmdir (tasks 는 루트로 환원됨)

    def set_schemata(self, group, lines):
        """lines: ['L3:0=ffff0;1=ffff0', 'MB:0=100;1=100'] — 한 줄씩 기록."""
        for line in lines:
            self._write(line + "\n", group, "schemata")

    def get_schemata(self, group):
        return self._read(group, "schemata")

    def assign_tasks(self, group, tids):
        for tid in tids:
            self._write(str(tid) + "\n", group, "tasks")

    def get_tasks(self, group):
        txt = self._try_read(group, "tasks")
        return [int(t) for t in txt.split()] if txt else []

    def assign_cpus(self, group, cpulist):
        self._write(cpulist + "\n", group, "cpus_list")

    # ---------- 모니터링 ----------

    def read_mon(self, group):
        """{domain: {event: int}} — mbm_* 는 누적 bytes, llc_occupancy 는 절대 bytes."""
        base = os.path.join(self.group_path(group), "mon_data")
        out = {}
        if not os.path.isdir(base):
            return out
        for dom in sorted(os.listdir(base)):  # mon_L3_00, mon_L3_01, ...
            ddir = os.path.join(base, dom)
            if not os.path.isdir(ddir):
                continue
            ev = {}
            for f in os.listdir(ddir):
                try:
                    with open(os.path.join(ddir, f)) as fh:
                        ev[f] = int(fh.read().strip())
                except (OSError, ValueError):
                    pass
            out[dom] = ev
        return out

    def monitor(self, groups, interval, duration, csv_path=None, quiet=False):
        """주기 판독. mbm_* 는 delta/interval → GB/s, llc_occupancy → MB."""
        csv = open(csv_path, "w") if csv_path else None
        if csv:
            csv.write("ts,group,domain,llc_occ_MB,mbm_total_GBps,mbm_local_GBps\n")
        prev = {g: self.read_mon(g) for g in groups}
        t0 = time.monotonic()
        rows_emitted = 0
        try:
            while time.monotonic() - t0 < duration:
                time.sleep(interval)
                now = time.monotonic() - t0
                for g in groups:
                    cur = self.read_mon(g)
                    for dom, ev in cur.items():
                        pv = prev.get(g, {}).get(dom, {})
                        occ_mb = ev.get("llc_occupancy", 0) / (1 << 20)
                        def gbps(key):
                            if key in ev and key in pv:
                                return max(0, ev[key] - pv[key]) / interval / 1e9
                            return float("nan")
                        tot, loc = gbps("mbm_total_bytes"), gbps("mbm_local_bytes")
                        line = (f"{now:.1f},{g},{dom},{occ_mb:.1f},"
                                f"{tot:.2f},{loc:.2f}")
                        if csv:
                            csv.write(line + "\n")
                        if not quiet:
                            print(line)
                        rows_emitted += 1
                    prev[g] = cur
                if csv:
                    csv.flush()
        finally:
            if csv:
                csv.close()
        return rows_emitted

    # ---------- way mask 생성 ----------

    def make_masks(self, harvest_ways=4):
        """비중첩 분할: harvest = 하위 N way, serving = 나머지 상위 way."""
        w = self.cbm_width()
        min_bits = int(self._try_read("info", "L3", "min_cbm_bits") or 1)
        assert harvest_ways >= min_bits, f"harvest_ways < min_cbm_bits({min_bits})"
        assert w - harvest_ways >= min_bits, "serving way 부족"
        harvest = (1 << harvest_ways) - 1
        serving = ((1 << w) - 1) & ~harvest
        return f"{serving:x}", f"{harvest:x}"

    def schemata_line(self, resource, value):
        """모든 도메인에 같은 값: 'L3:0=ffff0;1=ffff0'."""
        doms = self.domains(resource)
        return resource + ":" + ";".join(f"{d}={value}" for d in doms)


# ---------- self-test (test.md G0) ----------

def selftest(rc):
    rc.require_mounted()
    g = "__rdt_selftest"
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("[selftest] info 판독")
    info = rc.info()
    check("info/L3 존재", "L3" in info, str(info.get("L3")))
    check("L3_MON mon_features", "L3_MON" in info, str(info.get("L3_MON", {}).get("mon_features")))

    print("[selftest] 도메인")
    doms = rc.domains("L3")
    check("dual-socket 도메인 2개", len(doms) == 2, str(doms))

    print("[selftest] CLOS 생성→schemata 기록→재판독")
    rc.create(g)
    try:
        serving_mask, _ = rc.make_masks(4)
        line = rc.schemata_line("L3", serving_mask)
        rc.set_schemata(g, [line])
        back = rc.get_schemata(g)
        # 커널은 leading zero 등 정규화할 수 있으므로 int 비교
        got = {}
        for ln in back.splitlines():
            ln = ln.strip()
            if ln.startswith("L3:"):
                for tok in ln.split(":", 1)[1].split(";"):
                    d, v = tok.split("=")
                    got[d] = int(v, 16)
        check("schemata 재판독 일치",
              all(got.get(d) == int(serving_mask, 16) for d in doms), str(got))

        print("[selftest] TID 등록→tasks 재판독")
        import threading
        tid = threading.get_native_id()
        rc.assign_tasks(g, [tid])
        check("tasks 포함", tid in rc.get_tasks(g), f"tid={tid}")

        print("[selftest] mon_data 단조성 (mbm 누적)")
        m1 = rc.read_mon(g)
        time.sleep(1.0)
        m2 = rc.read_mon(g)
        mono = all(
            m2[d].get(k, 0) >= m1[d].get(k, 0)
            for d in m1 for k in m1[d] if k.startswith("mbm_")
        ) if m1 else False
        check("mbm delta >= 0", mono, f"domains={list(m1)}")
    finally:
        # 자기 TID 를 루트로 환원 후 그룹 삭제
        try:
            import threading
            rc.assign_tasks("root", [threading.get_native_id()])
        except OSError:
            pass
        rc.delete(g)

    print(f"[selftest] {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 0 if not fails else 1


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("RESCTRL_ROOT", "/sys/fs/resctrl"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info")
    sub.add_parser("selftest")

    p = sub.add_parser("setup")
    p.add_argument("--group", required=True)
    p.add_argument("--l3-ways", help="'top'(상위 나머지) | 'low4'(하위 4) | hex mask 직접")
    p.add_argument("--harvest-ways", type=int, default=4)
    p.add_argument("--mb", type=int, help="MBA % (예: 100, 20)")

    p = sub.add_parser("assign")
    p.add_argument("--group", required=True)
    p.add_argument("--cpus", help="cpus_list 형식 (예: 16-55)")
    p.add_argument("--tids", help="콤마 구분 TID")

    p = sub.add_parser("mon")
    p.add_argument("--groups", required=True, help="콤마 구분 (root 포함 가능)")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--csv")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("teardown")
    p.add_argument("--group", action="append", required=True)

    args = ap.parse_args()
    rc = Resctrl(args.root)

    if args.cmd == "info":
        rc.require_mounted()
        for res, d in rc.info().items():
            print(f"[{res}]")
            for k, v in d.items():
                print(f"  {k} = {v}")
        print(f"[domains] L3={rc.domains('L3')} MB={rc.domains('MB')}")
        print(f"[cbm_width] {rc.cbm_width()} ways")
        return 0

    if args.cmd == "selftest":
        return selftest(rc)

    if args.cmd == "setup":
        rc.require_mounted()
        rc.create(args.group)
        lines = []
        if args.l3_ways:
            serving_mask, harvest_mask = rc.make_masks(args.harvest_ways)
            mask = {"top": serving_mask, "low4": harvest_mask}.get(args.l3_ways, args.l3_ways)
            lines.append(rc.schemata_line("L3", mask))
        if args.mb is not None:
            mb_min = int(rc._try_read("info", "MB", "min_bandwidth") or 10)
            lines.append(rc.schemata_line("MB", max(args.mb, mb_min)))
        if lines:
            rc.set_schemata(args.group, lines)
        print(f"[setup] {args.group}:\n{rc.get_schemata(args.group)}")
        return 0

    if args.cmd == "assign":
        rc.require_mounted()
        if args.cpus:
            rc.assign_cpus(args.group, args.cpus)
        if args.tids:
            rc.assign_tasks(args.group, [int(t) for t in args.tids.split(",")])
        print(f"[assign] {args.group} cpus={args.cpus} tids={args.tids}")
        return 0

    if args.cmd == "mon":
        rc.require_mounted()
        n = rc.monitor(args.groups.split(","), args.interval, args.duration,
                       csv_path=args.csv, quiet=args.quiet)
        print(f"[mon] {n} rows" + (f" → {args.csv}" if args.csv else ""))
        return 0

    if args.cmd == "teardown":
        for g in args.group:
            rc.delete(g)
        print(f"[teardown] {args.group}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
