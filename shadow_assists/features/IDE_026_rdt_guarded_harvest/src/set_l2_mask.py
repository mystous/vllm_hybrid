#!/usr/bin/env python3
"""harvest CLOS 의 L2 CBM 을 전 도메인에 일괄 기록. 사용: sudo python3 set_l2_mask.py <group> <hexmask>"""
import re, sys

group, mask = sys.argv[1], sys.argv[2]
root = "/sys/fs/resctrl"
l2_line = next(l for l in open(f"{root}/schemata") if l.strip().startswith("L2:"))
doms = re.findall(r"(\d+)=[0-9a-fA-F]+", l2_line)
line = "L2:" + ";".join(f"{d}={mask}" for d in doms)
with open(f"{root}/{group}/schemata", "w") as f:
    f.write(line + "\n")
# 재판독 검증
back = next(l for l in open(f"{root}/{group}/schemata") if l.strip().startswith("L2:"))
vals = {int(v, 16) for v in re.findall(r"=([0-9a-fA-F]+)", back)}
assert vals == {int(mask, 16)}, f"재판독 불일치: {vals}"
print(f"[set_l2_mask] {group} L2={mask} ({len(doms)} domains) OK")
