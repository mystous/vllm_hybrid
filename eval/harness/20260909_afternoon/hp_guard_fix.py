P="/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"; s=open(P).read()
if s.index("kt_hp_alloc(size_t") > s.index("#ifndef CPUINFER_OPERATOR_AMX_MOE_BASE_H"):
    print("already inside guard"); raise SystemExit
a=s.index("#include <sys/mman.h>"); b=s.index("  return p;\n}\n",a)+len("  return p;\n}\n")
helper=s[a:b]; s=s[:a]+s[b:]
g="#define CPUINFER_OPERATOR_AMX_MOE_BASE_H\n"; j=s.index(g)+len(g)
s=s[:j]+helper+s[j:]; open(P,"w").write(s)
print("moved: guard", s.index("#ifndef CPUINFER"), "helper", s.index("kt_hp_alloc(size_t"))
