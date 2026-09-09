P="/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"; s=open(P).read()
old="      if (dump_on && qlen >= 10 && dump_ct < 4 && tp_part_idx == 0) {"
new="      if constexpr (requires(typename T::BufferA& x_) { x_.a; x_.d; }) if (dump_on && qlen >= 10 && dump_ct < 4 && tp_part_idx == 0) {"
assert s.count(old)==1; s=s.replace(old,new); open(P,"w").write(s); print("dump guard fix applied")
