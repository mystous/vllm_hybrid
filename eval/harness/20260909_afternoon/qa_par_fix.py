P="/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"; s=open(P).read()
old="    if (kt_qa_par_() && qlen >= 10) {"
assert s.count(old)==2, s.count(old)
new="    if constexpr (requires(typename T::BufferA& x_, ggml_bf16_t* p_) { x_.from_mat_block(1, p_, 0); }) if (kt_qa_par_() && qlen >= 10) {"
s=s.replace(old,new); open(P,"w").write(s); print("guard fix applied")
