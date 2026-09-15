#!/usr/bin/env bash
# sgl-kt 의 소프트웨어 상태(kt_kernel 0.7.0.post1 소스빌드 + sglang 로컬 수정 10파일 + deep_gemm 비활성)를
# cpuset 컨테이너(sgl-kt5, sgl-kt2)에 복제한다. nerdctl commit 이 실패해 파일 단위로 옮긴다 (/models 마운트 경유).
# 세 컨테이너 모두 같은 이미지·같은 sglang commit(71de97b)·같은 torch/flashinfer/triton 이므로 이 세 항목만 다르다.
set -euo pipefail
export PATH=$HOME/bin:$PATH
SYNC=/models/kt/ide069/sync; HOST_SYNC=$HOME/.cache/huggingface/kt/ide069/sync
mkdir -p "$HOST_SYNC"; chmod 777 "$HOST_SYNC"
DP=/usr/local/lib/python3.12/dist-packages
docker exec sgl-kt bash -c "
set -e; cd /sgl-workspace/sglang && git diff --name-only > $SYNC/sglang_files.txt
tar -C $DP -cf $SYNC/kt_kernel.tar kt_kernel kt_kernel-0.7.0.post1.dist-info
tar -C /sgl-workspace/sglang -cf $SYNC/sglang_mod.tar -T $SYNC/sglang_files.txt
ls -la $SYNC"
for c in "$@"; do
  echo "== sync → $c"
  docker exec $c bash -c "
set -e
cd $DP && rm -rf kt_kernel kt_kernel-0.7.0.post2.dist-info kt_kernel-0.7.0.post1.dist-info
tar -C $DP -xf $SYNC/kt_kernel.tar
cd /sgl-workspace/sglang && git checkout -- . && tar -xf $SYNC/sglang_mod.tar
[ -d $DP/deep_gemm ] && mv $DP/deep_gemm $DP/deep_gemm.disabled || true
pip show kt-kernel | grep ^Version; python3 -c 'import kt_kernel, sglang; print(\"import ok\")'
git diff --name-only | wc -l; ls -d $DP/deep_gemm* "
done
echo "sync done"
