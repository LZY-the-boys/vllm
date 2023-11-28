#include "bgmv_config.h"
#include "bgmv_impl.cuh"

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, nv_half, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, nv_half, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_bfloat16, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_bfloat16, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, nv_bfloat16, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_half, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, nv_bfloat16, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_half, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_half, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_half, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_bfloat16, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_bfloat16, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, float, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, float, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, float, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, float, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, float, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, float, nv_bfloat16)