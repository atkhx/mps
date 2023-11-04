#include <metal_stdlib>

using namespace metal;

kernel void dropout(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *mskBuffer [[ buffer(2) ]],
    constant float& probability [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (mskBuffer[id] > probability) {
        dstBuffer[id] = srcBuffer[id];
    } else {
        dstBuffer[id] = 0.0;
    }
}

kernel void dropoutBwd(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *mskBuffer [[ buffer(2) ]],
    constant float& probability [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (mskBuffer[id] > probability) {
        dstBuffer[id] += srcBuffer[id];
    }
}
