#include <metal_stdlib>

using namespace metal;

kernel void dropout(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *mskBuffer [[ buffer(2) ]],
    constant float& probability [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    // todo change to random filled matrix
    float randomValue = fract(sin(float(id)) * 43758.5453123);

    if (randomValue > probability) {
        dstBuffer[id] = srcBuffer[id];
        mskBuffer[id] = randomValue;
    } else {
        dstBuffer[id] = 0.0;
        mskBuffer[id] = randomValue;
    }
}
