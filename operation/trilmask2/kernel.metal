#include <metal_stdlib>

using namespace metal;

kernel void trilMask2(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant float& mask [[ buffer(2) ]],
    constant uint& colsCount [[ buffer(3) ]],
    constant uint& rowsCount [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i = gid.z*colsCount*rowsCount + gid.y*colsCount + gid.x;
    if (gid.x > gid.y) {
        outputData[i] = mask;
    } else {
        outputData[i] = inputData[i];
    }
}

kernel void trilMask2Bwd(
    device float *inputGrad [[ buffer(0) ]],
    constant uint& colsCount [[ buffer(1) ]],
    constant uint& rowsCount [[ buffer(2) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    if (gid.x > gid.y) {
        inputGrad[gid.z*colsCount*rowsCount + gid.y*colsCount + gid.x] = 0;
    }
}
