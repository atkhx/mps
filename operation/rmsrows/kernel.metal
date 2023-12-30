#include <metal_stdlib>

using namespace metal;

kernel void rmsByRows(
    device float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    float val = 0.0;
    for (uint i = gid.y*width; i < (gid.y+1)*width; ++i) {
        val += input[i] * input[i];
    }
    output[gid.y] = sqrt(1e-5 + (val / float(width)));
}

kernel void rmsByRowsGrads(
    device float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    device float *inputGrads [[ buffer(2) ]],
    device float *outputGrads [[ buffer(3) ]],
    constant uint& width [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    inputGrads[gid.y*width + gid.x] += outputGrads[gid.y] * (1/(2*output[gid.y])) * (2 * input[gid.y*width + gid.x] /float(width)) ;
}
