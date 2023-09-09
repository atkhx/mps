#include <metal_stdlib>

using namespace metal;

struct Parameters {
    uint width;
};

kernel void softmaxTrilBwd(
    device float *oGrad [[ buffer(0) ]],
    device float *iGrad [[ buffer(1) ]],
    device float *softmax [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    // constant struct Parameters &params [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    // uint startIdx = gid.y * params.width;
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float g = 0;
    float s = 0;

    for (uint i = startIdx; i < endIdx; ++i) {
        g = softmax[i] * oGrad[i];
        s += g;
        iGrad[i] += g;
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        iGrad[i] -= softmax[i] * s;
    }
}




kernel void mul1(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    device float *softmaxBuffer [[ buffer(2) ]],
    device float *softmaxGradBuffer [[ buffer(3) ]],
    constant struct Parameters &params [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    if (gid.x <= gid.y) {
        uint startIdx = gid.y * params.width+gid.x;
        float softmaxIG = softmaxBuffer[startIdx] * sourceBuffer[startIdx];

        destinationBuffer[startIdx] += softmaxIG;
        softmaxGradBuffer[startIdx] = softmaxIG;
    }
}


kernel void sum1(
    device float *softmaxGradBuffer [[ buffer(0) ]],
    device float *sumOutBuffer [[ buffer(1) ]],
    constant struct Parameters &params [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * params.width;
    float sumG = 0.0;
    for (uint i = startIdx; i < startIdx + gid.y+1; ++i) {
        sumG += softmaxGradBuffer[i];
    }

    sumOutBuffer[gid.y] = sumG;
}

kernel void sub1(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *softmaxBuffer [[ buffer(1) ]],
    device float *sumOutBuffer [[ buffer(2) ]],
    constant struct Parameters &params [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    if (gid.x <= gid.y) {
        uint startIdx = gid.y * params.width+gid.x;
        destinationBuffer[startIdx] -= softmaxBuffer[startIdx]*sumOutBuffer[gid.y];
    }
}
