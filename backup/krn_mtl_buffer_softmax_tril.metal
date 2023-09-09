#include <metal_stdlib>

using namespace metal;

struct Parameters {
    uint width;
};

kernel void softmaxTril(
    device float *inputBuffer [[ buffer(0) ]],
    device float *outputBuffer [[ buffer(1) ]],
    // constant struct Parameters &params [[ buffer(2) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    // uint startIdx = gid.y * params.width;
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float max = inputBuffer[startIdx];
    for (uint i = startIdx+1; i < endIdx; ++i) {
        if (max < inputBuffer[i]) {
            max = inputBuffer[i];
        }
    }

    float sumExp = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        outputBuffer[i] = exp(inputBuffer[i]-max);
        sumExp += outputBuffer[i];
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        outputBuffer[i] /= sumExp;
    }
}



kernel void max(
    device float *sourceBuffer [[ buffer(0) ]],
    device float *maxOutBuffer [[ buffer(1) ]],
    constant struct Parameters &params [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * params.width;
    float maxVal = sourceBuffer[startIdx];
    for (uint i = startIdx+1; i < startIdx + gid.y+1; ++i) {
        if (maxVal < sourceBuffer[i]) {
            maxVal = sourceBuffer[i];
        }
    }

    maxOutBuffer[gid.y] = maxVal;
}

kernel void exp(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    device float *maxOutBuffer [[ buffer(2) ]],
    constant struct Parameters &params [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    if (gid.x <= gid.y) {
        uint startIdx = gid.y * params.width+gid.x;
        destinationBuffer[startIdx] = exp(sourceBuffer[startIdx]-maxOutBuffer[gid.y]);
    }
}

kernel void sum(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sumOutBuffer [[ buffer(1) ]],
    constant struct Parameters &params [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * params.width;
    float sumExp = 0.0;
    for (uint i = startIdx; i < startIdx + gid.y+1; ++i) {
        sumExp += destinationBuffer[i];
    }

    sumOutBuffer[gid.y] = sumExp;
}

kernel void div(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sumOutBuffer [[ buffer(1) ]],
    constant struct Parameters &params [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * params.width+gid.x;
    destinationBuffer[startIdx] /= sumOutBuffer[gid.y];
}
