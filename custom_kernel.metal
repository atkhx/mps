#include <metal_stdlib>

using namespace metal;

struct Parameters {
    uint width;
};

kernel void copy(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = srcBuffer[id];
}

kernel void fill(
    device float *dstBuffer [[ buffer(0) ]],
    const uint id [[ thread_position_in_grid ]],
    constant float& value [[ buffer(1) ]])
{
    dstBuffer[id] = value;
}

kernel void add(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] += srcBuffer[id];
}

kernel void addTo(
    device float *dstBuffer [[ buffer(0) ]],
    device float *aBuffer [[ buffer(1) ]],
    device float *bBuffer [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = aBuffer[id] + bBuffer[id];
}

kernel void mul(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] *= srcBuffer[id];
}

kernel void divOnSum(
    device float *dstBuffer [[ buffer(0) ]],
    device float *sumBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    dstBuffer[gid.y * width+gid.x] /= sumBuffer[gid.y];
}

kernel void exp(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    destinationBuffer[id] = exp(sourceBuffer[id]);
}

kernel void sum(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx + width;

    float sumValue = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        sumValue += srcBuffer[i];
    }

    dstBuffer[gid.y] = sumValue;
}

kernel void relu(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (srcBuffer[id] < 0) {
        dstBuffer[id] = 0;
    } else {
        dstBuffer[id] = srcBuffer[id];
    }
}

kernel void reluBwd(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *mskBuffer [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (mskBuffer[id] > 0) {
        dstBuffer[id] += srcBuffer[id];
    }
}

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

kernel void updateWithAdam(
    device float *dataBuffer [[ buffer(0) ]],
    device float *gradBuffer [[ buffer(1) ]],
    device float *mBuffer [[ buffer(2) ]],
    device float *vBuffer [[ buffer(3) ]],
    constant float& beta1 [[ buffer(4) ]],
    constant float& beta2 [[ buffer(5) ]],
    constant float& beta1powIterationLR [[ buffer(6) ]],
    constant float& beta2powIteration [[ buffer(7) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    mBuffer[id] = beta1*mBuffer[id] + (1 - beta1)*gradBuffer[id];
    vBuffer[id] = beta2*vBuffer[id] + (1 - beta2)*gradBuffer[id]*gradBuffer[id];

    dataBuffer[id] -= mBuffer[id] * beta1powIterationLR / (sqrt(vBuffer[id] * beta2powIteration) + 0.000000001);
}

kernel void softmaxTril(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float max = srcBuffer[startIdx];
    for (uint i = startIdx+1; i < endIdx; ++i) {
        if (max < srcBuffer[i]) {
            max = srcBuffer[i];
        }
    }

    float sumExp = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] = exp(srcBuffer[i]-max);
        sumExp += dstBuffer[i];
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] /= sumExp;
    }
}

kernel void softmaxBufferTrilBwd(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *smxBuffer [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float g = 0;
    float s = 0;

    for (uint i = startIdx; i < endIdx; ++i) {
        g = smxBuffer[i] * srcBuffer[i];
        s += g;
        dstBuffer[i] += g;
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] -= smxBuffer[i] * s;
    }
}
