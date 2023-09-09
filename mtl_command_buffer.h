#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>
#include "mtl_custom_kernels.h"

void* createCommandBuffer(void *commandQueueID);
void releaseCommandBuffer(void *commandBufferID);
void commitAndWaitUntilCompletedCommandBuffer(void *commandBufferID);

void fillMTLBuffer(
    void *kernelID,
    void *commandBufferID,
    void *bufferID,
    float value
);

void fillPartMTLBuffer(
    void *kernelID,
    void *commandBufferID,
    void *bufferID,
    const uint offset,
    const uint length,
    float value
);

void reluMTLBuffer(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID
);

void reluMTLBufferBwd(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskBufferID
);

void mulBuffer(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *multiplierBufferID
);

void dropoutBuffer(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskOutBufferID,
    float probability
);

void softmaxBuffer(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *sumOutBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset,
    const char *kernelSource
);

void softmaxBufferTril(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
//    void *maxOutBufferID,
//    void *sumOutBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
);

void softmaxBufferTrilBwd(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *softmaxBufferID,
//    void *softmaxGradBufferID,
//    void *sumOutBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
);

