#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void* createCommandBuffer(void *commandQueueID);
void releaseCommandBuffer(void *commandBufferID);
void commitAndWaitUntilCompletedCommandBuffer(void *commandBufferID);

void fillMTLBuffer(
    const char *kernelSource,
    void *deviceID,
    void *commandBufferID,
    void *bufferID,
    float value
);

void fillPartMTLBuffer(
    const char *kernelSource,
    void *deviceID,
    void *commandBufferID,
    void *bufferID,
    const uint offset,
    const uint length,
    float value
);

void reluMTLBuffer(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    const char *kernelSource
);

void reluMTLBufferBwd(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskBufferID,
    const char *kernelSource
);

void mulBuffer(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *multiplierBufferID,
    const char *kernelSource
);

void dropoutBuffer(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskOutBufferID,
    float probability,
    const char *kernelSource
);