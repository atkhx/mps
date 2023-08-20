#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void* createCommandBuffer(void *commandQueueID);
void releaseCommandBuffer(void *commandBufferID);
void commitAndWaitUntilCompletedCommandBuffer(void *commandBufferID);

void clearMTLBuffer(
    void *deviceID,
    void *commandBufferID,
    void *bufferID,
    float value
);