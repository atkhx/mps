#include "mtl_command_buffer.h"
#import "krn_fill_mtl_buffer.h"

void* createCommandBuffer(void *commandQueueID) {
    id<MTLCommandQueue> commandQueue = (id<MTLCommandQueue>)commandQueueID;
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    return (__bridge void*)commandBuffer;
}

void releaseCommandBuffer(void *commandBufferID) {
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;
    [commandBuffer release];
}

void commitAndWaitUntilCompletedCommandBuffer(void *commandBufferID) {
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void clearMTLBuffer(
    void *deviceID,
    void *commandBufferID,
    void *bufferID,
    float value
) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> buffer = (id<MTLBuffer>)bufferID;
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;

    ClearBufferImpl *clearBufferImpl = [[ClearBufferImpl alloc] initWithDevice:device];
    [clearBufferImpl clearBuffer:buffer withValue:value commandBuffer:commandBuffer];
}