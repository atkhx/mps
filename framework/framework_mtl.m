#include "framework_mtl.h"

// MTLDevice

void* mtlDeviceCreate() {
    return MTLCreateSystemDefaultDevice();
}

void mtlDeviceRelease(void *deviceID) {
    [(id<MTLDevice>)deviceID release];
}

// MTLCommandQueue

void* mtlCommandQueueCreate(void *deviceID) {
    return [(id<MTLDevice>)deviceID newCommandQueue];
}

void mtlCommandQueueRelease(void *commandQueueID) {
    [(id<MTLCommandQueue>)commandQueueID release];
}

// MTLCommandBuffer

void* mtlCommandBufferCreate(void *commandQueueID) {
    return [(id<MTLCommandQueue>)commandQueueID commandBuffer];
}

void mtlCommandBufferRelease(void *commandBufferID) {
    [(id<MTLCommandBuffer>)commandBufferID release];
}

void mtlCommandBufferCommitAndWaitUntilCompleted(void *commandBufferID) {
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// MTLBuffer

void* mtlBufferCreateCreateWithBytes(void *deviceID, float *bytes, size_t length) {
    return [(id<MTLDevice>)deviceID
        newBufferWithBytes:bytes
        length:length*sizeof(float)
        options:MTLResourceStorageModeShared];
}

void* mtlBufferCreateWithLength(void *deviceID, size_t length) {
    return [(id<MTLDevice>)deviceID
        newBufferWithLength:length * sizeof(float)
        options:MTLResourceStorageModeShared];
}

void* mtlBufferCreatePrivateWithLength(void *deviceID, size_t length) {
    return [(id<MTLDevice>)deviceID
        newBufferWithLength:length * sizeof(float)
        options:MTLResourceStorageModePrivate];
}

void* mtlBufferGetContents(void *bufferID) {
    return [(id<MTLBuffer>)bufferID contents];
}

void mtlBufferRelease(void *bufferID) {
    [(id<MTLBuffer>)bufferID release];
}
