#include "mtl_command_buffer.h"

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


void fillMTLBuffer(
    void *kernelID,
    void *commandBufferID,
    void *bufferID,
    float value
) {
    KernelMTLBufferFillImpl *kernel = (__bridge KernelMTLBufferFillImpl*)kernelID;
    [kernel fill:(id<MTLBuffer>)bufferID withValue:value
        commandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

void fillPartMTLBuffer(
    void *kernelID,
    void *commandBufferID,
    void *bufferID,
    const uint offset,
    const uint length,
    float value
) {
    KernelMTLBufferFillImpl *kernel = (__bridge KernelMTLBufferFillImpl*)kernelID;
    [kernel fillPart:(id<MTLBuffer>)bufferID withValue:value
        commandBuffer:(id<MTLCommandBuffer>)commandBufferID offset:offset length:length];
}

void reluMTLBuffer(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID
) {
    KernelMTLBufferReluFwdImpl *kernel = (__bridge KernelMTLBufferReluFwdImpl*)kernelID;
    [kernel reluFwd:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

void reluMTLBufferBwd(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskBufferID
) {
    KernelMTLBufferReluBwdImpl *kernel = (__bridge KernelMTLBufferReluBwdImpl*)kernelID;

    [kernel reluBwd:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        maskBuffer:(id<MTLBuffer>)maskBufferID
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}


void mulBuffer(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *multiplierBufferID
) {
    KernelMTLBufferMulImpl *kernel = (__bridge KernelMTLBufferMulImpl*)kernelID;
    [kernel mul:(id<MTLBuffer>)destinationBufferID
        multiplierBuffer:(id<MTLBuffer>)multiplierBufferID
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

void dropoutBuffer(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskOutBufferID,
    float probability
) {
    KernelMTLBufferDropoutImpl *kernel = (__bridge KernelMTLBufferDropoutImpl*)kernelID;
    [kernel
        dropout:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        maskOutBuffer:(id<MTLBuffer>)maskOutBufferID
        probability:probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

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
) {
    KernelMTLBufferSoftmaxImpl *kernel = [[KernelMTLBufferSoftmaxImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    [kernel
        softmax:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        sumOutBuffer:(id<MTLBuffer>)sumOutBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}


void softmaxBufferTril(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
//     void *maxOutBufferID,
//     void *sumOutBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    KernelMTLBufferSoftmaxTrilImpl *kernel = (__bridge KernelMTLBufferSoftmaxTrilImpl*)kernelID;

    [kernel
        softmaxTril:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
//         maxOutBuffer:(id<MTLBuffer>)maxOutBufferID
//         sumOutBuffer:(id<MTLBuffer>)sumOutBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

void softmaxBufferTrilBwd(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *softmaxBufferID,
//     void *softmaxGradBufferID,
//     void *sumOutBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    KernelMTLBufferSoftmaxTrilBwdImpl *kernel = (__bridge KernelMTLBufferSoftmaxTrilBwdImpl*)kernelID;

    [kernel
        softmaxTrilBwd:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        softmaxBuffer:(id<MTLBuffer>)softmaxBufferID
//         softmaxGradBuffer:(id<MTLBuffer>)softmaxGradBufferID
//         sumOutBuffer:(id<MTLBuffer>)sumOutBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}
