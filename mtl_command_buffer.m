#include "mtl_command_buffer.h"
#import "krn_mtl_buffer_fill.h"
#import "krn_mtl_buffer_relu_fwd.h"
#import "krn_mtl_buffer_relu_bwd.h"
#import "krn_mtl_buffer_mul.h"
#import "krn_mtl_buffer_dropout.h"

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
    const char *kernelSource,
    void *deviceID,
    void *commandBufferID,
    void *bufferID,
    float value
) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> buffer = (id<MTLBuffer>)bufferID;
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;

    NSString *kernelSourceString = [NSString stringWithUTF8String:kernelSource];

    KernelMTLBufferFillImpl *kernel = [[KernelMTLBufferFillImpl alloc] initWithDevice:device kernelSource:kernelSourceString];
    [kernel fill:buffer withValue:value commandBuffer:commandBuffer];
}

void fillPartMTLBuffer(
    const char *kernelSource,
    void *deviceID,
    void *commandBufferID,
    void *bufferID,
    const uint offset,
    const uint length,
    float value
) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> buffer = (id<MTLBuffer>)bufferID;
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;

    NSString *kernelSourceString = [NSString stringWithUTF8String:kernelSource];

    KernelMTLBufferFillImpl *kernel = [[KernelMTLBufferFillImpl alloc] initWithDevice:device kernelSource:kernelSourceString];
    [kernel fillPart:buffer withValue:value commandBuffer:commandBuffer offset:offset length:length];
}

void reluMTLBuffer(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    const char *kernelSource
) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> destinationBuffer = (id<MTLBuffer>)destinationBufferID;
    id<MTLBuffer> sourceBuffer = (id<MTLBuffer>)sourceBufferID;
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;

    NSString *kernelSourceString = [NSString stringWithUTF8String:kernelSource];

    KernelMTLBufferReluFwdImpl *kernel = [[KernelMTLBufferReluFwdImpl alloc] initWithDevice:device kernelSource:kernelSourceString];
    [kernel reluFwd:destinationBuffer sourceBuffer:sourceBuffer withCommandBuffer:commandBuffer];
}

void reluMTLBufferBwd(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskBufferID,
    const char *kernelSource
) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> destinationBuffer = (id<MTLBuffer>)destinationBufferID;
    id<MTLBuffer> sourceBuffer = (id<MTLBuffer>)sourceBufferID;
    id<MTLBuffer> maskBuffer = (id<MTLBuffer>)maskBufferID;
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;

    NSString *kernelSourceString = [NSString stringWithUTF8String:kernelSource];

    KernelMTLBufferReluBwdImpl *kernel = [[KernelMTLBufferReluBwdImpl alloc] initWithDevice:device kernelSource:kernelSourceString];
    [kernel reluBwd:destinationBuffer sourceBuffer:sourceBuffer maskBuffer:maskBuffer withCommandBuffer:commandBuffer];
}


void mulBuffer(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *multiplierBufferID,
    const char *kernelSource
) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> destinationBuffer = (id<MTLBuffer>)destinationBufferID;
    id<MTLBuffer> multiplierBuffer = (id<MTLBuffer>)multiplierBufferID;
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;

    NSString *kernelSourceString = [NSString stringWithUTF8String:kernelSource];

    KernelMTLBufferMulImpl *kernel = [[KernelMTLBufferMulImpl alloc] initWithDevice:device kernelSource:kernelSourceString];
    [kernel mul:destinationBuffer multiplierBuffer:multiplierBuffer withCommandBuffer:commandBuffer];
}


void dropoutBuffer(
    void *deviceID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskOutBufferID,
    float probability,
    const char *kernelSource
) {
    KernelMTLBufferDropoutImpl *kernel = [[KernelMTLBufferDropoutImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    [kernel
        dropout:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        maskOutBuffer:(id<MTLBuffer>)maskOutBufferID
        probability:probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}