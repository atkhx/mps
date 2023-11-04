#import "krn_mtl_buffer_dropout.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferDropoutImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _dropoutFunction;
    id<MTLFunction> _dropoutBwdFunction;
    id<MTLComputePipelineState> _dropoutFunctionPSO;
    id<MTLComputePipelineState> _dropoutBwdFunctionPSO;
    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _dropoutFunction = [self.library newFunctionWithName:@"dropout"];
        if (!_dropoutFunction) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function dropout: %s!\n", errorCString);
        }

        _dropoutFunctionPSO = [_device newComputePipelineStateWithFunction:_dropoutFunction error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }

        _dropoutBwdFunction = [self.library newFunctionWithName:@"dropoutBwd"];
        if (!_dropoutBwdFunction) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function dropoutBwd: %s!\n", errorCString);
        }

        _dropoutBwdFunctionPSO = [_device newComputePipelineStateWithFunction:_dropoutBwdFunction error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }
    }
    return self;
}

- (void) dropout:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer
        probability:(float)probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_dropoutFunctionPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:mskBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&probability length:sizeof(float) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}


- (void) dropoutBwd:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer
        probability:(float)probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_dropoutBwdFunctionPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:mskBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&probability length:sizeof(float) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end