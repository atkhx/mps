#import "krn_mtl_buffer_relu_bwd.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferReluBwdImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _reluFunction;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        NSError *error = nil;
        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _reluFunction = [self.library newFunctionWithName:@"reluBwd"];
        if (!_reluFunction) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function: %s!\n", errorCString);
        }
    }
    return self;
}

- (void)reluBwd:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        maskBuffer:(id<MTLBuffer>)maskBuffer
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    NSError *error = nil;
    [computeEncoder setComputePipelineState:[_device newComputePipelineStateWithFunction:_reluFunction error:&error]];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to setComputePipelineState: %s\n", errorCString);
    }
    [computeEncoder setBuffer:destinationBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:sourceBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:maskBuffer offset:0 atIndex:2];

    [computeEncoder dispatchThreads:MTLSizeMake(destinationBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end