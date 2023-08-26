#import "krn_mtl_buffer_fill.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferFillImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunction;
}

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        NSError *error = nil;
        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _kernelFunction = [self.library newFunctionWithName:@"fill"];

        if (!_kernelFunction) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function: %s!\n", errorCString);
        }
    }
    return self;
}

- (void) fill:(id<MTLBuffer>)buffer withValue:(float)value commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSError *error = nil;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunction error:&error]];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to setComputePipelineState: %s\n", errorCString);
    }
    [computeEncoder setBuffer:buffer offset:0 atIndex:0];
    [computeEncoder setBytes:&value length:sizeof(float) atIndex:1];

    [computeEncoder dispatchThreads:MTLSizeMake(buffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) fillPart:(id<MTLBuffer>)buffer withValue:(float)value commandBuffer:(id<MTLCommandBuffer>)commandBuffer offset:(uint)offset length:(uint)length {
    NSError *error = nil;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunction error:&error]];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to setComputePipelineState: %s\n", errorCString);
    }
    [computeEncoder setBuffer:buffer offset:offset atIndex:0];
    [computeEncoder setBytes:&value length:sizeof(float) atIndex:1];

    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end