#import "krn_fill_mtl_buffer.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation ClearBufferImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _clearFunction;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;

        NSString *kernelSource = @""
            @"#include <metal_stdlib> \n"
            @"using namespace metal; \n"
            @"kernel void clearBuffer("
                @"device float *buffer [[ buffer(0) ]], "
                @"const uint id [[ thread_position_in_grid ]], "
                @"constant float& value [[ buffer(1) ]]) "
            @"{ "
                @"buffer[id] = value; "
            @"}";

        NSError *error = nil;
        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _clearFunction = [self.library newFunctionWithName:@"clearBuffer"];
        if (!_clearFunction) {
            printf("Failed to load function!\n");
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Error description: %s\n", errorCString);
        }
    }
    return self;
}

- (void)clearBuffer:(id<MTLBuffer>)buffer withValue:(float)value commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    NSError *error = nil;
    [computeEncoder setComputePipelineState:[_device newComputePipelineStateWithFunction:_clearFunction error:&error]];
    if (error != nil) {
            printf("Failed to setComputePipelineState!\n");
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Error description: %s\n", errorCString);
    }
    [computeEncoder setBuffer:buffer offset:0 atIndex:0];
    [computeEncoder setBytes:&value length:sizeof(float) atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(buffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end