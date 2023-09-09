#import "krn_mtl_buffer_softmax_tril.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferSoftmaxTrilImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunctionSoftmaxTril;
    id<MTLFunction> _kernelFunctionMax;
    id<MTLFunction> _kernelFunctionExp;
    id<MTLFunction> _kernelFunctionSum;
    id<MTLFunction> _kernelFunctionDiv;
    id<MTLComputePipelineState> _mFunctionPSO;
    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _kernelFunctionSoftmaxTril = [self.library newFunctionWithName:@"softmaxTril"];
        if (!_kernelFunctionSoftmaxTril) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function softmaxTril: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_kernelFunctionSoftmaxTril error:&error];

        _kernelFunctionMax = [self.library newFunctionWithName:@"max"];
        if (!_kernelFunctionMax) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function max: %s!\n", errorCString);
        }

        _kernelFunctionExp = [self.library newFunctionWithName:@"exp"];
        if (!_kernelFunctionExp) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function exp: %s!\n", errorCString);
        }

        _kernelFunctionSum = [self.library newFunctionWithName:@"sum"];
        if (!_kernelFunctionSum) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function sum: %s!\n", errorCString);
        }

        _kernelFunctionDiv = [self.library newFunctionWithName:@"div"];
        if (!_kernelFunctionDiv) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function div: %s!\n", errorCString);
        }
    }
    return self;
}

struct Parameters {
    uint width;
};

- (void) softmaxTril:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
//         maxOutBuffer:(id<MTLBuffer>)maxOutBuffer
//         sumOutBuffer:(id<MTLBuffer>)sumOutBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {


//     struct Parameters params;
//     params.width = colsCount;


    id<MTLComputeCommandEncoder> computeEncoderSoftmaxTril = [commandBuffer computeCommandEncoder];
    [computeEncoderSoftmaxTril setComputePipelineState:_mFunctionPSO];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to setComputePipelineState softmaxTril: %s\n", errorCString);
    }

    [computeEncoderSoftmaxTril setBuffer:sourceBuffer offset:offset atIndex:0];
    [computeEncoderSoftmaxTril setBuffer:destinationBuffer offset:offset atIndex:1];
//     [computeEncoderSoftmaxTril setBytes:&params length:sizeof(struct Parameters) atIndex:2];
    [computeEncoderSoftmaxTril setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoderSoftmaxTril dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoderSoftmaxTril endEncoding];




//
//     id<MTLComputeCommandEncoder> computeEncoderMax = [commandBuffer computeCommandEncoder];
//     [computeEncoderMax setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionMax error:&error]];
//     if (error != nil) {
//         const char *errorCString = [[error localizedDescription] UTF8String];
//         printf("Failed to setComputePipelineState exp: %s\n", errorCString);
//     }
//
//     [computeEncoderMax setBuffer:sourceBuffer offset:offset atIndex:0];
//     [computeEncoderMax setBuffer:maxOutBuffer offset:0 atIndex:1];
//     [computeEncoderMax setBytes:&params length:sizeof(struct Parameters) atIndex:2];
//     [computeEncoderMax dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
//     [computeEncoderMax endEncoding];
//
//
//
//
//
//     id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
//     [computeEncoder setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionExp error:&error]];
//     if (error != nil) {
//         const char *errorCString = [[error localizedDescription] UTF8String];
//         printf("Failed to setComputePipelineState exp: %s\n", errorCString);
//     }
//
//     [computeEncoder setBuffer:destinationBuffer offset:offset atIndex:0];
//     [computeEncoder setBuffer:sourceBuffer offset:offset atIndex:1];
//     [computeEncoder setBuffer:maxOutBuffer offset:0 atIndex:2];
//     [computeEncoder setBytes:&params length:sizeof(struct Parameters) atIndex:3];
//     [computeEncoder dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
//     [computeEncoder endEncoding];
//
//
//
//
//
//     id<MTLComputeCommandEncoder> computeEncoderSum = [commandBuffer computeCommandEncoder];
//     [computeEncoderSum setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionSum error:&error]];
//     if (error != nil) {
//         const char *errorCString = [[error localizedDescription] UTF8String];
//         printf("Failed to setComputePipelineState exp: %s\n", errorCString);
//     }
//
//     [computeEncoderSum setBuffer:destinationBuffer offset:offset atIndex:0];
//     [computeEncoderSum setBuffer:sumOutBuffer offset:0 atIndex:1];
//     [computeEncoderSum setBytes:&params length:sizeof(struct Parameters) atIndex:2];
//     [computeEncoderSum dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
//     [computeEncoderSum endEncoding];
//
//
//     id<MTLComputeCommandEncoder> computeEncoderDiv = [commandBuffer computeCommandEncoder];
//     [computeEncoderDiv setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionDiv error:&error]];
//     if (error != nil) {
//         const char *errorCString = [[error localizedDescription] UTF8String];
//         printf("Failed to setComputePipelineState exp: %s\n", errorCString);
//     }
//
//     [computeEncoderDiv setBuffer:destinationBuffer offset:offset atIndex:0];
//     [computeEncoderDiv setBuffer:sumOutBuffer offset:0 atIndex:1];
//     [computeEncoderDiv setBytes:&params length:sizeof(struct Parameters) atIndex:2];
//     [computeEncoderDiv dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
//     [computeEncoderDiv endEncoding];
}

@end