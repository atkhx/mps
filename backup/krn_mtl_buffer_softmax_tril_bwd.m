#import "krn_mtl_buffer_softmax_tril_bwd.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferSoftmaxTrilBwdImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunctionSoftmaxTrilBwd;
    id<MTLFunction> _kernelFunctionMul1;
    id<MTLFunction> _kernelFunctionSum1;
    id<MTLFunction> _kernelFunctionSub1;
    id<MTLComputePipelineState> _mFunctionPSO;
    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _kernelFunctionSoftmaxTrilBwd = [self.library newFunctionWithName:@"softmaxTrilBwd"];
        if (!_kernelFunctionSoftmaxTrilBwd) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function softmaxTrilBwd: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_kernelFunctionSoftmaxTrilBwd error:&error];

        _kernelFunctionMul1 = [self.library newFunctionWithName:@"mul1"];
        if (!_kernelFunctionMul1) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function mul1: %s!\n", errorCString);
        }

        _kernelFunctionSum1 = [self.library newFunctionWithName:@"sum1"];
        if (!_kernelFunctionSum1) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function sum1: %s!\n", errorCString);
        }

        _kernelFunctionSub1 = [self.library newFunctionWithName:@"sub1"];
        if (!_kernelFunctionSub1) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function sub1: %s!\n", errorCString);
        }
    }
    return self;
}

struct Parameters {
    uint width;
};

- (void) softmaxTrilBwd:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        softmaxBuffer:(id<MTLBuffer>)softmaxBuffer
//         softmaxGradBuffer:(id<MTLBuffer>)softmaxGradBuffer
//         sumOutBuffer:(id<MTLBuffer>)sumOutBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

//     struct Parameters params;
//     params.width = colsCount;

    id<MTLComputeCommandEncoder> computeEncoderSoftmaxTrilBwd = [commandBuffer computeCommandEncoder];

    [computeEncoderSoftmaxTrilBwd setComputePipelineState:_mFunctionPSO];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to setComputePipelineState softmaxTrilBwd: %s\n", errorCString);
    }

    [computeEncoderSoftmaxTrilBwd setBuffer:sourceBuffer offset:offset atIndex:0];
    [computeEncoderSoftmaxTrilBwd setBuffer:destinationBuffer offset:offset atIndex:1];
    [computeEncoderSoftmaxTrilBwd setBuffer:softmaxBuffer offset:offset atIndex:2];
//     [computeEncoderSoftmaxTrilBwd setBytes:&params length:sizeof(struct Parameters) atIndex:3];
    [computeEncoderSoftmaxTrilBwd setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [computeEncoderSoftmaxTrilBwd dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoderSoftmaxTrilBwd endEncoding];

//
//
//     id<MTLComputeCommandEncoder> computeEncoderMul1 = [commandBuffer computeCommandEncoder];
//     [computeEncoderMul1 setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionMul1 error:&error]];
//     if (error != nil) {
//         const char *errorCString = [[error localizedDescription] UTF8String];
//         printf("Failed to setComputePipelineState mul1: %s\n", errorCString);
//     }
//
//     [computeEncoderMul1 setBuffer:destinationBuffer offset:offset atIndex:0];
//     [computeEncoderMul1 setBuffer:sourceBuffer offset:offset atIndex:1];
//     [computeEncoderMul1 setBuffer:softmaxBuffer offset:offset atIndex:2];
//     [computeEncoderMul1 setBuffer:softmaxGradBuffer offset:offset atIndex:3];
//     [computeEncoderMul1 setBytes:&params length:sizeof(struct Parameters) atIndex:4];
//     [computeEncoderMul1 dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
//     [computeEncoderMul1 endEncoding];
//
//
//
//
//     id<MTLComputeCommandEncoder> computeEncoderSum1 = [commandBuffer computeCommandEncoder];
//     [computeEncoderSum1 setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionSum1 error:&error]];
//     if (error != nil) {
//         const char *errorCString = [[error localizedDescription] UTF8String];
//         printf("Failed to setComputePipelineState sum1: %s\n", errorCString);
//     }
//
//     [computeEncoderSum1 setBuffer:softmaxGradBuffer offset:offset atIndex:0];
//     [computeEncoderSum1 setBuffer:sumOutBuffer offset:0 atIndex:1];
//     [computeEncoderSum1 setBytes:&params length:sizeof(struct Parameters) atIndex:2];
//
//     [computeEncoderSum1 dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
//     [computeEncoderSum1 endEncoding];
//
//
//
//
//     id<MTLComputeCommandEncoder> computeEncoderSub1 = [commandBuffer computeCommandEncoder];
//     [computeEncoderSub1 setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionSub1 error:&error]];
//     if (error != nil) {
//         const char *errorCString = [[error localizedDescription] UTF8String];
//         printf("Failed to setComputePipelineState sub1: %s\n", errorCString);
//     }
//
//     [computeEncoderSub1 setBuffer:destinationBuffer offset:offset atIndex:0];
//     [computeEncoderSub1 setBuffer:softmaxBuffer offset:offset atIndex:1];
//     [computeEncoderSub1 setBuffer:sumOutBuffer offset:0 atIndex:2];
//     [computeEncoderSub1 setBytes:&params length:sizeof(struct Parameters) atIndex:3];
//     [computeEncoderSub1 dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
//     [computeEncoderSub1 endEncoding];
}

@end