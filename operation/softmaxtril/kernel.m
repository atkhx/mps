#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation SoftmaxtrilKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _softmaxTrilPSO;
    id<MTLComputePipelineState> _softmaxTrilBwdPSO;

    NSError *error;
}

- (id<MTLComputePipelineState>)createPipelineStateWithFunctionName:(NSString *)functionName {
    id<MTLFunction> function = [self.library newFunctionWithName:functionName];
    if (!function) {
        printf("Failed to load function %s!\n", [functionName UTF8String]);
        return nil;
    }

    id<MTLComputePipelineState> pipelineState = [_device newComputePipelineStateWithFunction:function error:&error];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to create pipeline state: %s\n", errorCString);
        return nil;
    }
    return pipelineState;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _softmaxTrilPSO = [self createPipelineStateWithFunctionName:@"softmaxTril"];
        _softmaxTrilBwdPSO = [self createPipelineStateWithFunctionName:@"softmaxBufferTrilBwd"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    uint depth = inputData.length/(sizeof(float)*colsCount*rowsCount);
    id<MTLComputeCommandEncoder> softmaxTril = [commandBuffer computeCommandEncoder];
    [softmaxTril setComputePipelineState:_softmaxTrilPSO];
    [softmaxTril setBuffer:inputData offset:0 atIndex:0];
    [softmaxTril setBuffer:outputData offset:0 atIndex:1];
    [softmaxTril setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [softmaxTril setBytes:&rowsCount length:sizeof(uint) atIndex:3];
    [softmaxTril dispatchThreads:MTLSizeMake(1, rowsCount, depth) threadsPerThreadgroup:MTLSizeMake(1, 32, 32)];
    [softmaxTril endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        outputData:(id<MTLBuffer>)outputData
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    uint depth = inputGrad.length/(sizeof(float)*colsCount*rowsCount);
    id<MTLComputeCommandEncoder> softmaxTrilGrads = [commandBuffer computeCommandEncoder];

    [softmaxTrilGrads setComputePipelineState:_softmaxTrilBwdPSO];
    [softmaxTrilGrads setBuffer:inputGrad offset:0 atIndex:0];
    [softmaxTrilGrads setBuffer:outputGrad offset:0 atIndex:1];
    [softmaxTrilGrads setBuffer:outputData offset:0 atIndex:2];
    [softmaxTrilGrads setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [softmaxTrilGrads setBytes:&rowsCount length:sizeof(uint) atIndex:4];
    [softmaxTrilGrads dispatchThreads:MTLSizeMake(1, rowsCount, depth) threadsPerThreadgroup:MTLSizeMake(1, 32, 32)];
    [softmaxTrilGrads endEncoding];
}

@end