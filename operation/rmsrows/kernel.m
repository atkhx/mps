#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation RmsRowsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _rmsByRowsPSO;
    id<MTLComputePipelineState> _rmsByRowsGradsPSO;

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

        _rmsByRowsPSO      = [self createPipelineStateWithFunctionName:@"rmsByRows"];
        _rmsByRowsGradsPSO = [self createPipelineStateWithFunctionName:@"rmsByRowsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> rmsByRows = [commandBuffer computeCommandEncoder];
    [rmsByRows setComputePipelineState:_rmsByRowsPSO];
    [rmsByRows setBuffer:inputData offset:0 atIndex:0];
    [rmsByRows setBuffer:outputData offset:0 atIndex:1];
    [rmsByRows setBytes:&chunkSize length:sizeof(uint) atIndex:2];
    [rmsByRows dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [rmsByRows endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inputGrads:(id<MTLBuffer>)inputGrads
        outputGrads:(id<MTLBuffer>)outputGrads
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> rmsByRowsGrads = [commandBuffer computeCommandEncoder];
    [rmsByRowsGrads setComputePipelineState:_rmsByRowsGradsPSO];
    [rmsByRowsGrads setBuffer:inputData offset:0 atIndex:0];
    [rmsByRowsGrads setBuffer:outputData offset:0 atIndex:1];
    [rmsByRowsGrads setBuffer:inputGrads offset:0 atIndex:2];
    [rmsByRowsGrads setBuffer:outputGrads offset:0 atIndex:3];
    [rmsByRowsGrads setBytes:&chunkSize length:sizeof(uint) atIndex:4];
    [rmsByRowsGrads dispatchThreads:MTLSizeMake(chunkSize, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [rmsByRowsGrads endEncoding];
}

@end