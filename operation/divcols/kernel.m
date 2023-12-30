#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation DivColsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _divColsPSO;
    id<MTLComputePipelineState> _calcInputGradsPSO;
    id<MTLComputePipelineState> _calcWeightsGradsPSO;

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

        _divColsPSO = [self createPipelineStateWithFunctionName:@"divCols"];
        _calcInputGradsPSO   = [self createPipelineStateWithFunctionName:@"calcInputGrads"];
        _calcWeightsGradsPSO = [self createPipelineStateWithFunctionName:@"calcWeightsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        rowWidth:(uint)rowWidth
        colHeight:(uint)colHeight
{
    uint depth = inputData.length / (sizeof(float) * rowWidth * colHeight);

    id<MTLComputeCommandEncoder> divCols = [commandBuffer computeCommandEncoder];
    [divCols setComputePipelineState:_divColsPSO];
    [divCols setBuffer:inputData offset:0 atIndex:0];
    [divCols setBuffer:weightsData offset:0 atIndex:1];
    [divCols setBuffer:outputData offset:0 atIndex:2];
    [divCols setBytes:&rowWidth length:sizeof(uint) atIndex:3];
    [divCols setBytes:&colHeight length:sizeof(uint) atIndex:4];
    [divCols dispatchThreads:MTLSizeMake(rowWidth, colHeight, depth) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [divCols endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        rowWidth:(uint)rowWidth
        colHeight:(uint)colHeight
{
    uint depth = inputData.length / (sizeof(float) * rowWidth * colHeight);

    id<MTLComputeCommandEncoder> calcInputGrads = [commandBuffer computeCommandEncoder];
    [calcInputGrads setComputePipelineState:_calcInputGradsPSO];
    [calcInputGrads setBuffer:inputGrad offset:0 atIndex:0];
    [calcInputGrads setBuffer:weightsData offset:0 atIndex:1];
    [calcInputGrads setBuffer:outputGrad offset:0 atIndex:2];
    [calcInputGrads setBytes:&rowWidth length:sizeof(uint) atIndex:3];
    [calcInputGrads setBytes:&colHeight length:sizeof(uint) atIndex:4];
    [calcInputGrads dispatchThreads:MTLSizeMake(rowWidth, colHeight, depth) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcInputGrads endEncoding];

    id<MTLComputeCommandEncoder> calcWeightsGrads = [commandBuffer computeCommandEncoder];
    [calcWeightsGrads setComputePipelineState:_calcWeightsGradsPSO];
    [calcWeightsGrads setBuffer:inputData offset:0 atIndex:0];
    [calcWeightsGrads setBuffer:weightsData offset:0 atIndex:1];
    [calcWeightsGrads setBuffer:weightsGrad offset:0 atIndex:2];
    [calcWeightsGrads setBuffer:outputGrad offset:0 atIndex:3];
    [calcWeightsGrads setBytes:&rowWidth length:sizeof(uint) atIndex:4]; // colsCount
    [calcWeightsGrads setBytes:&colHeight length:sizeof(uint) atIndex:5]; // rowsCount
    [calcWeightsGrads setBytes:&depth length:sizeof(uint) atIndex:6]; // rowsCount
    [calcWeightsGrads dispatchThreads:MTLSizeMake(colHeight, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcWeightsGrads endEncoding];
}

@end