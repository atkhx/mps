#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation TrilMask2KernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _trilMask2PSO;
    id<MTLComputePipelineState> _trilMask2BwdPSO;

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

        _trilMask2PSO = [self createPipelineStateWithFunctionName:@"trilMask2"];
        _trilMask2BwdPSO = [self createPipelineStateWithFunctionName:@"trilMask2Bwd"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        mask:(float)mask
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    uint depth = inputData.length/(sizeof(float)*colsCount*rowsCount);
    id<MTLComputeCommandEncoder> trilMask2 = [commandBuffer computeCommandEncoder];
    [trilMask2 setComputePipelineState:_trilMask2PSO];
    [trilMask2 setBuffer:inputData offset:0 atIndex:0];
    [trilMask2 setBuffer:outputData offset:0 atIndex:1];
    [trilMask2 setBytes:&mask length:sizeof(float) atIndex:2];
    [trilMask2 setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [trilMask2 setBytes:&rowsCount length:sizeof(uint) atIndex:4];
    [trilMask2 dispatchThreads:MTLSizeMake(colsCount, rowsCount, depth) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [trilMask2 endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    uint depth = inputGrad.length/(sizeof(float)*colsCount*rowsCount);
    id<MTLComputeCommandEncoder> trilMask2Grads = [commandBuffer computeCommandEncoder];

    [trilMask2Grads setComputePipelineState:_trilMask2BwdPSO];
    [trilMask2Grads setBuffer:inputGrad offset:0 atIndex:0];
    [trilMask2Grads setBytes:&colsCount length:sizeof(uint) atIndex:1];
    [trilMask2Grads setBytes:&rowsCount length:sizeof(uint) atIndex:2];
    [trilMask2Grads dispatchThreads:MTLSizeMake(colsCount, rowsCount, depth) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [trilMask2Grads endEncoding];
}

@end