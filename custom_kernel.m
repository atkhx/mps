#import "custom_kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

struct Parameters {
    uint width;
};

@implementation MPSCustomKernelImpl {
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _copyPSO;
    id<MTLComputePipelineState> _fillPSO;
    id<MTLComputePipelineState> _addPSO;
    id<MTLComputePipelineState> _addToPSO;
    id<MTLComputePipelineState> _mulPSO;
    id<MTLComputePipelineState> _divOnSumPSO;
    id<MTLComputePipelineState> _expPSO;
    id<MTLComputePipelineState> _sumPSO;
    id<MTLComputePipelineState> _reluPSO;
    id<MTLComputePipelineState> _reluBwdPSO;
    id<MTLComputePipelineState> _dropoutPSO;
    id<MTLComputePipelineState> _dropoutBwdPSO;
    id<MTLComputePipelineState> _updateWithAdamPSO;
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

        _copyPSO = [self createPipelineStateWithFunctionName:@"copy"];
        _fillPSO = [self createPipelineStateWithFunctionName:@"fill"];
        _addPSO = [self createPipelineStateWithFunctionName:@"add"];
        _addToPSO = [self createPipelineStateWithFunctionName:@"addTo"];
        _mulPSO = [self createPipelineStateWithFunctionName:@"mul"];
        _reluPSO = [self createPipelineStateWithFunctionName:@"relu"];
        _reluBwdPSO = [self createPipelineStateWithFunctionName:@"reluBwd"];
        _dropoutPSO = [self createPipelineStateWithFunctionName:@"dropout"];
        _dropoutBwdPSO = [self createPipelineStateWithFunctionName:@"dropoutBwd"];
        _updateWithAdamPSO = [self createPipelineStateWithFunctionName:@"updateWithAdam"];

        _expPSO = [self createPipelineStateWithFunctionName:@"exp"];
        _sumPSO = [self createPipelineStateWithFunctionName:@"sum"];
        _divOnSumPSO = [self createPipelineStateWithFunctionName:@"divOnSum"];
        //     id<MTLComputePipelineState> _softmaxTrilPSO;
//                id<MTLComputePipelineState> _softmaxTrilBwdPSO;

        _softmaxTrilPSO = [self createPipelineStateWithFunctionName:@"softmaxTril"];
        _softmaxTrilBwdPSO = [self createPipelineStateWithFunctionName:@"smxBufferTrilBwd"];

    }
    return self;
}

- (void) copy:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_copyPSO];
    [computeEncoder setBuffer:dstBuffer offset:dstOffset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:srcOffset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_fillPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBytes:&value length:sizeof(float) atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) add:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addPSO];
    [computeEncoder setBuffer:dstBuffer offset:dstOffset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:srcOffset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) addTo:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addToPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:aBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:bBuffer offset:0 atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) mul:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mulPSO];
    [computeEncoder setBuffer:dstBuffer offset:dstOffset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:srcOffset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) relu:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_reluPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void)reluBwd:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_reluBwdPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:mskBuffer offset:0 atIndex:2];

    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) exp:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_expPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(colsCount * rowsCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) sum:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_sumPSO];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) divOnSum:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_divOnSumPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:sumBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) softmax:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    [self exp:commandBuffer
        dstBuffer:dstBuffer srcBuffer:srcBuffer colsCount:colsCount rowsCount:rowsCount offset:offset];

    [self sum:commandBuffer
        dstBuffer:sumBuffer srcBuffer:dstBuffer colsCount:colsCount rowsCount:rowsCount offset:offset];

    [self divOnSum:commandBuffer
        dstBuffer:dstBuffer sumBuffer:sumBuffer colsCount:colsCount rowsCount:rowsCount offset:offset];
}

- (void) dropout:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer
        probability:(float)probability {

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_dropoutPSO];
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

    [computeEncoder setComputePipelineState:_dropoutBwdPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:mskBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&probability length:sizeof(float) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) updateWithAdam:(id<MTLCommandBuffer>)commandBuffer
        dataBuffer:(id<MTLBuffer>)dataBuffer
        gradBuffer:(id<MTLBuffer>)gradBuffer
        mBuffer:(id<MTLBuffer>)mBuffer
        vBuffer:(id<MTLBuffer>)vBuffer
        beta1:(float)beta1
        beta2:(float)beta2
        beta1powIterationLR:(float)beta1powIterationLR
        beta2powIteration:(float)beta2powIteration
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_updateWithAdamPSO];
    [computeEncoder setBuffer:dataBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:gradBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:mBuffer    offset:0 atIndex:2];
    [computeEncoder setBuffer:vBuffer    offset:0 atIndex:3];
    [computeEncoder setBytes:&beta1 length:sizeof(float) atIndex:4];
    [computeEncoder setBytes:&beta2 length:sizeof(float) atIndex:5];
    [computeEncoder setBytes:&beta1powIterationLR length:sizeof(float) atIndex:6];
    [computeEncoder setBytes:&beta2powIteration length:sizeof(float) atIndex:7];
    [computeEncoder dispatchThreads:MTLSizeMake(dataBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) softmaxTril:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoderSoftmaxTril = [commandBuffer computeCommandEncoder];
    [computeEncoderSoftmaxTril setComputePipelineState:_softmaxTrilPSO];
    [computeEncoderSoftmaxTril setBuffer:sourceBuffer offset:offset atIndex:0];
    [computeEncoderSoftmaxTril setBuffer:destinationBuffer offset:offset atIndex:1];
    [computeEncoderSoftmaxTril setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoderSoftmaxTril dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoderSoftmaxTril endEncoding];
}

- (void) softmaxTrilBwd:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        softmaxBuffer:(id<MTLBuffer>)softmaxBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoderSoftmaxTrilBwd = [commandBuffer computeCommandEncoder];

    [computeEncoderSoftmaxTrilBwd setComputePipelineState:_softmaxTrilBwdPSO];
    [computeEncoderSoftmaxTrilBwd setBuffer:sourceBuffer offset:offset atIndex:0];
    [computeEncoderSoftmaxTrilBwd setBuffer:destinationBuffer offset:offset atIndex:1];
    [computeEncoderSoftmaxTrilBwd setBuffer:softmaxBuffer offset:offset atIndex:2];
    [computeEncoderSoftmaxTrilBwd setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [computeEncoderSoftmaxTrilBwd dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoderSoftmaxTrilBwd endEncoding];
}

@end