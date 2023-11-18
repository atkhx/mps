#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

struct Parameters {
    uint width;
};

@implementation MPSCustomKernelImpl {
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _copyPSO;
    id<MTLComputePipelineState> _copyWHDPSO;
    id<MTLComputePipelineState> _fillPSO;
    id<MTLComputePipelineState> _addPSO;
    id<MTLComputePipelineState> _addToPSO;
    id<MTLComputePipelineState> _addToWHDPSO;
    id<MTLComputePipelineState> _addToWHDBwdPSO;
    id<MTLComputePipelineState> _addScalarPSO;
    id<MTLComputePipelineState> _mulPSO;
    id<MTLComputePipelineState> _subMaxByRowPSO;
    id<MTLComputePipelineState> _divOnSumPSO;
    id<MTLComputePipelineState> _expPSO;
    id<MTLComputePipelineState> _maxByRowPSO;
    id<MTLComputePipelineState> _sumByRowPSO;
    id<MTLComputePipelineState> _meanByRowsPSO;
    id<MTLComputePipelineState> _meanByRowsBwdPSO;
    id<MTLComputePipelineState> _sqsByRowPSO;
    id<MTLComputePipelineState> _rmsGradPSO;
    id<MTLComputePipelineState> _sqsGradByRowPSO;
    id<MTLComputePipelineState> _nllByPosPSO;
    id<MTLComputePipelineState> _crossEntropyPosBwdPOS;
    id<MTLComputePipelineState> _reluPSO;
    id<MTLComputePipelineState> _reluBwdPSO;
    id<MTLComputePipelineState> _dropoutPSO;
    id<MTLComputePipelineState> _dropoutBwdPSO;
    id<MTLComputePipelineState> _updateWithAdamPSO;
    id<MTLComputePipelineState> _softmaxTrilPSO;
    id<MTLComputePipelineState> _softmaxTrilBwdPSO;
    id<MTLComputePipelineState> _concatByRowsPSO;
    id<MTLComputePipelineState> _concatByRowsBwdPSO;
    id<MTLComputePipelineState> _embeddingsPSO;
    id<MTLComputePipelineState> _embeddingsBwdPSO;
    id<MTLComputePipelineState> _transposeToPSO;
    id<MTLComputePipelineState> _transposeAndAddToPSO;

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

//     MTLSize maxThreadsPerThreadgroup = [_device maxThreadsPerThreadgroup];
//
//     // Вывод информации
//     NSLog(@"Максимальное количество потоков в рабочей группе: %lu", maxThreadsPerThreadgroup.width);
//     NSLog(@"Максимальное количество потоков в рабочей группе: %lu", maxThreadsPerThreadgroup.height);
//     NSLog(@"Максимальное количество потоков в рабочей группе: %lu", maxThreadsPerThreadgroup.depth);

    return pipelineState;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _copyPSO = [self createPipelineStateWithFunctionName:@"copy"];
        _copyWHDPSO = [self createPipelineStateWithFunctionName:@"copyWHD"];
        _fillPSO = [self createPipelineStateWithFunctionName:@"fill"];
        _addPSO = [self createPipelineStateWithFunctionName:@"add"];
        _addToPSO = [self createPipelineStateWithFunctionName:@"addTo"];
        _addToWHDPSO = [self createPipelineStateWithFunctionName:@"addToWHD"];
        _addToWHDBwdPSO = [self createPipelineStateWithFunctionName:@"addToWHDBwd"];
        _addScalarPSO = [self createPipelineStateWithFunctionName:@"addScalar"];
        _mulPSO = [self createPipelineStateWithFunctionName:@"mul"];
        _reluPSO = [self createPipelineStateWithFunctionName:@"relu"];
        _reluBwdPSO = [self createPipelineStateWithFunctionName:@"reluBwd"];
        _dropoutPSO = [self createPipelineStateWithFunctionName:@"dropout"];
        _dropoutBwdPSO = [self createPipelineStateWithFunctionName:@"dropoutBwd"];
        _updateWithAdamPSO = [self createPipelineStateWithFunctionName:@"updateWithAdam"];

        _expPSO = [self createPipelineStateWithFunctionName:@"exp"];
        _sumByRowPSO = [self createPipelineStateWithFunctionName:@"sumByRow"];
        _meanByRowsPSO = [self createPipelineStateWithFunctionName:@"meanByRows"];
        _meanByRowsBwdPSO = [self createPipelineStateWithFunctionName:@"meanByRowsBwd"];
        _sqsByRowPSO = [self createPipelineStateWithFunctionName:@"sqsByRow"];
        _rmsGradPSO = [self createPipelineStateWithFunctionName:@"rmsGrad"];
        _sqsGradByRowPSO = [self createPipelineStateWithFunctionName:@"sqsGradByRow"];
        _nllByPosPSO = [self createPipelineStateWithFunctionName:@"nllByPos"];
        _maxByRowPSO = [self createPipelineStateWithFunctionName:@"maxByRow"];
        _divOnSumPSO = [self createPipelineStateWithFunctionName:@"divOnSum"];
        _subMaxByRowPSO = [self createPipelineStateWithFunctionName:@"subMaxByRow"];

        _crossEntropyPosBwdPOS = [self createPipelineStateWithFunctionName:@"crossEntropyPosBwd"];

        _softmaxTrilPSO = [self createPipelineStateWithFunctionName:@"softmaxTril"];
        _softmaxTrilBwdPSO = [self createPipelineStateWithFunctionName:@"softmaxBufferTrilBwd"];

        _concatByRowsPSO = [self createPipelineStateWithFunctionName:@"concatByRows"];
        _concatByRowsBwdPSO = [self createPipelineStateWithFunctionName:@"concatByRowsBwd"];

        _embeddingsPSO = [self createPipelineStateWithFunctionName:@"embeddings"];
        _embeddingsBwdPSO = [self createPipelineStateWithFunctionName:@"embeddingsBwd"];

        _transposeToPSO = [self createPipelineStateWithFunctionName:@"transposeTo"];
        _transposeAndAddToPSO = [self createPipelineStateWithFunctionName:@"transposeAndAddTo"];
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
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) copyWHD:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        W:(uint)W
        H:(uint)H
        D:(uint)D
{
    int square = W * H;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_copyWHDPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&W length:sizeof(uint) atIndex:2];
    [computeEncoder setBytes:&H length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&square length:sizeof(uint) atIndex:4];
    [computeEncoder dispatchThreads:MTLSizeMake(W, H, D) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
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

- (void) addToWHD:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer
        K:(float)K
        W:(uint)W
        H:(uint)H
        D:(uint)D
{
    int square = W * H;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addToWHDPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:aBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:bBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&K length:sizeof(float) atIndex:3];
    [computeEncoder setBytes:&W length:sizeof(uint) atIndex:4];
    [computeEncoder setBytes:&square length:sizeof(uint) atIndex:5];
    [computeEncoder dispatchThreads:MTLSizeMake(W, H, D) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) addToWHDBwd:(id<MTLCommandBuffer>)commandBuffer
        aGrad:(id<MTLBuffer>)aGrad
        bGrad:(id<MTLBuffer>)bGrad
        oGrad:(id<MTLBuffer>)oGrad
        W:(uint)W
        H:(uint)H
        D:(uint)D
{
    int square = W * H;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addToWHDBwdPSO];
    [computeEncoder setBuffer:aGrad offset:0 atIndex:0];
    [computeEncoder setBuffer:bGrad offset:0 atIndex:1];
    [computeEncoder setBuffer:oGrad offset:0 atIndex:2];
    [computeEncoder setBytes:&W length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&square length:sizeof(uint) atIndex:4];
    [computeEncoder dispatchThreads:MTLSizeMake(W, H, D) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) addScalar:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addScalarPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBytes:&value length:sizeof(float) atIndex:1];
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

- (void) maxByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_maxByRowPSO];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) subMaxByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        maxBuffer:(id<MTLBuffer>)maxBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_subMaxByRowPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:maxBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) sumByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_sumByRowPSO];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
//     [computeEncoder dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) divOnSum:(id<MTLCommandBuffer>)commandBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_divOnSumPSO];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:sumBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:3];
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
    [self maxByRow:commandBuffer
        dstBuffer:sumBuffer
        srcBuffer:srcBuffer
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];

    [self subMaxByRow:commandBuffer
        dstBuffer:srcBuffer
        maxBuffer:sumBuffer
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];

    [self exp:commandBuffer
        dstBuffer:dstBuffer
        srcBuffer:srcBuffer
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];

    [self sumByRow:commandBuffer
        dstBuffer:sumBuffer
        srcBuffer:dstBuffer
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];

    [self divOnSum:commandBuffer
        srcBuffer:dstBuffer
        dstBuffer:dstBuffer
        sumBuffer:sumBuffer
        colsCount:colsCount
        rowsCount:rowsCount];
}

- (void) nllByPos:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_nllByPosPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:smxBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:tgtBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(tgtBuffer.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) crossEntropyPos:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        chunkSize:(uint)chunkSize
{
    [self softmax:commandBuffer
        dstBuffer:smxBuffer
        srcBuffer:srcBuffer
        sumBuffer:sumBuffer
        colsCount:chunkSize
        rowsCount:srcBuffer.length / (chunkSize*4)
        offset:0
    ];

    [self nllByPos:commandBuffer
        dstBuffer:dstBuffer
        smxBuffer:smxBuffer
        tgtBuffer:tgtBuffer
        chunkSize:chunkSize
    ];
}

- (void) crossEntropyPosBwd:(id<MTLCommandBuffer>)commandBuffer
        oGrad:(id<MTLBuffer>)oGrad
        aGrad:(id<MTLBuffer>)aGrad
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_crossEntropyPosBwdPOS];
        [computeEncoder setBuffer:oGrad offset:0 atIndex:0];
        [computeEncoder setBuffer:aGrad offset:0 atIndex:1];
        [computeEncoder setBuffer:tgtBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:smxBuffer offset:0 atIndex:3];
        [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:4];
        [computeEncoder dispatchThreads:MTLSizeMake(aGrad.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
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
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
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
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
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

- (void) softmaxTril:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_softmaxTrilPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) softmaxTrilBwd:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_softmaxTrilBwdPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:1];
    [computeEncoder setBuffer:smxBuffer offset:offset atIndex:2];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) sqsByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_sqsByRowPSO];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) rmsGradPSO:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        aggData:(id<MTLBuffer>)aggData
        aggGrad:(id<MTLBuffer>)aggGrad
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_rmsGradPSO];
    [computeEncoder setBuffer:inputData offset:0 atIndex:0];
    [computeEncoder setBuffer:inputGrad offset:0 atIndex:1];
    [computeEncoder setBuffer:outputGrad offset:0 atIndex:2];
    [computeEncoder setBuffer:aggData offset:0 atIndex:3];
    [computeEncoder setBuffer:aggGrad offset:0 atIndex:4];
    [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:5];
    [computeEncoder dispatchThreads:MTLSizeMake(inputGrad.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) sqsGradByRow:(id<MTLCommandBuffer>)commandBuffer
        aggData:(id<MTLBuffer>)aggData
        aggGrad:(id<MTLBuffer>)aggGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_sqsGradByRowPSO];

    [computeEncoder setBuffer:aggData offset:0 atIndex:0];
    [computeEncoder setBuffer:aggGrad offset:0 atIndex:1];
    [computeEncoder setBuffer:outputData offset:0 atIndex:2];
    [computeEncoder setBuffer:outputGrad offset:0 atIndex:3];
    [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:4];

    [computeEncoder dispatchThreads:MTLSizeMake(aggData.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) rmsNorm:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        chunkSize:(uint)chunkSize
{
    [self sqsByRow:commandBuffer
        dstBuffer:sumBuffer
        srcBuffer:srcBuffer
        colsCount:chunkSize
        rowsCount:srcBuffer.length / (4 * chunkSize)];

    [self divOnSum:commandBuffer
        srcBuffer:srcBuffer
        dstBuffer:dstBuffer
        sumBuffer:sumBuffer
        colsCount:chunkSize
        rowsCount:srcBuffer.length / (4 * chunkSize)];
}

- (void) rmsNormBwd:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        aggData:(id<MTLBuffer>)aggData
        aggGrad:(id<MTLBuffer>)aggGrad
        chunkSize:(uint)chunkSize
{
    [self sqsGradByRow:commandBuffer
        aggData:aggData
        aggGrad:aggGrad
        outputData:outputData
        outputGrad:outputGrad
        chunkSize:chunkSize];

    [self rmsGradPSO:commandBuffer
        inputData:inputData
        inputGrad:inputGrad
        outputGrad:outputGrad
        aggData:aggData
        aggGrad:aggGrad
        chunkSize:chunkSize];
}

- (void) meanByRows:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_meanByRowsPSO];
        [computeEncoder setBuffer:inputData offset:0 atIndex:0];
        [computeEncoder setBuffer:outputData offset:0 atIndex:1];
        [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:2];
        [computeEncoder dispatchThreads:MTLSizeMake(1, inputData.length/(4*chunkSize), 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) meanByRowsBwd:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_meanByRowsBwdPSO];
        [computeEncoder setBuffer:inputGrad offset:0 atIndex:0];
        [computeEncoder setBuffer:outputGrad offset:0 atIndex:1];
        [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:2];
        [computeEncoder dispatchThreads:MTLSizeMake(inputGrad.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) concatByRows:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_concatByRowsPSO];
        [computeEncoder setBuffer:inputData offset:0 atIndex:0];
        [computeEncoder setBuffer:outputData offset:0 atIndex:1];
        [computeEncoder setBytes:&inputWidth length:sizeof(uint) atIndex:2];
        [computeEncoder setBytes:&outputWidth length:sizeof(uint) atIndex:3];
        [computeEncoder setBytes:&outputOffset length:sizeof(uint) atIndex:4];
        [computeEncoder dispatchThreads:MTLSizeMake(inputWidth, inputData.length/(4*inputWidth), 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) concatByRowsBwd:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_concatByRowsBwdPSO];
        [computeEncoder setBuffer:inputGrad offset:0 atIndex:0];
        [computeEncoder setBuffer:outputGrad offset:0 atIndex:1];
        [computeEncoder setBytes:&inputWidth length:sizeof(uint) atIndex:2];
        [computeEncoder setBytes:&outputWidth length:sizeof(uint) atIndex:3];
        [computeEncoder setBytes:&outputOffset length:sizeof(uint) atIndex:4];
        [computeEncoder dispatchThreads:MTLSizeMake(inputWidth, inputGrad.length/(4*inputWidth), 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) embeddings:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        posEmbedding:(id<MTLBuffer>)posEmbedding
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_embeddingsPSO];
        [computeEncoder setBuffer:inputData offset:0 atIndex:0];
        [computeEncoder setBuffer:outputData offset:0 atIndex:1];
        [computeEncoder setBuffer:posEmbedding offset:0 atIndex:2];
        [computeEncoder setBuffer:tokenEmbedding offset:0 atIndex:3];

        [computeEncoder setBytes:&featuresCount length:sizeof(uint) atIndex:4];
        [computeEncoder setBytes:&contextLength length:sizeof(uint) atIndex:5];

        [computeEncoder dispatchThreads:MTLSizeMake(featuresCount, outputData.length/(4*featuresCount), 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) embeddingsBwd:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputGrad:(id<MTLBuffer>)outputGrad
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGrad
        featuresCount:(uint)featuresCount
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_embeddingsBwdPSO];
        [computeEncoder setBuffer:inputData offset:0 atIndex:0];
        [computeEncoder setBuffer:outputGrad offset:0 atIndex:1];
        [computeEncoder setBuffer:tokenEmbeddingGrad offset:0 atIndex:2];
        [computeEncoder setBytes:&featuresCount length:sizeof(uint) atIndex:3];
        [computeEncoder dispatchThreads:MTLSizeMake(featuresCount, outputGrad.length/(4*featuresCount), 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}


- (void) transposeTo:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:(uint)width
        height:(uint)height
{
    int square = width * height;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_transposeToPSO];
        [computeEncoder setBuffer:inputData offset:0 atIndex:0];
        [computeEncoder setBuffer:outputData offset:0 atIndex:1];

        [computeEncoder setBytes:&width length:sizeof(uint) atIndex:2];
        [computeEncoder setBytes:&height length:sizeof(uint) atIndex:3];
        [computeEncoder setBytes:&square length:sizeof(uint) atIndex:4];

        [computeEncoder dispatchThreads:MTLSizeMake(width, height, outputData.length/(4*square)) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) transposeAndAddTo:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:(uint)width
        height:(uint)height
{
    int square = width * height;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_transposeAndAddToPSO];
        [computeEncoder setBuffer:inputData offset:0 atIndex:0];
        [computeEncoder setBuffer:outputData offset:0 atIndex:1];

        [computeEncoder setBytes:&width length:sizeof(uint) atIndex:2];
        [computeEncoder setBytes:&height length:sizeof(uint) atIndex:3];
        [computeEncoder setBytes:&square length:sizeof(uint) atIndex:4];

        [computeEncoder dispatchThreads:MTLSizeMake(width, height, outputData.length/(4*square)) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

@end