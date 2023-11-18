#ifndef MPSCustomKernel_h
#define MPSCustomKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MPSCustomKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) copy:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length;

- (void) copyWHD:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        W:(uint)W
        H:(uint)H
        D:(uint)D;

- (void) fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length;

- (void) add:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length;

// todo add offset/length
- (void) addTo:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer;

- (void) addToWHD:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer
        K:(float)K
        W:(uint)W
        H:(uint)H
        D:(uint)D;

- (void) addToWHDBwd:(id<MTLCommandBuffer>)commandBuffer
        aGrad:(id<MTLBuffer>)aGrad
        bGrad:(id<MTLBuffer>)bGrad
        oGrad:(id<MTLBuffer>)oGrad
        W:(uint)W
        H:(uint)H
        D:(uint)D;

- (void) addScalar:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value;

// todo add mulTo
- (void) mul:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length;

- (void) relu:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer;

- (void)reluBwd:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer;

- (void) exp:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

- (void) sumByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

// todo naming (here we div each value in row on their sum
- (void) divOnSum:(id<MTLCommandBuffer>)commandBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount;

- (void) softmax:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

- (void) dropout:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer
        probability:(float)probability;

- (void) dropoutBwd:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        mskBuffer:(id<MTLBuffer>)mskBuffer
        probability:(float)probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;

- (void) updateWithAdam:(id<MTLCommandBuffer>)commandBuffer
        dataBuffer:(id<MTLBuffer>)dataBuffer
        gradBuffer:(id<MTLBuffer>)gradBuffer
        mBuffer:(id<MTLBuffer>)mBuffer
        vBuffer:(id<MTLBuffer>)vBuffer
        beta1:(float)beta1
        beta2:(float)beta2
        beta1powIterationLR:(float)beta1powIterationLR
        beta2powIteration:(float)beta2powIteration;

- (void) softmaxTril:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;

- (void) softmaxTrilBwd:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset;


- (void) crossEntropyPos:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        chunkSize:(uint)chunkSize;

- (void) crossEntropyPosBwd:(id<MTLCommandBuffer>)commandBuffer
        oGrad:(id<MTLBuffer>)oGrad
        aGrad:(id<MTLBuffer>)aGrad
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        chunkSize:(uint)chunkSize;

- (void) rmsNorm:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        chunkSize:(uint)chunkSize;

- (void) rmsNormBwd:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        aggData:(id<MTLBuffer>)aggData
        aggGrad:(id<MTLBuffer>)aggGrad
        chunkSize:(uint)chunkSize;

- (void) meanByRows:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize;

- (void) meanByRowsBwd:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize;

- (void) concatByRows:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset;

- (void) concatByRowsBwd:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset;

- (void) embeddings:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        posEmbedding:(id<MTLBuffer>)posEmbedding
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength;

- (void) embeddingsBwd:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputGrad:(id<MTLBuffer>)outputGrad
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGrad
        featuresCount:(uint)featuresCount;

- (void) transposeTo:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:(uint)width
        height:(uint)height;

- (void) transposeAndAddTo:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:(uint)width
        height:(uint)height;

@end


@interface MPSCustomKernelImpl : NSObject <MPSCustomKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MPSCustomKernel_h */
