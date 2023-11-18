#include "framework.h"

void* customKernelCreate(void *deviceID, const char *kernelSource) {
    return [[MPSCustomKernelImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelCopy(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID copy:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        dstOffset:dstOffset
        srcOffset:srcOffset
        length:length];
}


void customKernelCopyWHD(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint W,
    const uint H,
    const uint D
) {
    [(__bridge MPSCustomKernelImpl*)kernelID copyWHD:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        W:W
        H:H
        D:D
    ];
}

void customKernelFill(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    float value,
    const uint offset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID fill:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        value:value
        offset:offset
        length:length];
}

void customKernelAdd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID add:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        dstOffset:dstOffset
        srcOffset:srcOffset
        length:length];
}

void customKernelAddTo(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *aBuffer,
    void *bBuffer
) {
    [(__bridge MPSCustomKernelImpl*)kernelID addTo:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer];
}

void customKernelAddToWHD(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *aBuffer,
    void *bBuffer,
    const float K,
    const uint W,
    const uint H,
    const uint D
) {
    [(__bridge MPSCustomKernelImpl*)kernelID addToWHD:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer
        K:K
        W:W
        H:H
        D:D
    ];
}

void customKernelAddToWHDBwd(
    void *kernelID,
    void *commandBufferID,
    void *aGrad,
    void *bGrad,
    void *oGrad,
    const uint W,
    const uint H,
    const uint D
) {
    [(__bridge MPSCustomKernelImpl*)kernelID addToWHDBwd:(id<MTLCommandBuffer>)commandBufferID
        aGrad:(id<MTLBuffer>)aGrad
        bGrad:(id<MTLBuffer>)bGrad
        oGrad:(id<MTLBuffer>)oGrad
        W:W
        H:H
        D:D
    ];
}

void customKernelAddScalar(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    float value
) {
    [(__bridge MPSCustomKernelImpl*)kernelID addScalar:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        value:value];
}

void customKernelMul(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID mul:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        dstOffset:dstOffset
        srcOffset:srcOffset
        length:length];
}

void customKernelReLU(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID
) {
    [(__bridge MPSCustomKernelImpl*)kernelID relu:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID];
}

void customKernelReLUBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID
) {
    [(__bridge MPSCustomKernelImpl*)kernelID reluBwd:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        mskBuffer:(id<MTLBuffer>)mskBufferID];
}

void customKernelSoftmax(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *sumBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge MPSCustomKernelImpl*)kernelID softmax:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        sumBuffer:(id<MTLBuffer>)sumBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];
}

void customKernelDropout(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID,
    float probability
) {
    [(__bridge MPSCustomKernelImpl*)kernelID dropout:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        mskBuffer:(id<MTLBuffer>)mskBufferID
        probability:probability];
}

void customKernelDropoutBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID,
    float probability
) {
    [(__bridge MPSCustomKernelImpl*)kernelID
        dropoutBwd:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        mskBuffer:(id<MTLBuffer>)mskBufferID
        probability:probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

void customKernelUpdateWithAdam(
    void *kernelID,
    void *commandBufferID,
    void *dataBufferID,
    void *gradBufferID,
    void *mBufferID,
    void *vBufferID,
    float beta1,
    float beta2,
    float beta1powIterationLR,
    float beta2powIteration
) {
    [(__bridge MPSCustomKernelImpl*)kernelID updateWithAdam:(id<MTLCommandBuffer>)commandBufferID
        dataBuffer:(id<MTLBuffer>)dataBufferID
        gradBuffer:(id<MTLBuffer>)gradBufferID
        mBuffer:(id<MTLBuffer>)mBufferID
        vBuffer:(id<MTLBuffer>)vBufferID
        beta1:beta1
        beta2:beta2
        beta1powIterationLR:beta1powIterationLR
        beta2powIteration:beta2powIteration];
}

void customKernelSoftmaxTrilFwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge MPSCustomKernelImpl*)kernelID softmaxTril:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];
}

void customKernelSoftmaxTrilBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *smxBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge MPSCustomKernelImpl*)kernelID softmaxTrilBwd:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        smxBuffer:(id<MTLBuffer>)smxBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];
}

void customKernelCrossEntropyPos(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *smxBufferID,
    void *sumBufferID,
    void *tgtBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID crossEntropyPos:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        smxBuffer:(id<MTLBuffer>)smxBufferID
        sumBuffer:(id<MTLBuffer>)sumBufferID
        tgtBuffer:(id<MTLBuffer>)tgtBufferID
        chunkSize:chunkSize];
}

void customKernelCrossEntropyPosBwd(
    void *kernelID,
    void *commandBufferID,
    void *oGradBufferID,
    void *aGradBufferID,
    void *tgtBufferID,
    void *smxBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID crossEntropyPosBwd:(id<MTLCommandBuffer>)commandBufferID
        oGrad:(id<MTLBuffer>)oGradBufferID
        aGrad:(id<MTLBuffer>)aGradBufferID
        tgtBuffer:(id<MTLBuffer>)tgtBufferID
        smxBuffer:(id<MTLBuffer>)smxBufferID
        chunkSize:chunkSize];
}

void customKernelRMSNorm(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *sumBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID rmsNorm:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        sumBuffer:(id<MTLBuffer>)sumBufferID
        chunkSize:chunkSize];
}

void customKernelRMSNormBwd(
    void *kernelID,
    void *commandBufferID,
    void *inputDataBufferID,
    void *inputGradBufferID,
    void *outputDataBufferID,
    void *outputGradBufferID,
    void *aggDataBufferID,
    void *aggGradBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID rmsNormBwd:(id<MTLCommandBuffer>)commandBufferID
        inputData:(id<MTLBuffer>)inputDataBufferID
        inputGrad:(id<MTLBuffer>)inputGradBufferID
        outputData:(id<MTLBuffer>)outputDataBufferID
        outputGrad:(id<MTLBuffer>)outputGradBufferID
        aggData:(id<MTLBuffer>)aggDataBufferID
        aggGrad:(id<MTLBuffer>)aggGradBufferID
        chunkSize:chunkSize];
}

void customKernelMeanByRows(
    void *kernelID,
    void *commandBufferID,
    void *inputDataBufferID,
    void *outputDataBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID meanByRows:(id<MTLCommandBuffer>)commandBufferID
        inputData:(id<MTLBuffer>)inputDataBufferID
        outputData:(id<MTLBuffer>)outputDataBufferID
        chunkSize:chunkSize];
}

void customKernelMeanByRowsBwd(
    void *kernelID,
    void *commandBufferID,
    void *inputGradBufferID,
    void *outputGradBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID meanByRowsBwd:(id<MTLCommandBuffer>)commandBufferID
        inputGrad:(id<MTLBuffer>)inputGradBufferID
        outputGrad:(id<MTLBuffer>)outputGradBufferID
        chunkSize:chunkSize];
}

void customKernelConcatByRows(
    void *kernelID,
    void *commandBufferID,
    void *inputDataBufferID,
    void *outputDataBufferID,
    uint inputWidth,
    uint outputWidth,
    uint outputOffset
) {
    [(__bridge MPSCustomKernelImpl*)kernelID concatByRows:(id<MTLCommandBuffer>)commandBufferID
        inputData:(id<MTLBuffer>)inputDataBufferID
        outputData:(id<MTLBuffer>)outputDataBufferID
        inputWidth:inputWidth
        outputWidth:outputWidth
        outputOffset:outputOffset];
}

void customKernelConcatByRowsBwd(
    void *kernelID,
    void *commandBufferID,
    void *inputGradBufferID,
    void *outputGradBufferID,
    uint inputWidth,
    uint outputWidth,
    uint outputOffset
) {
    [(__bridge MPSCustomKernelImpl*)kernelID concatByRowsBwd:(id<MTLCommandBuffer>)commandBufferID
        inputGrad:(id<MTLBuffer>)inputGradBufferID
        outputGrad:(id<MTLBuffer>)outputGradBufferID
        inputWidth:inputWidth
        outputWidth:outputWidth
        outputOffset:outputOffset];
}

void customKernelEmbeddings(
    void *kernelID,
    void *commandBufferID,
    void *inputDataBufferID,
    void *outputDataBufferID,
    void *posEmbeddingBufferID,
    void *tokenEmbeddingBufferID,
    uint featuresCount,
    uint contextLength
) {
    [(__bridge MPSCustomKernelImpl*)kernelID embeddings:(id<MTLCommandBuffer>)commandBufferID
        inputData:(id<MTLBuffer>)inputDataBufferID
        outputData:(id<MTLBuffer>)outputDataBufferID
        posEmbedding:(id<MTLBuffer>)posEmbeddingBufferID
        tokenEmbedding:(id<MTLBuffer>)tokenEmbeddingBufferID
        featuresCount:featuresCount
        contextLength:contextLength];
}

void customKernelEmbeddingsBwd(
    void *kernelID,
    void *commandBufferID,
    void *inputDataBufferID,
    void *outputGradBufferID,
    void *tokenEmbeddingGradBufferID,
    uint featuresCount
) {
    [(__bridge MPSCustomKernelImpl*)kernelID embeddingsBwd:(id<MTLCommandBuffer>)commandBufferID
        inputData:(id<MTLBuffer>)inputDataBufferID
        outputGrad:(id<MTLBuffer>)outputGradBufferID
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGradBufferID
        featuresCount:featuresCount];
}

void transposeTo(
    void *kernelID,
    void *commandBufferID,
    void *inputData,
    void *outputData,
    uint width,
    uint height
) {
    [(__bridge MPSCustomKernelImpl*)kernelID transposeTo:(id<MTLCommandBuffer>)commandBufferID
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:width
        height:height];
}

void transposeAndAddTo(
    void *kernelID,
    void *commandBufferID,
    void *inputData,
    void *outputData,
    uint width,
    uint height
) {
    [(__bridge MPSCustomKernelImpl*)kernelID transposeAndAddTo:(id<MTLCommandBuffer>)commandBufferID
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:width
        height:height];
}

