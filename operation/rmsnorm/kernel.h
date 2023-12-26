#ifndef RMSNormKernel_h
#define RMSNormKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol RMSNormKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        aggData:(id<MTLBuffer>)aggData
        chunkSize:(uint)chunkSize;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        aggData:(id<MTLBuffer>)aggData
        aggGrad:(id<MTLBuffer>)aggGrad
        chunkSize:(uint)chunkSize;

@end


@interface RMSNormKernelImpl : NSObject <RMSNormKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* RMSNormKernel_h */
