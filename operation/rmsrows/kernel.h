#ifndef RmsRowsKernel_h
#define RmsRowsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol RmsRowsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inputGrads:(id<MTLBuffer>)inputGrads
        outputGrads:(id<MTLBuffer>)outputGrads
        chunkSize:(uint)chunkSize;

@end


@interface RmsRowsKernelImpl : NSObject <RmsRowsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* RmsRowsKernel_h */
