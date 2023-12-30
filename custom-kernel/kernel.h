#ifndef MPSCustomKernel_h
#define MPSCustomKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MPSCustomKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length;

- (void) updateWithAdam:(id<MTLCommandBuffer>)commandBuffer
        dataBuffer:(id<MTLBuffer>)dataBuffer
        gradBuffer:(id<MTLBuffer>)gradBuffer
        mBuffer:(id<MTLBuffer>)mBuffer
        vBuffer:(id<MTLBuffer>)vBuffer
        beta1:(float)beta1
        beta2:(float)beta2
        beta1powIterationLR:(float)beta1powIterationLR
        beta2powIteration:(float)beta2powIteration;

@end


@interface MPSCustomKernelImpl : NSObject <MPSCustomKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MPSCustomKernel_h */
