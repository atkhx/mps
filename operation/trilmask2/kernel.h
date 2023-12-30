#ifndef TrilMask2Kernel_h
#define TrilMask2Kernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol TrilMask2Kernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        mask:(float)mask
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount;

@end


@interface TrilMask2KernelImpl : NSObject <TrilMask2Kernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* TrilMask2Kernel_h */
