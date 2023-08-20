#ifndef ClearBuffer_h
#define ClearBuffer_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol ClearBuffer <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device;
- (void) clearBuffer:(id<MTLBuffer>)buffer
        withValue:(float)value
        commandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface ClearBufferImpl : NSObject <ClearBuffer>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* ClearBuffer_h */
