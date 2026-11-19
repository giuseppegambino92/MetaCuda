// metal_baseline — native Metal reference numbers for the same launch pattern
// bench.c measures (16 threadgroups x 256 threads, serial encoder, barriers).
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <stdio.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                         "kernel void k(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] += 1.0f; }";
        NSError *e = nil;
        id<MTLLibrary> l = [d newLibraryWithSource:src options:nil error:&e];
        id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[l newFunctionWithName:@"k"] error:&e];
        id<MTLBuffer> b = [d newBufferWithLength:4096 * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [d newCommandQueue];
        printf("native Metal baseline on %s\n", d.name.UTF8String);
        for (int rep = 0; rep < 2; rep++) {
            double t0 = CACurrentMediaTime();
            for (int i = 0; i < 200; i++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                [en setComputePipelineState:ps];
                [en setBuffer:b offset:0 atIndex:0];
                [en dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [en endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
            double t1 = CACurrentMediaTime();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:ps];
            [en setBuffer:b offset:0 atIndex:0];
            for (int i = 0; i < 20000; i++) {
                [en dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [en memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [en endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double t2 = CACurrentMediaTime();
            if (rep) {
                printf("  dispatch + waitUntilCompleted   : %.1f us round trip\n", (t1 - t0) / 200 * 1e6);
                printf("  dependent dispatches, batched   : %.2f us/dispatch\n", (t2 - t1) / 20000 * 1e6);
            }
        }
    }
    return 0;
}
