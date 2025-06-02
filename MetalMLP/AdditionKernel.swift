////
////  AdditionKernel.swift
////  MetalMLP
////
////  Created by Liam D on 6/1/25.
////
//
//import MetalKit
//
//class AdditionKernel {
//    private let kernel: ComputeKernel
//    private let context = GPUContext.shared
//    
//    init() throws {
//        self.kernel = try ComputeKernel(functionName: "addition_compute_function")
//    }
//    
//    func add(array1: [Float], array2: [Float]) -> [Float] {
//        precondition(array1.count == array2.count, "Arrays must have the same length")
//        
//        let count = array1.count
//        let bufferSize = MemoryLayout<Float>.size * count
//        
//        // Create buffers
//        guard let arr1Buffer = context.device.makeBuffer(bytes: array1,
//                                                         length: bufferSize,
//                                                         options: .storageModeShared),
//              let arr2Buffer = context.device.makeBuffer(bytes: array2,
//                                                         length: bufferSize,
//                                                         options: .storageModeShared),
//              let resultBuffer = context.device.makeBuffer(length: bufferSize,
//                                                          options: .storageModeShared) else {
//            fatalError("Unable to create Metal buffers")
//        }
//        
//        // Create command queue and buffer
//        let commandQueue = context.makeCommandQueue()
//        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
//            fatalError("Unable to create command buffer")
//        }
//        
//        // Configure buffers for the kernel
//        let buffers: [(buffer: MTLBuffer, index: Int)] = [
//            (arr1Buffer, 0),
//            (arr2Buffer, 1),
//            (resultBuffer, 2)
//        ]
//        
//        // Execute kernel
//        let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
//        kernel.execute(commandBuffer: commandBuffer,
//                      buffers: buffers,
//                      threadsPerGrid: threadsPerGrid)
//        
//        // Commit and wait
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        // Read results
//        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: count)
//        return Array(UnsafeBufferPointer(start: resultPointer, count: count))
//    }
//}
