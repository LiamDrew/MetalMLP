////
////  ComputeKernel.swift
////  MetalMLP
////
////  Created by Liam D on 6/1/25.
////
//
//import MetalKit
//
//class ComputeKernel {
//    private let pipelineState: MTLComputePipelineState
//    private let functionName: String
//    
//    init(functionName: String) throws {
//        self.functionName = functionName
//        
//        let context = GPUContext.shared
//        guard let function = context.library.makeFunction(name: functionName) else {
//            throw ComputeKernelError.functionNotFound(functionName)
//        }
//        
//        self.pipelineState = try context.device.makeComputePipelineState(function: function)
//    }
//    
//    var maxTotalThreadsPerThreadgroup: Int {
//        return pipelineState.maxTotalThreadsPerThreadgroup
//    }
//    
//    func execute(commandBuffer: MTLCommandBuffer,
//                 buffers: [(buffer: MTLBuffer, index: Int)],
//                 threadsPerGrid: MTLSize,
//                 threadsPerThreadgroup: MTLSize? = nil) {
//        
//        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
//            fatalError("Unable to create compute command encoder")
//        }
//        
//        encoder.setComputePipelineState(pipelineState)
//        
//        // Set all buffers
//        for (buffer, index) in buffers {
//            encoder.setBuffer(buffer, offset: 0, index: index)
//        }
//        
//        // Calculate threads per threadgroup if not provided
//        let actualThreadsPerThreadgroup = threadsPerThreadgroup ?? MTLSize(
//            width: min(maxTotalThreadsPerThreadgroup, threadsPerGrid.width),
//            height: 1,
//            depth: 1
//        )
//        
//        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: actualThreadsPerThreadgroup)
//        encoder.endEncoding()
//    }
//}
//
//enum ComputeKernelError: Error {
//    case functionNotFound(String)
//}

import Metal
import Foundation

enum KernelError: Error {
    case functionNotFound(String)
    case failedToCreatePipeline(Error)
    case invalidBufferSize
    case commandCreationFailed
}

@GPUActor
class AdditionKernel {
    private let computePipelineState: MTLComputePipelineState
    private let commandQueue: MTLCommandQueue
    
    init() throws {
        let manager = GPUManager.shared
        let device = manager.device
        self.commandQueue = manager.makeCommandQueue()
        
        guard let additionFunction = manager.functionLibrary.makeFunction(name: "addition_compute_function") else {
            throw KernelError.functionNotFound("addition_compute_function")
        }
        
        do {
            self.computePipelineState = try device.makeComputePipelineState(function: additionFunction)
        } catch {
            throw KernelError.failedToCreatePipeline(error)
        }
    }
    
    func execute(array1: [Float], array2: [Float]) throws -> [Float] {
        let count = array1.count
        
        guard array2.count == count else {
            throw KernelError.invalidBufferSize
        }
        
        let device = GPUManager.shared.device
        
        // Create buffers
        guard let arr1Buff = device.makeBuffer(bytes: array1,
                                              length: MemoryLayout<Float>.size * count,
                                              options: .storageModeShared),
              let arr2Buff = device.makeBuffer(bytes: array2,
                                              length: MemoryLayout<Float>.size * count,
                                              options: .storageModeShared),
              let resultBuff = device.makeBuffer(length: MemoryLayout<Float>.size * count,
                                                options: .storageModeShared) else {
            throw KernelError.invalidBufferSize
        }
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KernelError.commandCreationFailed
        }
        
        // Set up encoder
        commandEncoder.setComputePipelineState(computePipelineState)
        commandEncoder.setBuffer(arr1Buff, offset: 0, index: 0)
        commandEncoder.setBuffer(arr2Buff, offset: 0, index: 1)
        commandEncoder.setBuffer(resultBuff, offset: 0, index: 2)
        
        // Dispatch threads
        let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
        let maxThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerThreadgroup = MTLSize(width: maxThreadsPerThreadgroup, height: 1, depth: 1)
        commandEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        // End encoding and commit
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Get results
        let resultPtr = resultBuff.contents().bindMemory(to: Float.self, capacity: count)
        let resultArray = Array(UnsafeBufferPointer(start: resultPtr, count: count))
        return resultArray
    }
}
