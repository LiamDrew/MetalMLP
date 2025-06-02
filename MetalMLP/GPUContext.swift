//
//  GPUContext.swift
//  MetalMLP
//
//  Created by Liam D on 6/1/25.
//

import Metal

@globalActor
actor GPUActor {
    static let shared = GPUActor()
}

@GPUActor
final class GPUManager {
    static let shared = GPUManager()
    
    let device: MTLDevice
    let functionLibrary: MTLLibrary
    
    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("GPU is not supported on this device")
        }
        self.device = device
        
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Unable to create default Metal library")
        }
        self.functionLibrary = library
    }
    
    func makeCommandQueue() -> MTLCommandQueue {
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        return commandQueue
    }
}
