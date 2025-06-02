import Foundation

import MetalKit

let count: Int = 3000000

// Helper function for creating random arrays
func getRandomArray() -> [Float] {
  var result = [Float].init(repeating: 0.0, count: count)
  for i in 0..<count {
    result[i] = Float(arc4random_uniform(10))
  }
  return result
}

// Using the do block for error handling
do {
    // Step 1: Import all the MNIST number data
    let trainImagesPath = "/Users/liam/Development/MetalMLP/MetalMLP/Data/train-images-idx3-ubyte"
    let trainImages = try MNISTLoader.loadImages(from: trainImagesPath)
    print("Training images: \(trainImages.count)")
    
    let trainLabelsPath = "/Users/liam/Development/MetalMLP/MetalMLP/Data/train-labels-idx1-ubyte"
    let trainLabels = try MNISTLoader.loadLabels(from: trainLabelsPath)
    print("Training labels: \(trainLabels.count)")
       
    // Step 2: Output a single imsage to confirm the data was imported successfully
    let outputImagePath = "/Users/liam/Development/MetalMLP/MetalMLP/SampleData/output.pgm"
    let fileURL = URL(fileURLWithPath: outputImagePath)
    
    let header = "P5\n\(28) \(28)\n255\n"
    let headerData = Data(header.utf8)
    let pixelData = Data(trainImages[0])
    let combinedData = headerData + pixelData
    
    do {
        try combinedData.write(to: fileURL)
    } catch {
        print("Error writing to file: \(error)")
    }
    
    // Step 3: Normalize all data and prepare it for training
    var mlpInputs: [[Double]] = []
    var mlpTargets: [[Double]] = []
    
    // Number of images to process
    let numToProcess = min(100, trainImages.count)
    
    for i in 0..<numToProcess {
        // Normalize every pixel value in each image to [0, 1]
        let normalizedPixels = trainImages[i].map { pixel in Double(pixel) / 255.0 }
        mlpInputs.append(normalizedPixels)
        
        // Use one-hot encoding for the lables
        var target = Array(repeating: 0.0, count: 10)
        target[Int(trainLabels[i])] = 1.0
        mlpTargets.append(target)
    }
    
    // Step 4: Prepare to use the GPU
//    print("Creating arrays")
//
//    // Create random arrays to sum
//    let array1 = getRandomArray()
//    let array2 = getRandomArray()
//
//    print("Done generating arrays")
//    
//    let device = MTLCreateSystemDefaultDevice()
//    let commandQueue = device?.makeCommandQueue()
//    let gpuFunctionLibary = device?.makeDefaultLibrary()
//    let additionGPUFunction = gpuFunctionLibary?.makeFunction(name: "addition_compute_function")
//    
//    var additionComputePipelineState: MTLComputePipelineState!
//    do {
//        additionComputePipelineState = try device?.makeComputePipelineState(function: additionGPUFunction!)
//    } catch {
//        print(error)
//    }
//    
//    // Create the buffers to be sent to the GPU
//    let arr1Buff = device?.makeBuffer(bytes: array1,
//                                      length: MemoryLayout<Float>.size * count,
//                                      options: .storageModeShared)
//    
//    let arr2Buff = device?.makeBuffer(bytes: array2,
//                                      length: MemoryLayout<Float>.size * count,
//                                      options: .storageModeShared)
//    
//    let resultBuff = device?.makeBuffer(length: MemoryLayout<Float>.size * count,
//                                        options: .storageModeShared)
//    
//    // make the command buffer
//    let commandBuffer = commandQueue?.makeCommandBuffer()
//    
//    // make a command encoder
//    let commandEncoder = commandBuffer?.makeComputeCommandEncoder()
//    commandEncoder?.setComputePipelineState(additionComputePipelineState)
//    
//    //set parameters
//    commandEncoder?.setBuffer(arr1Buff, offset: 0, index: 0)
//    commandEncoder?.setBuffer(arr2Buff, offset: 0, index: 1)
//    commandEncoder?.setBuffer(resultBuff, offset: 0, index: 2)
//    
//    // dispatch threads
//    let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
//    
//    let maxThreadsPerThreadgroup = additionComputePipelineState.maxTotalThreadsPerThreadgroup
//    let threadsPerThreadgroup = MTLSize(width: maxThreadsPerThreadgroup, height: 1, depth: 1)
//    commandEncoder?.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
//    
//    // end encoding
//    commandEncoder?.endEncoding()
//    
//    // commit to the command queue
//    commandBuffer?.commit()
//    
//    // wait until complete
//    commandBuffer?.waitUntilCompleted()
//    
//    var resultBufferPointer = resultBuff?.contents().bindMemory(to: Float.self,
//                                                                capacity: MemoryLayout<Float>.size * count)
//    
//    // Print out array information
//    for i in 0..<3 {
//      print("\(array1[i]) + \(array2[i]) = \(Float(resultBufferPointer!.pointee) as Any)")
//      resultBufferPointer = resultBufferPointer?.advanced(by: 1)
//    }
    
    // Step 4: Prepare to use the GPU
    print("Creating arrays")

    // Create random arrays to sum
    let array1 = getRandomArray()
    let array2 = getRandomArray()

    print("Done generating arrays")

    // Create a Task to work with the GPU actor
    Task { @GPUActor in
        do {
            let additionKernel = try AdditionKernel()
            let resultArray = try additionKernel.execute(array1: array1, array2: array2)
            
            // Print out array information
            for i in 0..<3 {
                print("\(array1[i]) + \(array2[i]) = \(resultArray[i])")
            }
        } catch {
            print("GPU execution failed: \(error)")
        }
    }

    // Since we're using async/await, we need to ensure the GPU work completes before continuing
    // You might want to restructure your main.swift to use async/await throughout
    Thread.sleep(forTimeInterval: 1.0) // Simple wait, better to restructure as async
    
    // Step 5: Train the MLP
    var mlp = MLP(inputSize: 784, hiddenSize: 128, outputSize: 10, learningRate: 0.01)
    
    print("Starting training...")
    let startTime = Date()
    mlp.train(inputs: mlpInputs, targets: mlpTargets, epochs: 20, batchSize: 128)
    let duration = Date().timeIntervalSince(startTime)
    print("Training completed in \(duration) seconds")
    
    // Test for a few images
    for i in 0..<5 {
        let prediction = mlp.predict(mlpInputs[i])
        let actualLabel = trainLabels[i]
        print("Image \(i): Prediction = \(prediction), Actual = \(actualLabel)")
    }
    
} catch {
    print("Encountered an error while attempting to train the MLP: \(error)")
}

