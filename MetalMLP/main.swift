import Foundation

import Metal
import MetalKit
import MetalPerformanceShaders
// Step 1: Import all the MNIST number data

struct MNISTFashionLoader {
    static func loadImages(from path: String) throws -> [[UInt8]] {
        
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        
        /* MNIST image files start with a 16-byte header
         * Bytes 0-3: magic number (should be 2051)
         * Bytes 4-7: number of images
         * Bytes 8-11: number of rows (28)
         * Bytes 12-15: number of columns (28) */
        
        let headerSize = 16
        let imageSize = 28 * 28 // 784 pixels per image
        
        let numImages = data.withUnsafeBytes { bytes in
            Int(bytes.load(fromByteOffset: 4, as: UInt32.self).bigEndian)
        }
        
        var images: [[UInt8]] = []
        
        for i in 0..<numImages {
            let offset = headerSize + (i * imageSize)
            let imageData = Array(data[offset..<(offset + imageSize)])
            images.append(imageData)
        }
        
        return images
    }
    
    static func loadLabels(from path: String) throws -> [UInt8] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        
        /* MNIST labels start with an 8 byte header
         * Bytes 0-3: magic number (should be 2049)
         * Bytes 4-7: number of labels */
        
        let headerSize = 8
        
        let numLabels = data.withUnsafeBytes { bytes in
            Int(bytes.load(fromByteOffset: 4, as: UInt32.self).bigEndian)
        }
        
        let labels = Array(data[headerSize..<(headerSize + numLabels)])
        return labels
    }
}



struct MLP {
    let inputSize: Int
    let hiddenSize: Int
    let outputSize: Int
    
    var weightsInputHidden: [[Double]]
    var biasesHidden: [Double]
    var weightsHiddenOutput: [[Double]]
    var biasesOutput: [Double]
    
//    let device = MTLCreateSystemDefaultDevice()
//    let commandQueue
    
    var learningRate: Double
    
    init(inputSize: Int, hiddenSize: Int, outputSize: Int, learningRate: Double = 0.01) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learningRate = learningRate
        
        // Xavier Glorot initialization
        let inputScale = sqrt(2.0 / Double(inputSize))
        let hiddenScale = sqrt(2.0 / Double(hiddenSize))
        
        // Initialize weights from input to hidden layer (2d Array)
        weightsInputHidden = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        for i in 0..<inputSize {
            for j in 0..<hiddenSize {
                weightsInputHidden[i][j] = (Double.random(in: -1.0...1.0) * inputScale)
            }
        }
        
        biasesHidden = Array(repeating: 0.0, count: hiddenSize)
        
        weightsHiddenOutput = Array(repeating: Array(repeating: 0.0, count: outputSize), count: hiddenSize)
        
        for i in 0..<hiddenSize {
            for j in 0..<outputSize {
                weightsHiddenOutput[i][j] = (Double.random(in: -1.0...1.0) * hiddenScale)
            }
        }
        
        // initialize output layer biases
        biasesOutput = Array(repeating: 0.0, count: outputSize)
    }
    
}

extension MLP {
    // relu activation function
    func relu(_ x: Double) -> Double {
        return max(0, x)
    }
    
    // relu derivative
    // if x > 0, return 1, else 0
    func reluDerivative(_ x: Double) -> Double {
        return x > 0 ? 1 : 0
    }
    
    // softmax activation function for output layer
    func softmax(_ x: [Double]) -> [Double] {
        let expValues = x.map { exp($0) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
    
    // forward pass thru the network
    func forward(input: [Double]) -> ([Double], [Double], [Double]) {
        // ensure input is the correct size
        assert(input.count == inputSize, "Input size mismatch")
        
        // hidden layer activations
        var hiddenLayerInput = Array(repeating: 0.0, count: hiddenSize)
        for j in 0..<hiddenSize {
            var sum = biasesHidden[j]
            for i in 0..<inputSize {
                sum += input[i] * weightsInputHidden[i][j]
            }
            hiddenLayerInput[j] = sum
        }
        
        let hiddenLayerOutput = hiddenLayerInput.map { relu($0) }
        
        // Calculating output layer activations
        var outputLayerInput = Array(repeating: 0.0, count: outputSize)
        for j in 0..<outputSize {
            var sum = biasesOutput[j]
            for i in 0..<hiddenSize {
                sum += hiddenLayerOutput[i] * weightsHiddenOutput[i][j]
            }
            outputLayerInput[j] = sum
        }
        
        // apply activation function
        let outputLayerOutput = softmax(outputLayerInput)
        
        return (hiddenLayerInput, hiddenLayerOutput, outputLayerOutput)
    }
    
    // calculate loss (cross-entropy loss)
    func calculateLoss(_ output: [Double], _ target: [Double]) -> Double {
        var loss: Double = 0.0
        for i in 0..<output.count {
            loss -= target[i] * log(max(output[i], 1e-10))
        }
        
        return loss
    }
    
}


extension MLP {
    // Train network on a single sample
    mutating func trainSample(_ input: [Double], _ target: [Double]) -> Double {
        let (hiddenLayerInput, hiddenLayerOutput, outputLayerOutput) = forward(input: input)
        
        let loss = calculateLoss(outputLayerOutput, target)
        
        // backpropogation
        var outputError = Array(repeating: 0.0, count: outputSize)
        for i in 0..<outputSize {
            outputError[i] = outputLayerOutput[i] - target[i]
        }
        
        // hidden layer error
        var hiddenError = Array(repeating: 0.0, count: hiddenSize)
        for i in 0..<hiddenSize {
            var error = 0.0
            for j in 0..<outputSize {
                error += outputError[j] * weightsHiddenOutput[i][j]
            }
            hiddenError[i] = error * reluDerivative(hiddenLayerInput[i])
        }
        
        // update weights and biases
        for i in 0..<hiddenSize {
            for j in 0..<outputSize {
                weightsHiddenOutput[i][j] -= learningRate * outputError[j] * hiddenLayerOutput[i]
            }
        }
        
        for i in 0..<outputSize {
            biasesOutput[i] -= learningRate * outputError[i]
        }
        
        // hidden layer
        for i in 0..<inputSize {
            for j in 0..<hiddenSize {
                weightsInputHidden[i][j] -= learningRate * hiddenError[j] * input[i]
            }
        }
        
        for i in 0..<hiddenSize {
            biasesHidden[i] -= learningRate * hiddenError[i]
        }
        
        return loss
    }
    
    // predict class for input
    func predict(_ input: [Double]) -> Int {
        let (_, _, output) = forward(input: input)
        
        // find index of highest probability
        return output.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? -1
    }
    
    // train on multiple samples
    mutating func train(inputs: [[Double]], targets: [[Double]], epochs: Int, batchSize: Int = 32) {
        assert(inputs.count == targets.count, "Number of inputs and targets must be equal")
        
        let numSamples = inputs.count
        print("Entering training")
        
        for epoch in 0..<epochs {
            var totalLoss: Double = 0.0
            var correctPredictions = 0
            
            // create shuffled indexes for the bunch
            let indices = Array(0..<numSamples).shuffled()
            
            for i in stride(from: 0, to: numSamples, by: batchSize) {
                let endIdx = min(i + batchSize, numSamples)
                let batchIndices = Array(indices[i..<endIdx])
                
                for idx in batchIndices {
                    let input = inputs[idx]
                    let target = targets[idx]
                    
                    let loss = trainSample(input, target)
                    totalLoss += loss
                    
                    // check if correct
                    let prediction = predict(input)
                    let targetClass = target.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? -1
                    
                    if prediction == targetClass {
                        correctPredictions += 1
                    }
                }
            }
            
            let accuracy = Double(correctPredictions) / Double(numSamples)
            let avgLoss = totalLoss / Double(numSamples)
            
            if (epoch + 1) % 10 == 0 {
                print("Epoch \(epoch + 1): Loss = \(avgLoss), Accuracy = \(accuracy)")
            }
            
        }
    }
}

// Using the do block for error handling
do {

    let trainImagesPath = "/Users/liam/Development/MetalMLP/MetalMLP/Data/train-images-idx3-ubyte"
    let trainImages = try MNISTFashionLoader.loadImages(from: trainImagesPath)
    print("Training images: \(trainImages.count)")
    
    let trainLabelsPath = "/Users/liam/Development/MetalMLP/MetalMLP/Data/train-labels-idx1-ubyte"
    let trainLabels = try MNISTFashionLoader.loadLabels(from: trainLabelsPath)
    print("Training labels: \(trainLabels.count)")
       
    
    // Step 2: Figure out how to represent and work with a single image
//    
//    let outputImagePath = "/Users/liam/Development/MetalMLP/MetalMLP/SampleData/output.pgm"
//    let fileURL = URL(fileURLWithPath: outputImagePath)
//    
//    let header = "P5\n\(28) \(28)\n255\n"
//    let headerData = Data(header.utf8)
//    let pixelData = Data(trainImages[0])
//    let combinedData = headerData + pixelData
//    
//    do {
//        try combinedData.write(to: fileURL)
//    } catch {
//        print("Error writing to file: \(error)")
//    }
//    print(trainImages[0])
    
    
    // Step 3: Figure out how to properly clean up and represent all the data
    // (this step is already done, not everything needs to be visualized)
    
    var mlpInputs: [[Double]] = []
    var mlpTargets: [[Double]] = []
    
//    let numToProcess = min(10000, trainImages.count)
    let numToProcess = min(10000, trainImages.count)
    
    
    for i in 0..<numToProcess {
        // normalize pixel values to [0,1]
        let normalizedPixels = trainImages[i].map { Double($0) / 255.0}
        mlpInputs.append(normalizedPixels)
        
        // one-hot encoding for the lables
        var target = Array(repeating: 0.0, count: 10)
        target[Int(trainLabels[i])] = 1.0
        mlpTargets.append(target)
    }
    
    
    // Step 4: Figure out how to do a single iteration of the training process manually.
    
    // create and train the mlp
    var mlp = MLP(inputSize: 784, hiddenSize: 128, outputSize: 10, learningRate: 0.01)
    
    print("Starting training...")
    let startTime = Date()
    mlp.train(inputs: mlpInputs, targets: mlpTargets, epochs: 20, batchSize: 128)
    let duration = Date().timeIntervalSince(startTime)
    print("Training completed in \(duration) seconds")
    
    // tests on a few images
    for i in 0..<5 {
        let prediction = mlp.predict(mlpInputs[i])
        let actualLabel = trainLabels[i]
        print("Image \(i): Prediction = \(prediction), Actual = \(actualLabel)")
    }
    
    
    
//    let arr4: [[UInt8]] = []
//    let arr5: [UInt8] = []
//    let x = MLP(input: arr4, hidden: arr5)
//    x.sayHi()
    
//    let mlp = MLP(hiddenSize: 128)
//    print("\nMLP initialized:")
//    print("Input->Hidden weights shape: \(mlp.weightsInputHidden.count) x \(mlp.weightsInputHidden[0].count)")
//    print("Hidden bias shape: \(mlp.biasHidden.count)")
//    print("Hidden->Output weights shape: \(mlp.weightsHiddenOutput.count) x \(mlp.weightsHiddenOutput[0].count)")
//    print("Output bias shape: \(mlp.biasOutput.count)")
//    
//    let normalizedFirstImage = MLP.normalizeInput(trainImages[0])
//    print("\nFirst pixel values (original): \(Array(trainImages[0][0..<5]))")
//    print("First pixel values (normalized): \(Array(normalizedFirstImage[0..<5]))")
    
    // training the model
    
    // input layer is already prepared
    
    
    // set up input layer, hidden layer, output layer
    
    /* MLP architecture:
     * Input layer has 28x28 = 784 nodes (each a pixel)
     * Hidden layer has however many nodes (say 100 for now)
     * Output layer has 10 nodes (each associated with an output number 0-9)
     */
    
    // training such an MLP on the CPU should take around 41 seconds
    
    // Step 5: Figure out how to accelerate the training process with the GPU

    
} catch {
    print("Error loading data: \(error)")
}

