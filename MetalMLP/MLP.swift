//
//  MLP.swift
//  MetalMLP
//
//  Created by Liam D on 6/1/25.
//
import Foundation

/* MLP architecture:
 * Input layer has 28x28 = 784 nodes (each a pixel)
 * Hidden layer has however many nodes (say 100 for now)
 * Output layer has 10 nodes (each associated with an output number 0-9)
 */

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
