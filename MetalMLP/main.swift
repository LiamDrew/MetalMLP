//
//  main.swift
//  MetalMLP
//
//  Created by Liam D on 5/31/25.
//

import Foundation



// Step 1: Import all the MNIST fashion data

struct MNISTFashionLoader {
    static func loadImages(from path: String) throws -> [[UInt8]] {
        
        // TODO: import data from path. This makes no sense
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

//
//var trainImages: [[UInt8]]
//var trainLabels: [UInt8]

// Using the do block for error handling
do {
    // NOTE: hardcoding the path like this is absolutely fucked. I need to figure out the idiomatic way to do this with Swift/Xcode
    let trainImagesPath = "/Users/liam/Development/MetalMLP/MetalMLP/Data/train-images-idx3-ubyte"
    let trainImages = try MNISTFashionLoader.loadImages(from: trainImagesPath)
    print("Training images: \(trainImages.count)")
    
    let trainLabelsPath = "/Users/liam/Development/MetalMLP/MetalMLP/Data/train-labels-idx1-ubyte"
    let trainLabels = try MNISTFashionLoader.loadLabels(from: trainLabelsPath)
    print("Training labels: \(trainLabels.count)")
    
    // Step 2: Figure out how to properly clean up and represent all the data
    
    let outputImagePath = "/Users/liam/Development/MetalMLP/MetalMLP/SampleData/output.png"
    let fileURL = URL(fileURLWithPath: outputImagePath)
    
    do {
        try trainImages[0].write(to: fileURL, atomically: true)
    } catch {
        print("Error writing to file: \(error)")
    }
    print(trainImages[0])
    
    
    
} catch {
    print("Error loading data: \(error)")
    print("Something got fucked up here")
}

//print("Training labels: \(trainLabels.count)")


// Step 3: Figure out how to do the training process normally

// Step 4: Figure out how to accelerate the training process with the GPU

//


