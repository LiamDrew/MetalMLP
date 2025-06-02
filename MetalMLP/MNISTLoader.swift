//
//  MNISTLoader.swift
//  MetalMLP
//
//  Created by Liam D on 6/1/25.
//

import Foundation

struct MNISTLoader {
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

