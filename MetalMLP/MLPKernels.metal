////
////  MLPKernels.metal
////  MetalMLP
////
////  Created on 6/1/25.
////
//
//#include <metal_stdlib>
//using namespace metal;
//
//// Matrix-vector multiplication for input to hidden layer
//// Computes: hiddenLayerInput[j] = sum(input[i] * weights[i][j]) + bias[j]
//kernel void input_to_hidden_forward(constant float *input [[buffer(0)]],
//                                   constant float *weights [[buffer(1)]],
//                                   constant float *biases [[buffer(2)]],
//                                   device float *output [[buffer(3)]],
//                                   constant uint &inputSize [[buffer(4)]],
//                                   uint gid [[thread_position_in_grid]]) {
//    float sum = biases[gid];
//    
//    for (uint i = 0; i < inputSize; i++) {
//        // weights are stored in row-major order: weights[i * hiddenSize + j]
//        sum += input[i] * weights[i * gridSize + gid];
//    }
//    
//    output[gid] = sum;
//}
//
//// ReLU activation function
//kernel void relu_activation(device float *input [[buffer(0)]],
//                           device float *output [[buffer(1)]],
//                           uint gid [[thread_position_in_grid]]) {
//    output[gid] = max(0.0f, input[gid]);
//}
//
//// Matrix-vector multiplication for hidden to output layer
//kernel void hidden_to_output_forward(constant float *hidden [[buffer(0)]],
//                                    constant float *weights [[buffer(1)]],
//                                    constant float *biases [[buffer(2)]],
//                                    device float *output [[buffer(3)]],
//                                    constant uint &hiddenSize [[buffer(4)]],
//                                    uint gid [[thread_position_in_grid]]) {
//    float sum = biases[gid];
//    
//    for (uint i = 0; i < hiddenSize; i++) {
//        // weights are stored in row-major order: weights[i * outputSize + j]
//        sum += hidden[i] * weights[i * gridSize + gid];
//    }
//    
//    output[gid] = sum;
//}
//
//// Softmax - Step 1: Find max value (reduction)
//kernel void softmax_max(constant float *input [[buffer(0)]],
//                       device atomic_float *maxValue [[buffer(1)]],
//                       constant uint &size [[buffer(2)]],
//                       uint gid [[thread_position_in_grid]]) {
//    if (gid < size) {
//        atomic_fetch_max_explicit(maxValue, input[gid], memory_order_relaxed);
//    }
//}
//
//// Softmax - Step 2: Compute exp(x - max) and sum
//kernel void softmax_exp_sum(constant float *input [[buffer(0)]],
//                           device float *expValues [[buffer(1)]],
//                           device atomic_float *sum [[buffer(2)]],
//                           constant float &maxValue [[buffer(3)]],
//                           uint gid [[thread_position_in_grid]]) {
//    float expVal = exp(input[gid] - maxValue);
//    expValues[gid] = expVal;
//    atomic_fetch_add_explicit(sum, expVal, memory_order_relaxed);
//}
//
//// Softmax - Step 3: Normalize
//kernel void softmax_normalize(constant float *expValues [[buffer(0)]],
//                             device float *output [[buffer(1)]],
//                             constant float &sum [[buffer(2)]],
//                             uint gid [[thread_position_in_grid]]) {
//    output[gid] = expValues[gid] / sum;
//}
