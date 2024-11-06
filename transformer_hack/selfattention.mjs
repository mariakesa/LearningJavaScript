import * as tf from '@tensorflow/tfjs';

class SelfAttention {
    constructor(C) {
        // Initialize weight matrices Q, K, V without biases
        this.Q_linear = tf.layers.dense({ units: C, useBias: false });
        this.K_linear = tf.layers.dense({ units: C, useBias: false });
        this.V_linear = tf.layers.dense({ units: C, useBias: false });
    }

    forward(x) {
        // Compute the Q, K, V projections
        const Q = this.Q_linear.apply(x);
        const K = this.K_linear.apply(x);
        const V = this.V_linear.apply(x);

        // Calculate attention weights
        const K_transposed = tf.transpose(K, [0, 2, 1]);  // Transpose K along last two dimensions
        const scores = tf.matMul(Q, K_transposed);         // Batch matrix multiplication Q * K^T
        const weights = tf.softmax(scores, -1);            // Apply softmax along the last axis

        // Compute attention output
        const output = tf.matMul(weights, V);              // Weighted sum of V
        output.print();  // This is similar to print in PyTorch

        return output;
    }
}

// Example usage:
// Assuming C is the dimension of the embedding and B is the batch size
const C = 32;  // Set your dimension here
const attention = new SelfAttention(C);

// Dummy input tensor, assuming batch size B=2 and sequence length T=10
const B = 4;
const T = 8;
const x = tf.randomNormal([B, T, C]);

// Run the forward pass
attention.forward(x);
