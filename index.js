import * as tf from '@tensorflow/tfjs';

// // 2x3 Tensor
// const shape = [2, 3]; // 2 rows, 3 columns
// const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
// a.print(); // print Tensor values
// // Output: [[1 , 2 , 3 ],
// //          [10, 20, 30]]
//
// // The shape can also be inferred:
// const b = tf.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
// b.print();
// // Output: [[1 , 2 , 3 ],
// //          [10, 20, 30]]
//
//
// const c = tf.tensor2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
// c.print();
//
// // 3x5 Tensor with all values set to 0
// const zeros = tf.zeros([3, 5]);
//
//
// zeros.print()
// // Output: [[0, 0, 0, 0, 0],
// //          [0, 0, 0, 0, 0],
// //          [0, 0, 0, 0, 0]]
//
//
// const initialValues = tf.zeros([5]);
// const biases = tf.variable(initialValues); // initialize biases
// biases.print(); // output: [0, 0, 0, 0, 0]
//
// const updatedValues = tf.tensor1d([0, 1, 0, 1, 0]);
// biases.assign(updatedValues); // update values of biases
// biases.print(); // output: [0, 1, 0, 1, 0]
//
// console.log('......')
// const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
// const d_squared = d.square();
// d_squared.print();
// // Output: [[1, 4 ],
// //          [9, 16]]
//
//
// const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
// const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);
//
// const e_plus_f = e.add(f);
// e_plus_f.print();
// // Output: [[6 , 8 ],
// //          [10, 12]]


async function learnLinear(){
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

    await model.fit(xs, ys, {epochs: 500});

    var a =
        model.predict(tf.tensor2d([10], [1, 1]));
    console.log(a)
}
learnLinear();