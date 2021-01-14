import * as tf from '@tensorflow/tfjs-node';

export default async function run() {
  const model = await tf.loadLayersModel('file://./lemon/model.json');
  (model.predict(tf.tensor([20])) as tf.Tensor).print();
}
