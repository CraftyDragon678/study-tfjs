import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';

export default async function run() {
  const model = await mobilenet.load();
  const image = tf.node.decodeJpeg(fs.readFileSync(path.join(__dirname, './dog.jpg')));
  const predictions = await model.classify(image);
  console.log(predictions);
};
