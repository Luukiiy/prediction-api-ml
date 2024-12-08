const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, imageBuffer) {
  try {
    const tensor = tf.node
      .decodeJpeg(imageBuffer)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat()

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const classes = ['Cancer', 'Non-cancer'];

    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    const label = classes[classResult];

    let explanation, suggestion;

    if (label === 'Cancer') {
      suggestion = "Segera periksa ke dokter!"
    }
  
    if (label === 'Non-cancer') {
      suggestion = "Penyakit kanker tidak terdeteksi."
    }

    return { confidenceScore, label, explanation, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

module.exports = predictClassification;