const tf = require('@tensorflow/tfjs');
async function loadModel() {
    return tf.loadGraphModel ("https://storage.googleapis.com/prediction-cancer_bucket-1/model.json");
}
module.exports = loadModel;