module.exports = function (RED) {
  /* Initial Setup */
  const { Readable } = require('stream')
  const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args))
  //const tf = require('@tensorflow/tfjs')
  const speechCommands = require('@tensorflow-models/speech-commands')
  const tf = require('@tensorflow/tfjs-node')
  const { spawn } = require('child_process');
  const child = spawn('python', ['AudioDeamon.py'], {cwd: "AudioDetectionDaemon"});

  function teachableMachine(config) {
    /* Node-RED Node Code Creation */
    RED.nodes.createNode(this, config)
    const node = this

    const nodeStatus = {
      MODEL: {
        LOADING: { fill: 'yellow', shape: 'ring', text: 'loading...' },
        RELOADING: { fill: 'yellow', shape: 'ring', text: 'reloading...' },
        READY: { fill: 'green', shape: 'dot', text: 'ready' },
        DECODING: { fill: 'green', shape: 'ring', text: 'decoding...' },
        PREPROCESSING: { fill: 'green', shape: 'ring', text: 'preprocessing...' },
        INFERENCING: { fill: 'green', shape: 'ring', text: 'inferencing...' },
        POSTPROCESSING: { fill: 'green', shape: 'ring', text: 'postprocessing...' },
        RESULT: (text) => { return { fill: 'green', shape: 'dot', text } }
      },
      ERROR: (text) => { node.error(text); return { fill: 'red', shape: 'dot', text } },
      CLOSE: {}
    }

    class ModelManager {
      constructor() {
        this.ready = false
        this.labels = []
      }

      async load(url) {
        if (this.ready) {
          node.status(nodeStatus.MODEL.RELOADING)
        } else {
          node.status(nodeStatus.MODEL.LOADING)
        }

        this.model = await this.getModel(url)
        this.labels = this.getLabels(this.model)


        this.ready = true
        return this.model
      }

      async getModel(url) {
        throw new Error('getModel(url) needs to be implemented')
      }

      async getLabels(url) {
        throw new Error('getLabels(url) needs to be implemented')
      }
    }

    class OnlineModelManager extends ModelManager {
      async getModel(url) {
        const modelURL = url + 'model.json'
        const response = await fetch(url + 'metadata.json')
        const body = await response.text();
        const json = JSON.parse(body);
        const speechModel = speechCommands.create(
          "BROWSER_FFT", // fourier transform type, not useful to change
          undefined, // speech commands vocabulary feature, not useful for your models
          modelURL,
          json);
        node.warn(JSON.stringify(speechModel.params()))
        await (speechModel).ensureModelLoaded();
        node.classLabels = speechModel.wordLabels();
        return speechModel
      }

      async getLabels(model) {
        return model.wordLabels();
      }
    }

    const modelManagerFactory = {
      online: new OnlineModelManager()
    }

    function nodeInit() {
      node.modelManager = modelManagerFactory[config.mode]
      if (config.modelUrl !== '') {
        loadModel(config.modelUrl)
      }
    }

    /**
     * Loads the Model trained from an Teachable Machine.
     * @param url where to load the model from
     */
    async function loadModel(url) {
      try {
        node.model = await node.modelManager.load(url)
        node.status(nodeStatus.MODEL.READY)
      } catch (error) {
        node.status(nodeStatus.ERROR(error))
      }
    }


    /**
     * Infers an audio buffer to obtain classification predictions.
     * @param audioBuffer audio buffer in wav format
     * @returns outputs of the model
     */
    async function inferAudioBuffer(audioBuffer) {
      const audioTensor = tf.tensor4d(
        audioBuffer, [1].concat((node.model).modelInputShape().slice(1)));
      const testAudioData = tf.randomUniform(
        shape = [1, 43, 232, 1],
        minval = -1,
        maxval = 1
      )
      node.status(nodeStatus.MODEL.INFERENCING)
      return await (node.model).recognize(audioTensor)
    }

    /**
     * Computes the probabilities of the topK classes given logits by computing
     * softmax to get probabilities and then sorting the probabilities.
     * @param logits Tensor representing the logits from MobileNet.
     * @param topK The number of top predictions to show.
     */
    async function getTopKClasses(logits, topK) {
      const values = await logits.scores
      topK = Math.min(topK, values.length)

      const valuesAndIndices = []
      for (let i = 0; i < values.length; i++) {
        valuesAndIndices.push({ value: values[i], index: i })
      }
      valuesAndIndices.sort((a, b) => {
        return b.value - a.value
      })
      const topkValues = new Float32Array(topK)
      const topkIndices = new Int32Array(topK)
      for (let i = 0; i < topK; i++) {
        topkValues[i] = valuesAndIndices[i].value
        topkIndices[i] = valuesAndIndices[i].index
      }
      const topClassesAndProbs = []
      const promiseLabels = await node.modelManager.labels
      for (let i = 0; i < topkIndices.length; i++) {
        topClassesAndProbs.push({
          class: promiseLabels[topkIndices[i]],
          score: topkValues[i]
        })
      }
      return topClassesAndProbs
    }

    /**
     * Post processes the outputs depending on the node configuration.
     * @param outputs
     * @returns a list of predictions
     */
    async function postprocess(outputs) {
      const tempLabels = await node.modelManager.labels
      const labelsLength = tempLabels.length
      const predictions = await getTopKClasses(outputs, labelsLength)

      const bestProbability = predictions[0].score.toFixed(2) * 100
      const bestPredictionText = bestProbability.toString() + '% - ' + predictions[0].class

      if (config.output === 'best') {
        node.status(nodeStatus.MODEL.RESULT(bestPredictionText))
        return [predictions[0]]
      } else if (config.output === 'all') {
        let filteredPredictions = predictions
        filteredPredictions = config.activeThreshold ? filteredPredictions.filter(prediction => prediction.score > config.threshold / 100) : filteredPredictions
        filteredPredictions = config.activeMaxResults ? filteredPredictions.slice(0, config.maxResults) : filteredPredictions

        if (filteredPredictions.length > 0) {
          node.status(nodeStatus.MODEL.RESULT(bestPredictionText))
        } else {
          const statusText = 'score < ' + config.threshold + '%'
          node.status(nodeStatus.MODEL.RESULT(statusText))
          return []
        }
        return filteredPredictions
      }
    }

    /* Main Node Logic */

    nodeInit()

    child.stdout.on('data', data => {
      node.warn("Connected to server!");
      socket.on('data', async function (data) {
        const floatArray = new Float32Array(data.buffer);
        let msg = {};
        if (floatArray.length <= 2) {
          msg.otherValue = floatArray[0];
          msg.dirOfArrival = floatArray[1];
        }
        else {
          msg.otherValue = floatArray[0];
          msg.dirOfArrival = floatArray[1];
          const outputs = await inferAudioBuffer(floatArray.slice(2));
          if (outputs === null) { node.status(nodeStatus.MODEL.READY); return }
          msg.payload = await postprocess(outputs)
          msg.classes = node.classLabels

        }
        node.send(msg)
      })

    });

    child.stderr.on('data', data => {
      console.error(`stderr: ${data}`);
    });

    node.on('close', function () {
      node.status(nodeStatus.CLOSE)
      child.kill();
    })


  }
  RED.nodes.registerType('teachable machine', teachableMachine)
}
