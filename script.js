const STATUS = document.getElementById('statusmodel');
const STATUS_COUNT = document.getElementById('statuscount');
const STATUS_COUNT2 = document.getElementById('statuscount2');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];
const NAME_CLASS1 = document.getElementById('class1name');
const NAME_CLASS2 = document.getElementById('class2name');
const R_CLASS1 = document.getElementById('class1p');
const R_CLASS2 = document.getElementById('class2p');
const BAR1 = document.getElementById('myBar1');
const BAR2 = document.getElementById('myBar2');

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
//RESET_BUTTON.addEventListener('click', reset);

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
    if (hasGetUserMedia()) {
      // getUsermedia parameters.
      const constraints = {
        video: true,
        width: 640, 
        height: 480 
      };
  
      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        VIDEO.srcObject = stream;
        VIDEO.addEventListener('loadeddata', function() {
          videoPlaying = true;
          ENABLE_CAM_BUTTON.classList.add('removed');
        });
      });
    } else {
      console.warn('getUserMedia() is not supported by your browser');
    }
}


async function trainAndPredict() {
    predict = false;
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);

    let results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 10, 
        callbacks: {onEpochEnd: logProgress} });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    predict = true;
    predictLoop();
    }

    function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
}


let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}


function gatherDataForClass() {
  // TODO: Fill this out later in the codelab!
}

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

async function loadMobileNetFeatureModel() {
    const URL = 
      'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
    
    mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
    STATUS.innerText = 'MobileNet v3 loaded successfully!';
    
    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
      let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
      console.log(answer.shape);
    });
  }
  
  // Call the function immediately to start loading.
  loadMobileNetFeatureModel();

let model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));

model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  // Adam changes the learning rate over time which is useful.
  optimizer: 'adam',
  // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  // Else categoricalCrossentropy is used if more than 2 classes.
  loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy', 
  // As this is a classification problem you can record accuracy in the logs too!
  metrics: ['accuracy']  
});

/**
 * Handle Data Gather for button mouseup/mousedown.
 **/
function gatherDataForClass() {
    let classNumber = parseInt(this.getAttribute('data-1hot'));
    gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
    dataGatherLoop();
}

function dataGatherLoop() {
    if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
      let imageFeatures = tf.tidy(function() {
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, 
            MOBILE_NET_INPUT_WIDTH], true);
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
      });
  
      trainingDataInputs.push(imageFeatures);
      trainingDataOutputs.push(gatherDataState);
      
      // Intialize array index element if currently undefined.
      if (examplesCount[gatherDataState] === undefined) {
        examplesCount[gatherDataState] = 0;
      }
      examplesCount[gatherDataState]++;
  
      STATUS_COUNT.innerText = '';
      STATUS_COUNT2.innerText = '';
      STATUS_COUNT.innerText = examplesCount[0];
      STATUS_COUNT2.innerText = examplesCount[1];
    //   for (let n = 0; n < CLASS_NAMES.length; n++) {
    //     STATUS_COUNT.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
    //   }
      window.requestAnimationFrame(dataGatherLoop);
    }
}

function predictLoop() {
    if (predict) {
      tf.tidy(function() {
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor,[MOBILE_NET_INPUT_HEIGHT, 
            MOBILE_NET_INPUT_WIDTH], true);
  
        let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
        let prediction = model.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();
  
        //STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
        const bars1 = Math.floor(predictionArray[0] * 100);
        const bars2 = Math.floor(predictionArray[1] * 100);
        BAR1.style.width = bars1 + "%";
        BAR1.innerText= bars1 + '%';
        BAR2.style.width = bars2 + "%";
        BAR2.innerText= bars2 + '%';
      });
  
      window.requestAnimationFrame(predictLoop);
    }
}

function reset() {
    predict = false;
    examplesCount.length = 0;
    for (let i = 0; i < trainingDataInputs.length; i++) {
      trainingDataInputs[i].dispose();
    }
    trainingDataInputs.length = 0;
    trainingDataOutputs.length = 0;
    STATUS.innerText = 'No data collected';
    
    console.log('Tensors in memory: ' + tf.memory().numTensors);
  }

  function ubh_class1() {
    R_CLASS1.innerText = NAME_CLASS1;
    
  }