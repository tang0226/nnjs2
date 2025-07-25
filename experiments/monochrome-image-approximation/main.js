/**

Rendering / training pipeline:

When the training session is triggered, the main script sends hyperparameters
to the master worker, which 


main script handles creation / termination of workers, as well as
sending hyperparameter updates to the workers.


Render worker(s) (responsible for a chunk of the rendered image)
Learning worker(s)


Tasks:
1. Run some number of batches, noting the parameter deltas
2. Apply the parameter deltas and draw the image (using requestAnimationFrame with putImageData)
3. repeat


Batching options:
1. Full-batch (run through every point in the training dataset, then end the batch and apply the average derivative)
2. Stochastic  / mini-batch: more frequent updates using random sampling and smaller batch sizes

*/

function rgbToGrayscale(r, g, b) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

// Transforms a 4wh Uint8ClampedArray from an image data object
// to a wh array of grayscale values between 0 and 255.
function imgDataToGrayscale(data) {
  let res = [];
  let l = data.length;
  for (let i = 0; i < l; i += 4) {
    res.push(rgbToGrayscale(data[i], data[i + 1], data[i + 2]));
  }
  return res;
}

// Normalize color values to the range [0, 1]
// This will be used to encode data for the neural network
function normalizeColorValues(vals) {
  return vals.map((x) => x / 255);
}


function enableInput(ele)  {
  ele.removeAttribute("disabled");
}

function disableInput(ele) {
  ele.setAttribute("disabled", "true");
}


const mainCanvas = document.getElementById("main-canvas");
const ctx = mainCanvas.getContext("2d");

const targetCanvas = document.getElementById("target-canvas");
const targetCtx = targetCanvas.getContext("2d");

const imageInput = document.getElementById("image-input");
const canvasSizeInput = document.getElementById("canvas-size");

const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const resetNetworkButton = document.getElementById("reset-network-button");

const trainingWorkersInput = document.getElementById("training-workers");
const renderWorkersInput = document.getElementById("render-workers");

const activationFunctionInput = document.getElementById("activation-function");
const hiddenLayersInput = document.getElementById("hidden-layers");
const learningRateInput = document.getElementById("learning-rate");
const batchSizeInput = document.getElementById("batch-size-input");
const bpfInput = document.getElementById("bpf-input");

// Hyperparameters that cannot be changed during training
const coreInputs = [
  trainingWorkersInput,
  renderWorkersInput,
  activationFunctionInput,
  hiddenLayersInput
];


canvasSizeInput.value = "200";

renderWorkersInput.value = 4;
trainingWorkersInput.value = 4;

activationFunctionInput.value = "leaky-relu";
hiddenLayersInput.value = "150, 50";
learningRateInput.value = "0.01";
batchSizeInput.value = "1024";
bpfInput.value = "1";

var width, height;

function setCanvasDim(w, h) {
  mainCanvas.width = w;
  mainCanvas.height = h;
  mainCanvas.style.width = w + "px";
  mainCanvas.style.height = h + "px";
  targetCanvas.width = w;
  targetCanvas.height = h;
  targetCanvas.style.width = w + "px";
  targetCanvas.style.height = h + "px";
  width = w;
  height = h;
}

var canvasSize = Number(canvasSizeInput.value);
setCanvasDim(canvasSize, canvasSize);

// Stores the size of the raw input image

// Set a junk default value; code will validate use of these variables
// by checking that targetImg is set
var imgFileWidth = imgFileHeight = 100;
var targetImg, targetImgData, targetNNOutput;

// Update canvas dimensions based on the target canvas size and the AR of the image file
function updateCanvasDim() {
  let ar = imgFileWidth / imgFileHeight;
  if (ar > 1) {
    setCanvasDim(canvasSize, Math.round(canvasSize / ar));
  }
  else {
    setCanvasDim(Math.round(canvasSize * ar), canvasSize);
  }
}

// Draws the target image to the target canvas and reads the image data
// so it can be used with the network.
function updateTargetImg() {
    targetCtx.drawImage(targetImg, 0, 0, width, height);
    targetImgData = targetCtx.getImageData(0, 0, width, height).data;
    targetNNOutput = normalizeColorValues(imgDataToGrayscale(targetImgData));

    // Send the data to the workers
    messageAllWorkers({
      type: "targetData",
      targetData: targetNNOutput,
    });
}


function formatHiddenLayers(layers) {
  return layers.join(", ");
}

function parseHiddenLayersString(str) {
  let nums = str.split(",").map((x) => Number(x.trim()));
  for (let n of nums) {
    if (Number.isNaN(n) || !Number.isInteger(n)) {
      return false;
    }
  }
  return nums;
}


function updateActivationFunction() {
  let val = activationFunctionInput.value;
  switch(val) {
    case "relu":
      activationFunction = NN.RELU;
      break;
    case "leaky-relu":
      activationFunction = NN.LEAKY_RELU(0.3);
      break;
    case "sigmoid":
      activationFunction = NN.SIGMOID;
      break;
    case "tanh":
      activationFunction = NN.TANH;
      break;
  }
}


var activationFunction;
updateActivationFunction();

var hiddenLayers = parseHiddenLayersString(hiddenLayersInput.value),
  batchSize = Number(batchSizeInput.value),
  isTraining = true,
  batchInProgress = false,
  isRendering = false,
  learningRate = Number(learningRateInput.value),
  batch = 0,
  batchesPerFrame = Number(bpfInput.value),
  trainingWorkers = [],
  trainingWorkerCount = 0,
  trainingParameterTotals = {},
  trainingWorkersDone = 0,
  renderWorkers = [],
  renderWorkerCount = 0,
  renderChunks = [],
  renderWorkersDone = 0;

var nn;
function initNetwork(hiddenLayers, af) {
  nn = new NN({
    layerSizes: [2].concat(hiddenLayers).concat([1]),
    activationFunctions: [af, NN.SIGMOID],
    wInit: {
      method: NN.RANDOM,
      range: 0.1,
    },
    bInit: {
      method: NN.RANDOM,
      range: 0.1,
    },
  });
}

function createRenderWorker() {
  let worker = new Worker("render.js");
  // When the worker is done, draw its data to the correct
  // position on the canvas
  worker.onmessage = function(event) {
    let data = event.data;
    if (data.type == "done") {
      renderWorkersDone++;
      renderChunks[data.chunkI] = data.imgDataArr;

      // If the render is done, draw the image and finish the render
      if (renderWorkersDone == renderWorkerCount) {
        // only draw if the canvas dimensions were not changed during the render
        if (width == data.width && height == data.height) {
          // Combine all render chunks into one data array
          let cumImgDataArr = new Uint8ClampedArray(renderChunks.reduce((acc, curr) => [...acc, ...curr], []));
          ctx.clearRect(0, 0, width, height);
          ctx.putImageData(new ImageData(cumImgDataArr, width, height), 0, 0);
        }
        isRendering = false;
      }
    }
  };

  // Send the current network to the worker so it can render when prompted
  worker.postMessage({
    type: "nn",
    nn: nn.serialize(),
  });

  return worker;
}

function createTrainingWorker() {
  let worker = new Worker("training.js");

  worker.onmessage = function(event) {
    let data = event.data;
    if (data.type == "done") {
      trainingWorkersDone++;

      // Check if the parameter totals exist yet; if so, add to them
      if (Object.keys(trainingParameterTotals).length) {
        trainingParameterTotals = {
          w: NN.add3d(trainingParameterTotals.w, data.gradientTotals.w),
          b: NN.add2d(trainingParameterTotals.b, data.gradientTotals.b),
        };
      }
      else {
        // Otherwise, initialize them
        trainingParameterTotals = {
          w: data.gradientTotals.w,
          b: data.gradientTotals.b,
        };
      }

      // If the batch is done, apply changes
      if (trainingWorkersDone == trainingWorkerCount) {
        // Apply the training parameter totals, averaged based on LR and batch size
        nn.applyChanges(trainingParameterTotals, -learningRate / data.batchSize);

        // Send new parameters to all workers:
        messageAllWorkers({
          type: "weights",
          w: nn.w,
        });
        messageAllWorkers({
          type: "biases",
          b: nn.b,
        });

        // Attempt to draw the frame;
        // Will only message render workers if a render is not already in progress (which will happen for small batch sizes)
        draw();

        batchInProgress = false;
        console.log("batch done!");
        if (isTraining) {
          setTimeout(train, 0);
        }
      }

    }
  }

  // Send the current network to the worker so it can train when prompted
  worker.postMessage({
    type: "nn",
    nn: nn.serialize(),
  });

  return worker;
}

function initRenderWorkers(n) {
  renderWorkers = [];
  renderWorkerCount = n;
  for (let i = 0; i < n; i++) {
    renderWorkers.push(createRenderWorker());
  }
}

function initTrainingWorkers(n) {
  trainingWorkers = [];
  trainingWorkerCount = n;
  for (let i = 0; i < n; i++) {
    trainingWorkers.push(createTrainingWorker());
  }
}

function messageAllWorkers(msg) {
  for (let w of renderWorkers.concat(trainingWorkers)) {
    w.postMessage(msg);
  }
}

function draw() {
  if (!targetImgData) {
    return false;
  }
  if (!isRendering) {
    isRendering = true;

    renderStartTime = performance.now();

    renderWorkersDone = 0;
    renderChunks = new Array(renderWorkerCount);

    let baseChunkHeight = Math.floor(height / renderWorkerCount);
    let yStart = 0;
    for (let i = 0; i < renderWorkerCount; i++) {
      let worker = renderWorkers[i];
      let chunkHeight = baseChunkHeight + Number(height % baseChunkHeight > i);
      
      worker.postMessage({
        type: "render",
        yStart: yStart,
        chunkHeight: chunkHeight,
        chunkI: i,
        width: width,
        height: height
      });

      yStart += chunkHeight;
    }
  }

}

// Perform one batch of training
function train() {
  if (!targetImgData) {
    return false;
  }
  if (!batchInProgress) {
    trainingWorkersDone = 0;
    batchInProgress = true;

    let baseTrialsPerWorker = Math.floor(batchSize / trainingWorkerCount);

    for (let i = 0; i < trainingWorkerCount; i++) {
      let worker = trainingWorkers[i];
      worker.postMessage({
        type: "train",
        batchSize: batchSize,
        trials: baseTrialsPerWorker + Number(batchSize % trainingWorkerCount > i),
        width: width,
        height: height,
      });
    }
  }
}

initNetwork(hiddenLayers, activationFunction);
initRenderWorkers(Number(renderWorkersInput.value));
initTrainingWorkers(Number(trainingWorkersInput.value));

//disableInput(stopButton);


imageInput.addEventListener("change", (event) => {
  var fr = new FileReader;

  fr.onload = function() {
    var img = new Image();
    img.onload = function() {
      imgFileWidth = img.width;
      imgFileHeight = img.height;
      updateCanvasDim();

      targetImg = img;
      updateTargetImg();
    };

    img.src = fr.result;
  }

  if (event.target.files.length) {
    fr.readAsDataURL(event.target.files[0]);
  }
  else {
    // Reset image variables
    targetImg = targetImgData = targetNNOutput = null;
    imgFileWidth = imgFileHeight = 100;
    targetCtx.clearRect(0, 0, width, height);
  }
});

canvasSizeInput.addEventListener("change", () => {
  let n = Number(canvasSizeInput.value);
  if (Number.isNaN(n) || !Number.isInteger(n) || n <= 0) {
    canvasSizeInput.value = canvasSize;
  }
  else {
    canvasSize = n;
    updateCanvasDim();
    if (targetImg) {
      updateTargetImg();
    }
  }
});

trainingWorkersInput.addEventListener("change", function() {
  let newCount = Number(trainingWorkersInput.value);
  let oldCount = trainingWorkerCount;
  let diff = newCount - oldCount;
  trainingWorkerCount = newCount;
  if (diff > 0) {
    // Add new workers
    for (let i = 0; i < diff; i++) {
      trainingWorkers.push(createRenderWorker());
    }
  }
  if (diff < 0) {
    // Delete workers
    for (let i = 0; i < Math.abs(diff); i++) {
      trainingWorkers[oldCount - i - 1].terminate();
    }
    trainingWorkers = trainingWorkers.slice(0, newCount);
  }
});


renderWorkersInput.addEventListener("change", function() {
  let newCount = Number(renderWorkersInput.value);
  let oldCount = renderWorkerCount;
  let diff = newCount - oldCount;
  renderWorkerCount = newCount;
  if (diff > 0) {
    // Add new workers
    for (let i = 0; i < diff; i++) {
      renderWorkers.push(createRenderWorker());
    }
  }
  if (diff < 0) {
    // Delete workers
    for (let i = 0; i < Math.abs(diff); i++) {
      renderWorkers[oldCount - i - 1].terminate();
    }
    renderWorkers = renderWorkers.slice(0, newCount);
  }
});

hiddenLayersInput.addEventListener("change", () => {
  let layers = parseHiddenLayersString(hiddenLayersInput.value);
  if (layers) {
    // update the layers and reset the agent
    hiddenLayers = layers;
    initNetwork(hiddenLayers, activationFunction);
    messageAllWorkers({
      type: "nn",
      nn: nn.serialize(),
    });
    batch = 0;
  }
  else {
    hiddenLayersInput.value = formatHiddenLayers(hiddenLayers);
  }
});

activationFunctionInput.addEventListener("change", () => {
  // Update the af and reset the agent
  updateActivationFunction();
  initNetwork(hiddenLayers, activationFunction);
  // Send the new network to all workers
  messageAllWorkers({
    type: "nn",
    nn: nn.serialize(),
  });
  batch = 0;
});

learningRateInput.addEventListener("change", () => {
  let n = Number(learningRateInput.value);
  if (Number.isNaN(n)) {
    learningRateInput.value = learningRate;
  }
  else {
    learningRate = n;
  }
});


batchSizeInput.addEventListener("change", () => {
  let n = Number(batchSizeInput.value);
  if (Number.isNaN(n) || !Number.isInteger(n) || n <= 0) {
    batchSizeInput.value = batchSize;
  }
  else {
    batchSize = n;
  }
});


bpfInput.addEventListener("change", () => {
  let n = Number(bpfInput.value);
  if (Number.isNaN(n) || !Number.isInteger(n) || n <= 0) {
    bpfInput.value = batchesPerFrame;
  }
  else {
    batchesPerFrame = n;
  }
});

startButton.addEventListener("click", () => {
  if (targetImgData) {
    isTraining = true;
    disableInput(startButton);
    enableInput(stopButton);
    coreInputs.forEach((input) => {disableInput(input)});

    train();
  }
  else {
    alert("Please select an image file");
  }
});

stopButton.addEventListener("click", () => {
  isTraining = false;
  disableInput(stopButton);
  enableInput(startButton);
  coreInputs.forEach((input) => {enableInput(input)});
});

resetNetworkButton.addEventListener("click", () => {
  initNetwork(hiddenLayers, activationFunction);
  messageAllWorkers({
    type: "nn",
    nn: nn.serialize(),
  });
});

var keys = {};

window.addEventListener("keydown", (e) => {
  keys[e.key] = true;
});

window.addEventListener("keyup", (e) => {
  keys[e.key] = false;
});
