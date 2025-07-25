/**

Rendering / training pipeline:

When the training session is triggered, the main script sends hyperparameters
to the master worker, which 


main script handles creation / termination of workers, as well as
sending hyperparameter updates to the workers.


Render worker(s) (responsible for a chunk of the rendered image)
Learning worker(s)


Tasks:
1. Run some number of iterations, noting the parameter deltas
2. Apply the parameter deltas and draw the image (using requestAnimationFrame with putImageData)
3. repeat


Batching options:
1. Full-batch (run through every point in the training dataset, then end the iteration and apply the average derivative)
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
const ipfInput = document.getElementById("ipf-input");

// Hyperparameters that cannot be changed during training
const coreInputs = [
  trainingWorkersInput,
  renderWorkersInput,
  activationFunctionInput,
  hiddenLayersInput
];


canvasSizeInput.value = "200";

hiddenLayersInput.value = "40, 20";
learningRateInput.value = "0.2";
batchSizeInput.value = "64";
ipfInput.value = "1";

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
    setCanvasDim(Math.round(canvasSize / ar), canvasSize);
  }
  else {
    setCanvasDim(canvasSize, Math.round(canvasSize / ar));
  }
}

// Draws the target image to the target canvas and reads the image data
// so it can be used with the network.
function updateTargetImg() {
    targetCtx.drawImage(targetImg, 0, 0, width, height);
    targetImgData = targetCtx.getImageData(0, 0, width, height).data;
    targetNNOutput = normalizeColorValues(imgDataToGrayscale(targetImgData));
}


// Scale of the neural network's inputs vs. canvas
/*var xScale = 2, yScale = 2, xCenter = 0, yCenter = 0;

function planeXToCanvas(x) {
  return width / 2 + width * (x - xCenter) / xScale;
}

function planeYToCanvas(y) {
  return height / 2 - height * (y - yCenter) / yScale;
}

function canvasXToPlane(x) {
  return xCenter + xScale * (x - width / 2) / width;
}

function canvasYToPlane(y) {
  return yCenter - yScale * (y - height / 2) / height;
}*/


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
    case "sigmoid":
      activationFunction = NN.SIGMOID;
      break;
    case "tanh":
      activationFunction = NN.TANH;
      break;
  }
}


var hiddenLayers = parseHiddenLayersString(hiddenLayersInput.value);
var activationFunction;
updateActivationFunction();

var batchSize = Number(batchSizeInput.value);

var agent = {
  isTraining: false,
  isRendering: false,
  learningRate: Number(learningRateInput.value),

  renderWorkers: [],
  renderWorkerCount: 0,

  renderChunks: [],
  renderChunksDone: 0,

  initNetwork(hiddenLayers, af) {
    this.nn = new NN({
      layerSizes: [2].concat(hiddenLayers).concat([1]),
      activationFunctions: [af, NN.SIGMOID],
      wInit: {
        method: NN.RANDOM,
        range: 0.5,
      },
      bInit: {
        method: NN.RANDOM,
        range: 0.5,
      },
    });
  },

  initWorkers() {

  },

  initRenderWorkers(n) {
    this.renderWorkerCount = n;
    for (let i = 0; i < n; i++) {
      let worker = new Worker("render.js");
      this.renderWorkers.push(worker);

      // When the worker is done, draw its data to the correct
      // position on the canvas
      worker.onmessage = function(event) {
        let data = event.data;
        if (data.type == "done") {
          console.log("worker done: ", data.chunkI, data.y);
          this.renderChunksDone++;
          this.renderChunks[data.chunkI] = data.imgDataArr;

          // If the render is done, draw the image and finish the render
          if (this.renderChunksDone == this.renderWorkerCount) {
            // Combine all render chunks into one data array
            let cumImgDataArr = new Uint8ClampedArray(this.renderChunks.reduce((acc, curr) => [...acc, ...curr], []));
            ctx.putImageData(new ImageData(cumImgDataArr, width, height), 0, 0);

            this.renderChunksDone = 0;
            this.isRendering = false;
            console.log(performance.now() - this.renderStartTime);
          }
        }
      }.bind(this);

      // Send the current network to the worker so it can render when prompted
      worker.postMessage({
        type: "nn",
        nn: agent.nn.serialize(),
      });
    }
  },

  draw() {
    if (!this.isRendering) {
      this.isRendering = true;
      this.renderStartTime = performance.now();

      this.renderChunksDone = 0;
      this.renderChunks = new Array(this.renderWorkerCount);

      let baseChunkHeight = Math.floor(height / this.renderWorkerCount);
      let yStart = 0;
      for (let i = 0; i < this.renderWorkerCount; i++) {
        let worker = this.renderWorkers[i];
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
  },

  learn() {

  }
}
agent.initNetwork(hiddenLayers, activationFunction);
agent.initRenderWorkers(4);
agent.draw();


var iteration = 0;
var iterationsPerFrame = Number(ipfInput.value);

stopButton.setAttribute("disabled", true);


function draw() {

}


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

hiddenLayersInput.addEventListener("change", () => {
  let layers = parseHiddenLayersString(hiddenLayersInput.value);
  if (layers) {
    // update the layers and reset the agent
    hiddenLayers = layers;
    agent.initNetwork(hiddenLayers, activationFunction);
    iteration = 0;
  }
  else {
    hiddenLayersInput.value = formatHiddenLayers(hiddenLayers);
  }
});

activationFunctionInput.addEventListener("change", () => {
  // Update the af and reset the agent
  updateActivationFunction();
  agent.initNetwork(hiddenLayers, activationFunction);
  iteration = 0;
});

learningRateInput.addEventListener("change", () => {
  let n = Number(learningRateInput.value);
  if (Number.isNaN(n)) {
    learningRateInput.value = agent.learningRate;
  }
  else {
    agent.learningRate = n;
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


ipfInput.addEventListener("change", () => {
  let n = Number(ipfInput.value);
  if (Number.isNaN(n) || !Number.isInteger(n) || n <= 0) {
    ipfInput.value = iterationsPerFrame;
  }
  else {
    iterationsPerFrame = n;
  }
});

startButton.addEventListener("click", () => {
  agent.isTraining = true;
  disableInput(startButton);
  enableInput(stopButton);
  coreInputs.forEach((input) => {disableInput(input)});
});

stopButton.addEventListener("click", () => {
  agent.isTraining = false;
  disableInput(stopButton);
  enableInput(startButton);
  coreInputs.forEach((input) => {enableInput(input)});
});

resetNetworkButton.addEventListener("click", () => {

});

var keys = {};

window.addEventListener("keydown", (e) => {
  keys[e.key] = true;
});

window.addEventListener("keyup", (e) => {
  keys[e.key] = false;
});
