const mainCanvas = document.getElementById("main-canvas");
const ctx = mainCanvas.getContext("2d");

const showDataCheckbox = document.getElementById("show-data-checkbox");
const activationFunctionInput = document.getElementById("activation-function");
const hiddenLayersInput = document.getElementById("hidden-layers-input");
const learningRateInput = document.getElementById("learning-rate");
const epfInput = document.getElementById("epf-input");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const resetNetworkButton = document.getElementById("reset-network-button");
const clearPointsButton = document.getElementById("clear-points-button");


hiddenLayersInput.value = "40, 20";
learningRateInput.value = "0.2";
epfInput.value = "1";

var width, height;
function setCanvasDim(w, h) {
  mainCanvas.width = w;
  mainCanvas.height = h;
  mainCanvas.style.width = w + "px";
  mainCanvas.style.height = h + "px";
  width = w;
  height = h;
}

setCanvasDim(200, 200);


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
}

var xScale = 2, yScale = 2, xCenter = 0, yCenter = 0;
var points = [];

var nnFuncRes = 2;

var targetRadius = 3;
var targetStroke = 1;


var agent = {
  learningRate: Number(learningRateInput.value),

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

  draw() {
    let imgData = ctx.createImageData(width, height);
    let i = 0;
    for (let y = 0; y < height; y++) {
      let sy = canvasYToPlane(y);
      for (let x = 0; x < width; x++) {
        let sx = canvasXToPlane(x);
        let bw = Math.round(this.nn.feedForward([sx, sy])[0] * 255);
        imgData.data[i] = bw;
        imgData.data[i + 1] = bw;
        imgData.data[i + 2] = bw;
        imgData.data[i + 3] = 255;
        i += 4;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  },

  learn() {
    for (let p of points) {
      this.nn.feedForward([p[0], p[1]]);
      this.nn.backpropagate([p[2]]);
    }

    this.nn.endIteration(this.learningRate);
  }
}

function drawPoints() {
  for (let p of points) {
    ctx.fillStyle = p[2] ? "#FFFFFF" : "#000000";
    ctx.strokeStyle = p[2] ? "#000000" : "#FFFFFF";
    let cx = planeXToCanvas(p[0]);
    let cy = planeYToCanvas(p[1]);
    ctx.beginPath();
    ctx.arc(cx, cy, targetRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
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
    case "sigmoid":
      activationFunction = NN.SIGMOID;
      break;
    case "tanh":
      activationFunction = NN.TANH;
      break;
  }
}

var epoch = 0;
var running = false;
var drawInterval;
var epochsPerFrame = Number(epfInput.value);
var hiddenLayers = parseHiddenLayersString(hiddenLayersInput.value);
var activationFunction;
updateActivationFunction();

agent.initNetwork(hiddenLayers, activationFunction);

function draw() {
  agent.nn.startIteration();
  ctx.clearRect(0, 0, width, height);
  agent.draw();
  if (showDataCheckbox.checked) {
    drawPoints();
  }
  if (points.length) {
    for (let i = 0; i < epochsPerFrame; i++) {
      agent.learn();
      epoch++;
    }
    document.querySelector("#loss").innerText = agent.nn.avgLoss.toFixed(4);
    document.querySelector("#epoch").innerText = epoch;
  }
}

// Add a data point where the canvas is clicked
mainCanvas.addEventListener("click", (event) => {
  points.push([
    canvasXToPlane(event.offsetX),
    canvasYToPlane(event.offsetY),
    Number(Boolean(keys["Shift"]))
  ]);
  if (!running) {
    ctx.clearRect(0, 0, width, height);
    drawPoints();
  }
});

hiddenLayersInput.addEventListener("change", () => {
  let layers = parseHiddenLayersString(hiddenLayersInput.value);
  if (layers) {
    // update the layers and reset the agent
    hiddenLayers = layers;
    agent.initNetwork(hiddenLayers, activationFunction);
    epoch = 0;
  }
  else {
    hiddenLayersInput.value = formatHiddenLayers(hiddenLayers);
  }
});

activationFunctionInput.addEventListener("change", () => {
  // Update the af and reset the agent
  updateActivationFunction();
  agent.initNetwork(hiddenLayers, activationFunction);
  epoch = 0;
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

epfInput.addEventListener("change", () => {
  let n = Number(epfInput.value);
  if (Number.isNaN(n) || !Number.isInteger(n)) {
    epfInput.value = epochsPerFrame;
  }
  else {
    epochsPerFrame = n;
  }
});

startButton.addEventListener("click", () => {
  if (!running) {
    running = true;
    drawInterval = window.setInterval(draw, 0);
  }
});

stopButton.addEventListener("click", () => {
  if (running) {
    running = false;
    window.clearInterval(drawInterval);
  }
});

resetNetworkButton.addEventListener("click", () => {
  agent.initNetwork(hiddenLayers, activationFunction);
  epoch = 0;
});

clearPointsButton.addEventListener("click", () => {
  points = [];
  if (!running) {
    ctx.clearRect(0, 0, width, height);
    epoch = 0;
  }
});

var keys = {};

window.addEventListener("keydown", (e) => {
  keys[e.key] = true;
});

window.addEventListener("keyup", (e) => {
  keys[e.key] = false;
});
