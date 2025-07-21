const mainCanvas = document.getElementById("main-canvas");
const ctx = mainCanvas.getContext("2d");

const learningRateInput = document.getElementById("learning-rate")
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const resetNetworkButton = document.getElementById("reset-network-button");
const clearPointsButton = document.getElementById("clear-points-button");

var width, height;
function setCanvasDim(w, h) {
  mainCanvas.width = w;
  mainCanvas.height = h;
  mainCanvas.style.width = w + "px";
  mainCanvas.style.height = h + "px";
  width = w;
  height = h;
}

setCanvasDim(600, 300);


function planeToCanvas(x, y) {
  return {
    x: width / 2 + width * (x - xCenter) / xScale,
    y: height / 2 - height * (y - yCenter) / yScale,
  };
}

function canvasToPlane(x, y) {
  return {
    x: xCenter + xScale * (x - width / 2) / width,
    y: yCenter - yScale * (y - height / 2) / height,
  }
}

var xScale = 10, yScale = 1, xCenter = 0, yCenter = 0.5;
var points = [];

var nnFuncRes = 1;

var targetColor = "#FF0000";
var approximationColor = "#4287f5";
var targetSquareSize = 5;


var agent = {
  learningRate: Number(learningRateInput.value),

  initNetwork() {
    this.nn = new NN({
      layerSizes: [1, 30, 15, 1],
      activationFunctions: [NN.RELU, NN.SIGMOID],
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

  drawFunction() {
    ctx.lineWidth = 2;
    ctx.strokeStyle = approximationColor;
    ctx.beginPath();
    for (let x = 0; x < width; x += nnFuncRes) {
      let planeX = xCenter + xScale * (x - width / 2) / width;

      let output = this.nn.feedForward([planeX])[0];

      if (x == 0) {
        ctx.moveTo(x, height / 2 - height * (output - yCenter) / yScale);
      }
      else {
        ctx.lineTo(x, height / 2 - height * (output - yCenter) / yScale)
      }
    }
    ctx.stroke();
  },

  updateNetwork() {
    for (let p of points) {
      this.nn.feedForward([p[0]]);
      this.nn.backpropagate([p[1]]);
    }

    this.nn.endEpoch(this.learningRate);
  }
}
agent.initNetwork();

function drawPoints() {
  for (let p of points) {
    ctx.fillStyle = targetColor;
    let coords = planeToCanvas(p[0], p[1]);
    ctx.fillRect(
      coords.x - Math.floor(targetSquareSize / 2),
      coords.y - Math.floor(targetSquareSize / 2),
      targetSquareSize,
      targetSquareSize
    );
  }
}



var generation = 0;
var running = false;
var drawInterval;
function draw() {
  ctx.clearRect(0, 0, width, height);
  drawPoints();
  agent.drawFunction();
  if (points.length) {
    agent.updateNetwork();
    generation++;
    document.querySelector("#generation").innerText = generation;
  }
}

// Add a data point where the canvas is clicked
mainCanvas.addEventListener("click", (event) => {
  let coords = canvasToPlane(event.offsetX, event.offsetY);
  points.push([coords.x, coords.y]);
  if (!running) {
    ctx.clearRect(0, 0, width, height);
    drawPoints();
  }
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
  agent.initNetwork();
  generation = 0;
});

clearPointsButton.addEventListener("click", () => {
  points = [];
  if (!running) {
    ctx.clearRect(0, 0, width, height);
    generation = 0;
  }
});
