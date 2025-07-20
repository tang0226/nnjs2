const mainCanvas = document.getElementById("main-canvas");
const ctx = mainCanvas.getContext("2d");

var width, height;
function setCanvasDim(w, h) {
  mainCanvas.width = w;
  mainCanvas.height = h;
  mainCanvas.style.width = w + "px";
  mainCanvas.style.height = h + "px";
  width = w;
  height = h;
}

setCanvasDim(1000, 1000);


class _2048Agent {
  constructor(obj) {
    this.nn = obj.nn;
    this.game = obj.game;
  }

  act() {
    if (this.game.gameOver) {
      return false;
    }
    let a = this.nn.feedForward(formatGrid(this.game.grid));
    let choices = [
      {choice: _2048.UP, val: a[0]},
      {choice: _2048.RIGHT, val: a[1]},
      {choice: _2048.DOWN, val: a[2]},
      {choice: _2048.LEFT, val: a[3]},
    ];

    choices.sort((a, b) => a.val < b.val ? 1 : (b.val < a.val ? -1 : 0));

    let i = 0;
    while (i < 4) {
      if (this.game.move(choices[i].choice) != -1) {
        break;
      }
      i++;
    }
    this.score = this.game.score;
    this.turns = this.game.turns;
    if (game.gameOver) {
      this.done = true;
    }
    return true;
  }
}


var tileMargin = 10;
var tileSize = 100;
var colors = {
  boardBorder: "#b9ae9f",
  tiles: [
    "#cbc0b3", // empty
    "#ece4da", // 2
    "#ebe1ca", // 4
    "#eab477", // 8
    "#ea9762", // 16
    "#e87b5f", // 32
    "#e85e3d", // 64
    "#e6d36a", // 128
    "#e7d058", // 256
    "#e6cd42", // 512
    "#e5c92b", // 1024
    "#e4c505", // 2048
    "#b182b6", // 4096
    "#a259ae", // 8192
    "#9a3ca7", // 16384
    "#790084", // 32768
    "#5a0048", // 65536
    "#8f7ee7", // 131072
  ],
  twoFourFont: "#726551",
  otherTileFont: "#ffffff",
};

var fontSizeCoeffs = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.35, 0.35, 0.35, 0.35, 0.3, 0.3, 0.3, 0.275];


function drawGame(game) {
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = colors.boardBorder;
  ctx.fillRect(0, 0, width, height);

  for (let y = 0; y < game.gridSize; y++) {
    let tileCornerY = tileMargin + (tileSize + tileMargin) * y;
    for (let x = 0; x < game.gridSize; x++) {
      let tileCornerX = tileMargin + (tileSize + tileMargin) * x;
      let tile = game.grid[y][x];
      ctx.fillStyle = colors.tiles[tile];
      ctx.fillRect(tileCornerX, tileCornerY, tileSize, tileSize);

      if (tile > 0) {
        ctx.fillStyle = (tile == 1 || tile == 2) ? colors.twoFourFont : colors.otherTileFont;
        ctx.font = `${tileSize * fontSizeCoeffs[tile]}px Arial`;
        let text = _2048.POW2[tile].toString();
        let textMetrics = ctx.measureText(text);
        let textWidth = textMetrics.width;
        let textHeight = textMetrics.actualBoundingBoxAscent - textMetrics.actualBoundingBoxDescent;
        ctx.fillText(
          text,
          tileCornerX + tileSize / 2 - textWidth / 2,
          tileCornerY + tileSize / 2 + textHeight / 2
        );
      }
    }
  }

  document.querySelector("#score").innerText = game.score;
  document.querySelector("#turns").innerText = game.turns;
}


var nn = new NN({
  layerSizes: [256, 32, 4],
  af: [NN.SIGMOID, NN.SIGMOID],
  wInit: {
    method: NN.RANDOM,
    range: 3,
  },
  bInit: {
    method: NN.RANDOM,
    range: 3,
  },
});

var game = new _2048({
  gridSize: 4
});

var agent = new _2048Agent({
  nn: nn,
  game: game,
});


function formatGridOneHot(grid) {
  let inputs = (new Array(256)).fill(0);
  for (let i = 0; i < 16; i++) {
    inputs[i * 16 + grid[Math.floor(i / 4)][i % 4] - 1] = 1;
  }
  return inputs;
}

function formatGridValue(grid) {
  return grid.reduce((a = [], b) => a.concat(b));
}

var formatGrid = formatGridOneHot;


var _canvasSideLength = tileSize * game.gridSize + tileMargin * (game.gridSize + 1);
setCanvasDim(_canvasSideLength, _canvasSideLength);

var simulationSpeed = 100;

var gamesPlayed = 0;

var gameAvgCount = 300;
var scoreAvgTotal = 0;
var scoreAvg;
var diff;
var learningRate = 0.1;

var scores = [];
var state = "agent playing";


function draw() {
  if (state == "agent playing") {
    for (let i = 0; i < simulationSpeed; i++) {
      if (agent.act()) {
        //drawGame(game);
      }
      else {
        drawGame(game);
        scores.push(game.score);
        gamesPlayed++;

        if (gamesPlayed > gameAvgCount) {
          // Train
          scoreAvg = scores.slice(-gameAvgCount - 1, -1).reduce((a, b) => a + b) / gameAvgCount;
          diff = game.score - scoreAvg;
          state = "agent training";
        }
        else {
          // Get more game data first
          game.reset();
        }

        break;
      }
    }
  }
  else if (state == "agent training") {
    console.log(scoreAvg, diff, game.score, game.turns);
    
    var wSum, bSum;
    let replay = new _2048({
      gridSize: game.gridSize,
      fourSpawnRate: game.fourSpawnRate,
      initialSpawns: game.initialSpawns
    });
    replay.grid = replay.grid.map((x) => (new Array(game.gridSize)).fill(0));
    
    for (let event of game.replay) {
      if (event.spawn) {
        let spawn = event.spawn;
        replay.grid[spawn[1]][spawn[0]] = spawn[2];
      }
      else if (event.move) {
        let y, bp;
        // Did the ai do well or poorly?
        if (game.score > scoreAvg) {
          // Well = reinforce actions taken
          y = [0, 0, 0, 0];
          switch (event.move) {
            case _2048.UP:
              y[0] = 1;
              break;
            case _2048.RIGHT:
              y[1] = 1;
              break;
            case _2048.DOWN:
              y[2] = 1;
              break;
            case _2048.LEFT:
              y[3] = 1;
              break;
          }
          bp = agent.nn.backpropagate(y, formatGrid(replay.grid));
        }
        else {
          y = agent.nn.feedForward(formatGrid(replay.grid));
          // Poorly = discourage actions taken
          switch (event.move) {
            case _2048.UP:
              y[0] = 0;
              break;
            case _2048.RIGHT:
              y[1] = 0;
              break;
            case _2048.DOWN:
              y[2] = 0;
              break;
            case _2048.LEFT:
              y[3] = 0;
              break;
          }
          bp = agent.nn.backpropagate(y);
        }

        if (wSum) {
          wSum = NN.add3d(wSum, bp.w);
        }
        else {
          wSum = bp.w;
        }

        if (bSum) {
          bSum = NN.add2d(bSum, bp.b);
        }
        else {
          bSum = bp.b;
        }
        replay.move(event.move, false);
      }
    }

    wSum = NN.mulScalar3d(wSum, -learningRate / game.turns);
    bSum = NN.mulScalar2d(bSum, -learningRate / game.turns);

    agent.nn.w = NN.add3d(wSum, agent.nn.w);
    agent.nn.b = NN.add2d(wSum, agent.nn.b);

    //return;
    state = "agent playing"
    game.reset();
  }

  setTimeout(draw, 0);
}

document.querySelector("#restart").addEventListener("click", () => {
  if (game.gameOver) {
    nn = new NN({
      layerSizes: [16, 48, 32, 4],
      af: [NN.LEAKY_RELU(), NN.SIGMOID],
      wInit: {
        method: NN.RANDOM,
        range: 2,
      },
      bInit: {
        method: NN.RANDOM,
        range: 2,
      },
    });
    game.reset();
    agent = new _2048Agent({
      nn: nn,
      game: game,
    });
    document.querySelector("#alert").innerText = "";
    draw();
  }
});

draw();

/*
document.addEventListener("keydown", (event) => {
  if (!game.gameOver) {
    switch (event.key) {
      case "ArrowUp":
        if (game.up() != -1) {
          draw();
        }
        break;
      case "ArrowDown":
        if (game.down() != -1) {
          draw();
        }
        break;
      case "ArrowRight":
        if (game.right() != -1) {
          draw();
        }
        break;
      case "ArrowLeft":
        if (game.left() != -1) {
          draw();
        }
        break;
      case " ":
        agent.act();
        draw();
        break;
    }
    if (game.gameOver) {
      document.querySelector("#alert").innerText = "Game Over!";
    }
  }

});
*/
