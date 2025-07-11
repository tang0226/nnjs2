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


class Agent {
  constructor(obj) {
    this.nn = obj.nn;
    this.game = obj.game;
  }

  act() {
    let a = this.nn.feedForward(this.game.grid.reduce((a = [], b) => a.concat(b)));
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
        console.log(choices[i].choice);
        break;
      }
      i++;
    }
  }
}


var nn = new NN({
  layerSizes: [16, 48, 48, 4],
  af: [NN.RELU, NN.SIGMOID],
  wInit: {
    method: NN.RANDOM,
    range: 2,
  },
  bInit: {
    method: NN.RANDOM,
    range: 2,
  },
});


var game = new _2048();
var agent = new Agent({
  nn: nn,
  game: game,
});

var tileMargin = 10;
var tileSize = 100;
var tileFontSize = tileSize * 0.5;
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

var fontSizeCoeffs = [0, 1, 1, 1, 1, 1, 1, 0.9, 0.9, 0.9, 0.75, 0.75, 0.75, 0.75, 0.6, 0.6, 0.6];


var _canvasSideLength = tileSize * game.gridSize + tileMargin * (game.gridSize + 1);
setCanvasDim(_canvasSideLength, _canvasSideLength);

function draw() {
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
        ctx.font = `${tileFontSize * fontSizeCoeffs[tile]}px Arial`;
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


draw();
