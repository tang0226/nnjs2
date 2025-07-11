class _2048 {
  constructor(obj = {}) {
    this.gridSize = obj.gridSize || 4;
    this.fourSpawnRate = obj.fourSpawnRate || 0.1;
    this.initialSpawns = obj.initialSpawns || 2;

    this.grid = [];
    for (let i = 0; i < this.gridSize; i++) {
      this.grid.push(new Array(this.gridSize));
    }

    this.replay = [];

    this.reset();
  }

  reset() {
    this.score = 0;
    this.turns = 0;
    this.replay = [];
    this.gameOver = false;

    for (let row of this.grid) {
      row.fill(0);
    }
    this.randomSpawn();
    this.randomSpawn();
  }

  randomSpawn() {
    let open = [];
    for (let y = 0; y < this.gridSize; y++) {
      for (let x = 0; x < this.gridSize; x++) {
        if (!this.grid[y][x]) {
          open.push([x, y]);
        }
      }
    }

    let coords = open[Math.floor(Math.random() * open.length)];
    let tile = Math.random() < this.fourSpawnRate ? 2 : 1;
    this.grid[coords[1]][coords[0]] = tile;
    this.replay.push({spawn: [...coords, tile]});
  }

  move(dir) {
    if (this.gameOver) return -1;

    let turnScore = 0, ortho, orthoDir;

    switch (dir) {
      case this.constructor.UP:
        ortho = 1;
        orthoDir = -1;
        break;
      case this.constructor.DOWN:
        ortho = 1;
        orthoDir = 1;
        break;
      case this.constructor.RIGHT:
        ortho = 0;
        orthoDir = 1;
        break;
      case this.constructor.LEFT:
        ortho = 0;
        orthoDir = -1;
        break;
    }

    let lines = [];
    for (let i = 0; i < this.gridSize; i++) {
      let line = [];
      if (ortho == 1) {
        for (let j = 0; j < this.gridSize; j++) {
          line.push([i, j]);
        }
      }
      else {
        for (let j = 0; j < this.gridSize; j++) {
          line.push([j, i]);
        }
      }
      if (orthoDir == 1) {
        line.reverse();
      }
      lines.push(line);
    }

    let action = false;
    for (let i = 0; i < this.gridSize; i++) {
      let posI = 0;
      let line = lines[i]
      let lastUncombinedTile = -1;
      for (let j = 0; j < this.gridSize; j++) {
        let c = line[j]
        let tile = this.grid[c[1]][c[0]];
        if (tile) {
          // If the two tiles match;
          if (tile == lastUncombinedTile) {
            // Combine
            this.grid[line[posI - 1][1]][line[posI - 1][0]] = tile + 1;
            this.grid[c[1]][c[0]] = 0;
            turnScore += this.constructor.POW2[tile + 1];

            // Reset the combining
            lastUncombinedTile = -1;
            action = true;
          }
          else {
            lastUncombinedTile = tile;
            // If tile is already in the next position, it can't move
            if (j != posI) {
              // Move the tile
              this.grid[c[1]][c[0]] = 0;
              this.grid[line[posI][1]][line[posI][0]] = tile;
              action = true;
            }
            posI++;
          }
        }
      }
    }
    if (!action) {
      return -1;
    }

    // Add move to replay
    this.replay.push({move: dir});

    // Spawn another tile
    this.randomSpawn();
    
    this.score += turnScore;
    this.turns++;

    if (this.checkGameEnd()) {
      this.gameOver = true;
    }

    return turnScore;
  }

  up() {
    return this.move(this.constructor.UP);
  }

  down() {
    return this.move(this.constructor.DOWN);
  }

  right() {
    return this.move(this.constructor.RIGHT);
  }

  left() {
    return this.move(this.constructor.LEFT);
  }

  checkGameEnd() {
    for (let y = 0; y < this.gridSize; y++) {
      for (let x = 0; x < this.gridSize; x++) {
        if (this.grid[y][x] == 0) {
          return false;
        }
      }
    }

    for (let i = 0; i < this.gridSize; i++) {
      for (let j = 0; j < this.gridSize - 1; j++) {
        if (
          this.grid[i][j] == this.grid[i][j + 1] ||
          this.grid[j][i] == this.grid[j + 1][i]
        ) {
          return false;
        }
      }
    }

    return true;
  }


  static UP = 1;
  static RIGHT = 2;
  static DOWN = 3;
  static LEFT = 4;

  static POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216];
}