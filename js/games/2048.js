class _2048 {
  constructor(obj = {}) {
    this.gridSize = obj.gridSize || 4;
    this.fourSpawnRate = obj.fourSpawnRate || 0.1;
    this.initialSpawns = obj.initialSpawns || 2;

    this.grid = [];
    for (let i = 0; i < this.gridSize; i++) {
      this.grid.push(new Array(this.gridSize));
    }

    this.score = 0;
    this.turns = 0;

    this.reset();
  }

  reset() {
    this.score = 0;
    this.turns = 0;
    
    for (let row of this.grid) {
      row.fill(0);
    }
    this.randomSpawn();
    this.randomSpawn();
    console.log(this.grid);
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
    if (Math.random() < this.fourSpawnRate) {
      this.grid[coords[1]][coords[0]] = 2;
    }
    else {
      this.grid[coords[1]][coords[0]] = 1;
    }
  }



  static UP = 1;
  static RIGHT = 2;
  static DOWN = 3;
  static LEFT = 4;
}