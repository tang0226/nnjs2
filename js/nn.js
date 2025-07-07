class NN {
  constructor(obj) {

    // layers
    this.layerSizes = obj.layerSizes;
    if (obj.layerSizes < 2) {
      throw new Error("ValueError: Network cannot have fewer than 2 layers");
    }

    this.numLayers = obj.numLayers;
    this.inputLayerSize = obj.layerSizes[0];
    this.outputLayerSize = obj.layerSizes[obj.numLayers - 1];

    // Activation functions
    if (obj.activationFunctions || obj.af) {
      let af = obj.activationFunctions || obj.af;
      if (af.length > obj.layerSizes - 1) {
        throw new Error("ValueError: Wrong number of activation functions");
      }
      else if (af.length == obj.numLayers - 1) {
        this.af = af;
      }
      else {
        if (af.length == 2) {
          this.af = new Array(this.numLayers - 2);
          this.af.fill(af[0], 0, this.numLayers - 2);
          this.af[this.numLayers - 2] = af[1];
        }
        else {
          throw new Error("ValueError: Wrong number of activation functions");
        }
      }
    }

    this.activationFunctions = this.af;

    // Weights [layerFrom][nTo][nFrom]
    this.weights = this.w = [];

    if (obj.wInit) {
      if (obj.wInit.method) {
        if (obj.wInit.method == NN.RANDOM) {
          if (obj.wInit.range) {
            this.wInitFunc = function() {
              return Math.random() * obj.wInit.range * 2 - obj.wInit.range;
            }
          }
          else {
            throw new Error("Error: Random weight init method requires a range parameter");
          }
        }
        else {
          throw new Error("Error: unrecognized weight init method");
        }
      }
    }
    else {
      this.wInitFunc = function() {return 0};
    }

    for (let i = 0; i < this.numLayers - 1; i++) {
      this.weights.push(NN.init2d(this.layerSizes[i + 1], this.layerSizes[i], this.wInitFunc));
    }

    // Biases [nonInputLayer][n]
    this.biases = this.b = [];

    if (obj.bInit) {
      if (obj.bInit.method) {
        if (obj.bInit.method == NN.RANDOM) {
          if (obj.bInit.range) {
            this.bInitFunc = function() {
              return Math.random() * obj.bInit.range * 2 - obj.bInit.range;
            }
          }
          else {
            throw new Error("Error: Random bias init method requires a range parameter");
          }
        }
        else {
          throw new Error("Error: unrecognized bias init method");
        }
      }
    }
    else {
      this.bInitFunc = function() {return 0};
    }

    for (let i = 0; i < this.numLayers - 1; i++) {
      let l = [];
      for (let j = 0; j < this.layerSizes[i + 1]; j++) {
        l.push(this.bInitFunc());
      }
      this.biases.push(l);
    }

    // outputs (z = sum(prevA * weight) + bias; a = activationFunc(z))
    this.outputs = this.z = [];
    for (let i = 1; i < this.numLayers; i++) {
      let arr = new Array(this.layerSizes[i]);
      arr.fill(0);
      this.z.push(arr);
    }

    // Activations [nonInputLayer][n]
    this.activations = this.a = [];
    for (let i = 1; i < this.numLayers; i++) {
      let arr = new Array(this.layerSizes[i]);
      arr.fill(0);
      this.a.push(arr);
    }

    this.inputActivations = this.inputs = this.a0 = new Array(this.layerSizes[0]);
    this.inputActivations.fill(0);
  }

  static RANDOM = "RANDOM";
  static XAVIER = "XAVIER";
  static HE = "HE";

  static SIGMOID = {
    func(x) {
      return 1 / (1 + Math.exp(-x))
    },
    dFunc(x) {
      let s = this.func(x);
      return s * (1 - s);
    },
    name: "sigmoid"
  };

  static RELU = {
    func(x) {
      return Math.max(0, x);
    },
    dFunc(x) {
      return x > 0 ? 1 : 0;
    },
    name: "relu"
  };

  static zero2d(a, b) {
    let res = [];
    for (let i = 0; i < a; i++) {
      let arr = new Array(b);
      arr.fill(0);
      res.push(arr);
    }
    return res;
  }

  static init2d(a, b, func) {
    let res = [];
    for (let i = 0; i < a; i++) {
      let arr = [];
      for (let j = 0; j < b; j++) {
        arr.push(func());
      }
      res.push(arr);
    }
    return res;
  }
}
