const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage() });
const MODEL_PATH = path.join(__dirname, 'asset', 'best-fp16.tflite');
const LABELS_PATH = path.join(__dirname, 'asset', 'labels.txt');

const WASM_DIR = path.join(__dirname, 'node_modules', '@tensorflow', 'tfjs-tflite', 'wasm');
const WASM_JS = path.join(WASM_DIR, 'tflite_web_api_cc.js');
const WASM_BIN = path.join(WASM_DIR, 'tflite_web_api_cc.wasm');

const MODEL_WIDTH = 416;
const MODEL_HEIGHT = 416;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

const ANCHORS = [
  [10, 13, 16, 30, 33, 23],
  [30, 61, 62, 45, 59, 119],
  [116, 90, 156, 198, 373, 326],
];

const STRIDES = [8, 16, 32];
const GRID_SIZES = [52, 26, 13];
const NUM_CLASSES = 8;
const NUM_ANCHORS = 3;
const BOX_ATTRS = 5;

let modelRunner;
let inputShape;
let outputShape;
let labels;

function loadLabels() {
  const content = fs.readFileSync(LABELS_PATH, 'utf-8');
  return content.split('\n').filter(line => line.trim()).map(line => line.trim());
}

function fileUrl(p) {
  return 'file:///' + path.resolve(p).replace(/\\/g, '/');
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function iou(boxA, boxB) {
  const xA = Math.max(boxA.x1, boxB.x1);
  const yA = Math.max(boxA.y1, boxB.y1);
  const xB = Math.min(boxA.x2, boxB.x2);
  const yB = Math.min(boxA.y2, boxB.y2);
  const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
  const areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);
  return inter / (areaA + areaB - inter);
}

function nms(detections, iouThreshold) {
  detections.sort((a, b) => b.confidence - a.confidence);
  const result = [];
  for (let i = 0; i < detections.length; i++) {
    const det = detections[i];
    let keep = true;
    for (let j = 0; j < result.length; j++) {
      if (iou(det, result[j]) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) result.push(det);
  }
  return result;
}

function decodeYoloOutput(rawData) {
  const detections = [];
  let offset = 0;

  for (let scale = 0; scale < 3; scale++) {
    const gridSize = GRID_SIZES[scale];
    const stride = STRIDES[scale];
    const anchors = ANCHORS[scale];
    const scaleLen = gridSize * gridSize * NUM_ANCHORS * (BOX_ATTRS + NUM_CLASSES);

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        for (let a = 0; a < NUM_ANCHORS; a++) {
          const idx = offset + (i * gridSize + j) * NUM_ANCHORS * (BOX_ATTRS + NUM_CLASSES) + a * (BOX_ATTRS + NUM_CLASSES);
          const tx = sigmoid(rawData[idx]);
          const ty = sigmoid(rawData[idx + 1]);
          const tw = Math.exp(rawData[idx + 2]);
          const th = Math.exp(rawData[idx + 3]);
          const objScore = sigmoid(rawData[idx + 4]);

          if (objScore < CONF_THRESHOLD) continue;

          let maxClassScore = 0;
          let maxClassIdx = -1;
          for (let c = 0; c < NUM_CLASSES; c++) {
            const score = sigmoid(rawData[idx + 5 + c]);
            if (score > maxClassScore) {
              maxClassScore = score;
              maxClassIdx = c;
            }
          }

          const confidence = objScore * maxClassScore;
          if (confidence < CONF_THRESHOLD) continue;

          const cx = (j + tx) * stride;
          const cy = (i + ty) * stride;
          const w = tw * anchors[a * 2];
          const h = th * anchors[a * 2 + 1];

          detections.push({
            class: maxClassIdx,
            label: labels[maxClassIdx] || `class_${maxClassIdx}`,
            confidence: Math.round(confidence * 10000) / 10000,
            x1: Math.round((cx - w / 2) * 100) / 100,
            y1: Math.round((cy - h / 2) * 100) / 100,
            x2: Math.round((cx + w / 2) * 100) / 100,
            y2: Math.round((cy + h / 2) * 100) / 100,
          });
        }
      }
    }
    offset += scaleLen;
  }

  return nms(detections, IOU_THRESHOLD);
}

async function loadWasmModule() {
  const wasmBinary = fs.readFileSync(WASM_BIN);
  const factory = require(WASM_JS);
  const Module = await new Promise((resolve, reject) => {
    factory({ wasmBinary, locateFile: (fp) => fileUrl(path.join(WASM_DIR, fp)) })
      .then(resolve).catch(reject);
  });
  return Module;
}

async function loadModel() {
  const Module = await loadWasmModule();
  const modelBuffer = fs.readFileSync(MODEL_PATH);
  const modelBytes = new Uint8Array(modelBuffer);

  const ptr = Module._malloc(modelBytes.length);
  Module.HEAPU8.set(modelBytes, ptr);

  const options = { numThreads: 4, enableProfiling: false, maxProfilingBufferEntries: 1024 };
  const statusOr = Module.TFLiteWebModelRunner.CreateFromBufferAndOptions(ptr, modelBytes.length, options);

  if (!statusOr || !statusOr.ok()) {
    Module._free(ptr);
    throw new Error('Failed to create model runner: ' + (statusOr ? statusOr.errorMessage() : 'unknown error'));
  }

  const cppRunner = statusOr.value();
  modelRunner = {
    module: Module, memOffsets: [ptr], cppRunner,
    cleanUp() { this.memOffsets.forEach(o => this.module._free(o)); this.cppRunner.delete(); },
  };

  const inputs = readTensorInfos(Module, cppRunner.GetInputs());
  const outputs = readTensorInfos(Module, cppRunner.GetOutputs());
  inputShape = inputs[0].shape;
  outputShape = outputs[0].shape;
  labels = loadLabels();
}

function readTensorInfos(Module, vec) {
  const result = [];
  for (let i = 0; i < vec.size(); i++) {
    const info = vec.get(i);
    result.push({
      id: info.id, name: info.name,
      dataType: info.dataType, shape: info.shape.split(',').map(Number),
      data: () => info.data(),
    });
  }
  if (typeof vec.delete === 'function') vec.delete();
  return result;
}

async function preprocessImage(imageBuffer) {
  const image = sharp(imageBuffer);
  const metadata = await image.metadata();
  const ow = metadata.width, oh = metadata.height;

  let pixelData;
  let scale, padX, padY;

  if (ow === MODEL_WIDTH && oh === MODEL_HEIGHT) {
    scale = 1; padX = 0; padY = 0;
    const { data } = await image.toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
    pixelData = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) pixelData[i] = data[i] / 255.0;
  } else {
    scale = Math.min(MODEL_WIDTH / ow, MODEL_HEIGHT / oh);
    const sw = Math.round(ow * scale);
    const sh = Math.round(oh * scale);

    const resized = await image.resize(sw, sh).toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });

    const canvas = Buffer.alloc(MODEL_WIDTH * MODEL_HEIGHT * 3, 114);
    padX = Math.floor((MODEL_WIDTH - sw) / 2);
    padY = Math.floor((MODEL_HEIGHT - sh) / 2);

    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const srcIdx = (y * sw + x) * 3;
        const dstIdx = ((padY + y) * MODEL_WIDTH + (padX + x)) * 3;
        canvas[dstIdx] = resized.data[srcIdx];
        canvas[dstIdx + 1] = resized.data[srcIdx + 1];
        canvas[dstIdx + 2] = resized.data[srcIdx + 2];
      }
    }

    pixelData = new Float32Array(MODEL_WIDTH * MODEL_HEIGHT * 3);
    for (let i = 0; i < canvas.length; i++) pixelData[i] = canvas[i] / 255.0;
  }

  return { pixelData, originalWidth: ow, originalHeight: oh, scale, padX, padY };
}

app.post('/predict', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file provided' });
  if (!modelRunner) return res.status(503).json({ error: 'Model not loaded' });

  try {
    const { pixelData, originalWidth, originalHeight, scale, padX, padY } = await preprocessImage(req.file.buffer);
    const inputs = modelRunner.cppRunner.GetInputs();
    const inputTensor = inputs.get(0);
    inputTensor.data().set(pixelData);
    inputs.delete();

    const success = modelRunner.cppRunner.Infer();
    if (!success) throw new Error('Inference failed');

    const outputs = modelRunner.cppRunner.GetOutputs();
    const outputTensor = outputs.get(0);
    const outputData = Array.from(outputTensor.data());
    outputs.delete();

    const rawDets = decodeYoloOutput(outputData);

    const detections = rawDets.map(d => {
      const x1 = Math.max(0, Math.round(((d.x1 - padX) / scale) * 100) / 100);
      const y1 = Math.max(0, Math.round(((d.y1 - padY) / scale) * 100) / 100);
      const x2 = Math.min(originalWidth, Math.round(((d.x2 - padX) / scale) * 100) / 100);
      const y2 = Math.min(originalHeight, Math.round(((d.y2 - padY) / scale) * 100) / 100);
      return {
        class: d.class,
        label: d.label,
        confidence: d.confidence,
        x1, y1, x2, y2,
      };
    }).filter(d => d.x2 > d.x1 && d.y2 > d.y1);

    const topDets = detections.sort((a, b) => b.confidence - a.confidence).slice(0, 3);

    res.json({
      detections: topDets,
      numDetections: topDets.length,
      originalSize: { width: originalWidth, height: originalHeight },
    });
  } catch (err) {
    console.error('Prediction error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

async function start() {
  try {
    await loadModel();
    console.log('Model loaded: %s', path.basename(MODEL_PATH));
    console.log('Input: %s  Output: %s', JSON.stringify(inputShape), JSON.stringify(outputShape));
    console.log('Labels:', labels);
    app.listen(PORT, () => console.log('Server running on http://localhost:%d', PORT));
  } catch (err) {
    console.error('Failed to start:', err);
    process.exit(1);
  }
}

start();
