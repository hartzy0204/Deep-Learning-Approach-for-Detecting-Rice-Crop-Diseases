# Rice Disease Detection

A web application that detects rice leaf diseases using a YOLOv5 model (TensorFlow Lite). Upload an image and get the top-3 detections with bounding boxes drawn on the image.

## Detected Diseases

| Label | Description |
|-------|-------------|
| Blight | Bacterial leaf blight |
| Brown spot | Brown spot disease |
| False Smut | False smut disease |
| Healthy | Healthy leaf |
| Leaf Smut | Leaf smut disease |
| Rice blast | Rice blast disease |
| Stem Rot | Stem rot disease |
| Tungro | Tungro disease |

## Setup

```bash
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000).

## Deploy to Vercel

```bash
npm i -g vercel
vercel --prod
```

The project includes `vercel.json` and `api/index.js` for serverless deployment. The model loads lazily on the first request and is cached for subsequent calls.

## API

`POST /predict` — Upload an image and receive detections.

- **Content-Type**: `multipart/form-data`
- **Field**: `image` (file)
- **Response**:
  ```json
  {
    "detections": [
      {
        "class": 4,
        "label": "Leaf Smut",
        "confidence": 0.43,
        "x1": 121.01,
        "y1": 80.63,
        "x2": 367.65,
        "y2": 440.11
      }
    ],
    "numDetections": 3,
    "originalSize": { "width": 1920, "height": 1080 }
  }
  ```

Coordinates are in the original image's pixel space. Only the top-3 highest-confidence detections are returned.

## Project Structure

```
├── api/index.js          # Vercel serverless entry point
├── asset/
│   ├── best-fp16.tflite  # YOLOv5 TFLite model
│   └── labels.txt        # Class labels
├── public/
│   └── index.html        # Frontend UI
├── index.js              # Express server & model inference
├── package.json
└── vercel.json           # Vercel deployment config
```
