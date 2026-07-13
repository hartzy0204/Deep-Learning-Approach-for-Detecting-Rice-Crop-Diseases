const fs = require('fs');
const path = require('path');
const http = require('http');

const imgPath = path.join(__dirname, 'test-image.png');
const imgData = fs.readFileSync(imgPath);
const boundary = '----TestBoundary' + Date.now();

const header = '--' + boundary + '\r\n' +
  'Content-Disposition: form-data; name="image"; filename="test.png"\r\n' +
  'Content-Type: image/png\r\n\r\n';

const footer = '\r\n--' + boundary + '--\r\n';

const bodyBuffer = Buffer.concat([
  Buffer.from(header, 'utf-8'),
  imgData,
  Buffer.from(footer, 'utf-8'),
]);

const options = {
  hostname: 'localhost',
  port: 3000,
  path: '/predict',
  method: 'POST',
  headers: {
    'Content-Type': 'multipart/form-data; boundary=' + boundary,
    'Content-Length': bodyBuffer.length,
  },
};

const req = http.request(options, (res) => {
  let data = '';
  res.on('data', (chunk) => (data += chunk));
  res.on('end', () => {
    const parsed = JSON.parse(data);
    console.log('Num detections:', parsed.numDetections);
    console.log('Original size:', JSON.stringify(parsed.originalSize));
    const dets = parsed.detections || [];
    console.log('');
    if (dets.length === 0) {
      console.log('No detections');
    } else {
      console.log('All detections:');
      for (const d of dets) {
        console.log('  ' + d.label + ' (' + (d.confidence * 100).toFixed(1) + '%) x1=' + d.x1 + ' y1=' + d.y1 + ' x2=' + d.x2 + ' y2=' + d.y2);
      }
      console.log('');
      console.log('Top 3:');
      console.log(JSON.stringify(dets.slice(0, 3), null, 2));
    }
  });
});
req.on('error', (e) => console.error('Error:', e.message));
req.write(bodyBuffer);
req.end();
