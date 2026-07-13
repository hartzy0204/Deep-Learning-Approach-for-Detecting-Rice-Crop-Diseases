const fs = require('fs');
const http = require('http');
const path = require('path');

const imagePath = path.join(__dirname, 'test-image.png');
const imageData = fs.readFileSync(imagePath);

const boundary = '----formdataopencodeboundary';
const headers = {
  'Content-Type': `multipart/form-data; boundary=${boundary}`,
};

const body = Buffer.concat([
  Buffer.from(`--${boundary}\r\n`),
  Buffer.from(`Content-Disposition: form-data; name="image"; filename="test-image.png"\r\n`),
  Buffer.from(`Content-Type: image/png\r\n\r\n`),
  imageData,
  Buffer.from(`\r\n--${boundary}--\r\n`),
]);

const req = http.request(
  { hostname: 'localhost', port: 3000, path: '/predict', method: 'POST', headers: { ...headers, 'Content-Length': body.length } },
  (res) => {
    let data = '';
    res.on('data', (chunk) => (data += chunk));
    res.on('end', () => {
      console.log('Status:', res.statusCode);
      console.log('Response:', data);
    });
  }
);

req.on('error', (e) => console.error('Error:', e.message));
req.write(body);
req.end();
