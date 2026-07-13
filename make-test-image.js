const sharp = require('sharp');
const w = 416, h = 416;
const img = Buffer.alloc(w * h * 3, 128);
sharp(img, { raw: { width: w, height: h, channels: 3 } })
  .png()
  .toFile('test-image.png', (e) => {
    if (e) throw e;
    console.log('Test image created: test-image.png');
  });
