const path = require('path');
const fs = require('fs');

const wasmDir = path.join(__dirname, 'node_modules', '@tensorflow', 'tfjs-tflite', 'wasm');
const wasmJsPath = path.join(wasmDir, 'tflite_web_api_cc.js');
const wasmBinaryPath = path.join(wasmDir, 'tflite_web_api_cc.wasm');

const wasmBinary = fs.readFileSync(wasmBinaryPath);
const factory = require(wasmJsPath);

function fileUrl(p) {
  const resolved = path.resolve(p);
  return 'file:///' + resolved.replace(/\\/g, '/');
}

const moduleConfig = {
  wasmBinary: wasmBinary,
  locateFile: function(filePath) {
    const resolved = path.join(wasmDir, filePath);
    return fileUrl(resolved);
  },
};

factory(moduleConfig).then(function(Module) {
  // Check TFLiteWebModelRunner methods
  console.log('TFLiteWebModelRunner type:', typeof Module.TFLiteWebModelRunner);
  
  // Try to see if it's a class/constructor
  if (typeof Module.TFLiteWebModelRunner === 'function') {
    console.log('TFLiteWebModelRunner is a function');
    console.log('TFLiteWebModelRunner.prototype keys:', Object.getOwnPropertyNames(Module.TFLiteWebModelRunner.prototype));
    
    // Check for 'create' method
    console.log('TFLiteWebModelRunner.create:', typeof Module.TFLiteWebModelRunner.create);
    
    // Check prototype methods for 'create', 'infer', etc
    const proto = Module.TFLiteWebModelRunner.prototype;
    const methods = Object.getOwnPropertyNames(proto).filter(k => k !== 'constructor');
    console.log('Methods on prototype:', methods);
  }
  
  // Also check the tfweb object
  console.log('\nModule keys with "Runner":', Object.keys(Module).filter(k => k.includes('Runner')));
  console.log('Module keys with "create":', Object.keys(Module).filter(k => k.includes('create')));
  
  process.exit(0);
}).catch(function(e) {
  console.log('Error:', e);
  process.exit(1);
});
