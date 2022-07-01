//process.env.PYTHONPATH = '/home/pi/.local/lib/python3.7/site-packages/'

const { spawn } = require('child_process');
const { exit } = require('process');

const child = spawn('python3', ['./AudioDetectionDaemon/AudioDeamon.py']);
//const child = spawn('python3', ['-m' ,'site']);



const hiBuf = Buffer.from('Hi!');
const testBuif = Buffer.from('test!');

console.log(hiBuf.toString())
console.log(testBuif.toString())

let length=hiBuf.byteLength+testBuif.byteLength
console.log(length)
const DataBuffer=Buffer.alloc(length);

hiBuf.copy(DataBuffer);

testBuif.copy(DataBuffer,hiBuf.byteLength);

console.log(DataBuffer.toString())

exit();
let filledstatus=0;




console.log(child.pid)

child.stderr.on('data', data => {
    console.log(`stderr: ${data}`);
  });

  child.stdout.on('data', async data =>  {
console.log(String(data))}

);