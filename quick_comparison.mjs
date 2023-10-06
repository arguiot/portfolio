#!/usr/bin/env bun

import { spawn } from 'child_process';
import { runAppleScript } from 'run-applescript';
import fs from 'fs';
import path from 'path';
import os from 'os';
const cliProgress = require('cli-progress');

// If `--rebalance <period>` is specified, pass it to the Python script
const rebalanceIndex = process.argv.indexOf('--rebalance');
const rebalancePeriod = rebalanceIndex !== -1 ? process.argv[rebalanceIndex + 1] : '';

const runBacktestWithProgressBar = () => new Promise((resolve, reject) => {
  // Create a new progress bar instance
  const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
  progressBar.start(100, 0); // 100% is the max

  const shared_file_path = './portfolio_progress';
  // If the file doesn't exist, create it
  if (!fs.existsSync(shared_file_path)) {
    fs.writeFileSync(shared_file_path, Buffer.alloc(8));
  }

  const mmap = Bun.mmap(shared_file_path, { shared: true });
  const log_progress = async () => {
    
    while (mmap[8]) { // wait until no other process is reading/writing
      // sleep for 10 milliseconds
      await Bun.sleep(10);
    }
    const buffer = new Uint8Array(mmap);
    const dataView = new DataView(buffer.buffer);

    // First 4 bytes represents total step count (integer)
    const totalSteps = dataView.getUint32(0, true); // get 4 bytes as UInt32 little-endian
    // Next 4 bytes represents current step (integer)
    const currentStep = dataView.getUint32(4, true); // get 4 bytes as UInt32 little-endian

    // Update the progress bar
    progressBar.update(currentStep);
    progressBar.setTotal(totalSteps);
  }


  // Clear the log file
  fs.writeFile('out/run_backtest.log', '', (err) => {
    if (err) throw err; // Optional; depends on whether you want to handle errors here
  });
  // If `--class <asset_class>` is specified, pass it to the Python script
  const assetClassIndex = process.argv.indexOf('--class');
  const assetClass = assetClassIndex !== -1 ? process.argv[assetClassIndex + 1] : '';
  // Construct the arguments to pass to the Python script
  const args = ['run_backtest.py'];
  if (assetClass) {
    args.push('--class', assetClass);
  }
  if (rebalancePeriod) {
    args.push('--rebalance', rebalancePeriod);
  }
  const pythonProcess = spawn('python', args);

  pythonProcess.stdout.on('data', (data) => {
    log_progress();
    // Write data to file without blocking
    fs.appendFile('out/run_backtest.log', data, (err) => {
      if (err) throw err; // Optional; depends on whether you want to handle errors here
    });
  });

  pythonProcess.stderr.on('data', (data) => {
    // Write data to file without blocking
    fs.appendFile('out/run_backtest.log', data, (err) => {
      if (err) throw err; // Optional; depends on whether you want to handle errors here
    });
  });

  pythonProcess.on('close', (code) => {
    progressBar.stop(); // Stop the progress bar when finished
    if (code !== 0) {
      reject(new Error(`Python script exited with code ${code}`));
    } else {
      resolve();
    }
  });
});

// 1. Force Quit Excel
try {
  if (os.platform() === 'darwin') {  // Check if it is macOS
    spawn('killall', ['Microsoft Excel']);
  }
} catch { }

if (!process.argv.includes('--skip-backtest')) {
  // Time the backtest
  console.log("⏱️ Starting backtest");
  const startTime = new Date();
  await runBacktestWithProgressBar();
  const endTime = new Date();
  const timeDiff = endTime - startTime;
  const seconds = Math.round(timeDiff / 1000);
  const minutes = Math.round(seconds / 60);
  console.log(`⏱️ Backtest took ${minutes} minutes and ${seconds % 60} seconds`);
  console.log("✅ Backtest done");
}

// 3. Open all .xlsx files and arrange windows using AppleScript
function getSortedFilePaths(dirPath) {
  let filePaths = [];

  // Fetch all files in the directory
  const files = fs.readdirSync(dirPath);

  // Build full file paths and filter out
  // temporary lock files created by Excel
  for (const file of files) {
    const filePath = path.join(dirPath, file);
    if (path.extname(filePath) === '.xlsx' && !filePath.includes('~$')) {
      filePaths.push(filePath);
    }
  }

  // Sort the file paths by modified time
  filePaths.sort((a, b) => {
    return fs.statSync(a).mtime.getTime() -
      fs.statSync(b).mtime.getTime();
  });

  return filePaths;
}

let filePaths = getSortedFilePaths(`./out/${rebalancePeriod}`);

console.log(`📂 Found ${filePaths.length} files`);
for (let filePath of filePaths) {
  if (filePath) {
    if (os.platform() === 'darwin') {  // Check if it is macOS
      spawn('open', ['-a', 'Microsoft Excel', filePath]);
    }
  }
}

if (os.platform() === 'darwin') {  // Check if it is macOS
  // Wait for Excel to open all files
  await new Promise(resolve => setTimeout(resolve, 5000));

  // Ensuring Excel remembers the last used screen
  await runAppleScript('tell application "Microsoft Excel" to activate')

  // Get screen's dimensions
  const screenSizeScript = `
  tell application "Finder"
      get bounds of window of desktop
  end tell
  `

  const screenSize = await runAppleScript(screenSizeScript);
  const [startX, startY, endX, endY] = screenSize.split(', ').map(Number);
  const screenWidth = endX - startX;
  const screenHeight = endY - startY;

  for (let index = 0; index < filePaths.length; index++) {
    const script = `
    tell application "Microsoft Excel"
      delay 1
      set windowWidth to ${screenWidth} / ${filePaths.length}
      set newX to ${startX} + ${index} * windowWidth
      set newY to ${startY}
      set newW to newX + windowWidth
      set newH to ${startY} + ${screenHeight}
      set bounds of window ${index + 1} to {newX, newY, newW, newH}
    end tell
    `;

    await runAppleScript(script);
  }
}

