#!/usr/bin/env zx

import { spawn } from 'child_process';
import { runAppleScript } from 'run-applescript';
import fs from 'fs';
import os from 'os';
const cliProgress = require('cli-progress');

$.verbose = false;

// If `--rebalance <period>` is specified, pass it to the Python script
const rebalanceIndex = process.argv.indexOf('--rebalance');
const rebalancePeriod = rebalanceIndex !== -1 ? process.argv[rebalanceIndex + 1] : '';

const runBacktestWithProgressBar = () => new Promise((resolve, reject) => {
  // Create a new progress bar instance
  const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
  progressBar.start(100, 0); // 100% is the max
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
    // Extract the percentage from the string, assuming the format is always [PROGRESS]: X%
    const progressMatch = String(data).match(/\[PROGRESS\]: (\d+)%/);
    if (progressMatch) {
      const progressPercentage = Number(progressMatch[1]);
      progressBar.update(progressPercentage);
    }
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
    await $`killall "Microsoft Excel"`;
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
let filePaths = [];

let globResult = (await $`find ./out/${rebalancePeriod} -name "*.xlsx" -type f -print0 | xargs -0 ls -tu`).stdout.split('\n');
// Reverse the order of files so that the most recent one is opened first
globResult = globResult.reverse().filter((filePath) => !filePath.includes('~$')); // Remove temporary files
console.log(`📂 Found ${globResult.length} files`);
for (let filePath of globResult) {
  if (filePath) {
    filePaths.push(filePath);
    if (os.platform() === 'darwin') {  // Check if it is macOS
      await $`open "${filePath}"`;
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

