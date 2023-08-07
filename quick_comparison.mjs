#!/usr/bin/env zx

import { runAppleScript } from 'run-applescript';

// 1. Force Quit Excel
try {
  await $`killall "Microsoft Excel"`
} catch {}

// 2. Run python script
await $`python run_backtest.py`

// 3. Open all .xlsx files and arrange windows using AppleScript
let filePaths = [];
let globResult = (await $`find ./out -name "*.xlsx"`).stdout.split('\n');
for (let filePath of globResult) {
  if (filePath) {
    filePaths.push(filePath);
    await $`open "${filePath}"`;
  }
}

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
