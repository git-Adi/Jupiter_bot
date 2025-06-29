const { execSync } = require('child_process');
const fs = require('fs');

console.log('Starting Vercel build process...');

// Ensure we're using Python 3.9
try {
  console.log('Setting Python version to 3.9...');
  execSync('python3.9 --version', { stdio: 'inherit' });
} catch (error) {
  console.error('Python 3.9 is required but not found. Please install it first.');
  process.exit(1);
}

// Install dependencies
console.log('Installing Python dependencies...');
try {
  execSync('python3.9 -m pip install --upgrade pip', { stdio: 'inherit' });
  execSync('python3.9 -m pip install -r requirements-vercel.txt', { stdio: 'inherit' });
  console.log('Dependencies installed successfully!');
} catch (error) {
  console.error('Failed to install dependencies:', error);
  process.exit(1);
}

console.log('Build completed successfully!');
