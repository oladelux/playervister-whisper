#!/bin/bash

# Football Whisper Project Cleanup Script
# This script automatically cleans up and organizes the project structure

echo "===== Cleaning up Football Whisper project ====="

# 1. Remove unnecessary files
echo "Removing unnecessary files..."
rm -f simple_train.py simple_inference.py debug_preparation.py

# 2. Remove macOS system files
echo "Removing system files..."
find . -name ".DS_Store" -delete
find . -name "._*" -delete

# 3. Remove Python cache
echo "Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete

# 4. Remove temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -delete
find . -name "*.temp" -delete
find . -name "*.swp" -delete
find . -name "*.swo" -delete
find . -name "*.log" -delete

# 5. Create directory structure
echo "Creating directory structure..."
mkdir -p models
mkdir -p data/processed
mkdir -p data/audio
mkdir -p data/transcripts
mkdir -p evaluation_results

# 6. Make src a proper Python package
echo "Setting up source package..."
touch src/__init__.py

# 7. Create .gitignore file
echo "Creating .gitignore file..."
cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
soccer_whisper_env/

# Models and data
models/*/
!models/.gitkeep
data/processed/
evaluation_results/

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Editor directories and files
.idea
.vscode
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints
EOL

# 8. Create empty .gitkeep files to preserve empty directories
echo "Adding .gitkeep files to preserve directory structure..."
touch models/.gitkeep
touch data/processed/.gitkeep
touch evaluation_results/.gitkeep

# 9. Make scripts executable
echo "Setting file permissions..."
chmod +x train_and_evaluate.sh
chmod +x cleanup_project.sh

# 10. Clean up virtual environment if it exists in the project
if [ -d "soccer_whisper_env" ]; then
    echo "Removing virtual environment from project directory..."
    rm -rf soccer_whisper_env
fi

echo "===== Cleanup complete! ====="
echo "The project structure is now clean and organized."
echo "Run './train_and_evaluate.sh' to start the training pipeline." 