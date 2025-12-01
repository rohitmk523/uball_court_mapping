#!/bin/bash
# Start the Basketball Court Tracking server

# Activate conda environment
eval "$(/Users/rohitkale/miniconda3/bin/conda shell.bash hook)"
conda activate court_tracking

# Start FastAPI server
echo "Starting Basketball Court Tracking Server..."
echo "Server will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
