#!/bin/bash
set -x
LOG_FILE="server.log"

# Activate virtual environment
source venv/bin/activate

# Enable debug logging
export LOG_LEVEL=DEBUG

    # Truncate log file
    : > "$LOG_FILE"
    
    # Run server with reload and log redirection
    uvicorn main:app --host 0.0.0.0 --port 4000 --reload 2>&1 | tee "$LOG_FILE"

    
