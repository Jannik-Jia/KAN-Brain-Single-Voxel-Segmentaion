#!/bin/bash

# --- Script Configuration ---
PYTHON_SCRIPT_NAME="comparison_experiment.py"
LOG_FILE="comparison_experiment_run.log"
PYTHON_EXECUTABLE="python" # Or "python3" or path to specific python

# --- Optional: Activate Conda Environment ---
# If you use Conda, uncomment and set your environment name
# CONDA_ENV_NAME="your_pytorch_env_name"
#
# if command -v conda &> /dev/null; then
#     echo "Attempting to activate Conda environment: $CONDA_ENV_NAME..."
#     # Source conda.sh to make 'conda activate' available in scripts
#     CONDA_BASE=$(conda info --base)
#     if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
#         source "$CONDA_BASE/etc/profile.d/conda.sh"
#         conda activate "$CONDA_ENV_NAME"
#         if [ $? -ne 0 ]; then
#             echo "Failed to activate Conda environment: $CONDA_ENV_NAME. Ensure it exists."
#             # exit 1 # Decide if you want to exit if env activation fails
#         else
#             echo "Conda environment '$CONDA_ENV_NAME' activated."
#         fi
#     else
#         echo "conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh. Cannot activate environment."
#     fi
# else
#     echo "Conda command not found. Proceeding with default Python environment."
# fi

# --- Set Environment Variables ---
export PYTHONUNBUFFERED=1 # Recommended for nohup so output appears in log file more readily

# --- Directory Setup ---
# Assuming this run.sh script is in the same directory as the Python script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/$PYTHON_SCRIPT_NAME"

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_PATH' not found!"
    exit 1
fi

# --- Execution with nohup ---
echo "Starting Python script '$PYTHON_SCRIPT_NAME' with nohup..."
echo "Output will be redirected to '$LOG_FILE'."
echo "Results and plots will be in the output directory defined within the Python script (usually './comparison_results_batch')."
echo "You can monitor the log with: tail -f $LOG_FILE"

# Change to the script's directory so that relative paths inside
# the Python script (like for saving results) work as expected.
cd "$SCRIPT_DIR"

nohup $PYTHON_EXECUTABLE -u "$PYTHON_SCRIPT_NAME" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID."
echo "To check status: ps -p $PID"
echo "To view live output: tail -f $LOG_FILE"
echo "To stop the process: kill $PID"
echo "-----------------------------------------------------"

exit 0