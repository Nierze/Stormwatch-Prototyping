# Surface Estimation on CPU

This minimizes the Metric3D project to run **only surface normal estimation** on CPU.

## Prerequisites

- Linux OS.
- Python 3 with `venv` (Virtual Environment provided in `./venv`).

## Dependencies

All dependencies are pre-installed in the local virtual environment `./venv`.

## Project Structure

| File/Directory | Description |
| :--- | :--- |
| `run_model_cpu.py` | The main Python script adapted for CPU inference. |
| `run_cpu.sh` | **Start here.** A helper bash script to easily run the inference. |
| `mono/` | The core model library code. |
| `weight/` | Contains the pre-trained model checkpoint (`metric_depth_vit_small_800k.pth`). |
| `new test images/` | **Put your images here.** This is the default input directory. |
| `output/` | **Check here for results.** This directory is created automatically. |

## How to Run

### Step 1: Prepare Input Images
Place the `.jpg` or `.png` images you want to analyze into the **`new test images`** folder.

### Step 2: Run the Script
We have provided a convenient shell script `run_cpu.sh` that handles the virtual environment and arguments for you.

#### Run on all images in a folder (Default)
1.  Open your terminal.
2.  Navigate to this project folder.
3.  Run:
    ```bash
    ./run_cpu.sh
    ```

#### Run on just one image
To run on a specific image folder or file, pass the path as an argument. The script now handles paths with spaces even without quotes:
```bash
./run_cpu.sh new test images/1_flood.png
```

*Note: The model runs on CPU, so it may take a moment per image.*

### Step 3: View Results
Once finished, check the **`output/vis/new test images`** directory.
*   The script outputs **only** the surface normal prediction images.
*   Files are named `normal_<original_name>_normal.jpg`.

## Advanced Usage

To run manually with custom paths:

```bash
source ./venv/bin/activate
python run_model_cpu.py \
    'mono/configs/HourglassDecoder/vit.raft5.small.py' \
    --load-from ./weight/metric_depth_vit_small_800k.pth \
    --test_data_path "new test images" \
    --launcher None \
    --show-dir ./output
```
