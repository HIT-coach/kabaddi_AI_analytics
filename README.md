# kabaddi_AI_analytics

This repository provides tools for analyzing Kabaddi videos using AI-based techniques. It includes functionalities for team classification and overall analytics derived from the input footage.

## Getting Started

**Prerequisites:**
- Required Python packages (install via `requirements.txt`)
- Input video files in `data/videos/` directory

**Running the Tool:**
Use the following command to process a video and perform team classification:

```bash
python main.py \
    --source_video_path data/videos/kabedi.mp4 \
    --target_video_path data/videos/results/kabadi_teamclass.mp4 \
    --device cpu \
    --mode TEAM_CLASSIFICATION
```

## Model Files

The .pt model file is sourced from the SportVision repository.
The team classification logic and main analytical approach are adapted from the Roboflow Sport repository https://github.com/roboflow/sports.

Ensure that the .pt model file is placed or referenced correctly so that the main.py script can access it.