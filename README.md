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
## Example of Classification
![image](https://github.com/user-attachments/assets/2de5b86b-a019-43a6-b833-c24a5bae3f20)

## Model Files

The football .pt model file is sourced from the SportVision repository.
The team classification logic and main analytical approach are adapted from the Roboflow Sport repository https://github.com/roboflow/sports.

And the kabbadi .pt models are fine-tuned on kabbadi dataset.


## 2D Map of the raider
![vlcsnap-2024-12-26-23h07m53s487](https://github.com/user-attachments/assets/842b2428-1b19-4113-af97-2023ba9e161a)
![vlcsnap-2024-12-28-13h59m05s776](https://github.com/user-attachments/assets/25b1555d-7694-4194-ad3e-ce4a45e5ebca)

## Time spend on the opponent court
![vlcsnap-2024-12-28-14h01m20s028](https://github.com/user-attachments/assets/c830a5a5-03a2-4629-9dc0-9c7092351ff6)
