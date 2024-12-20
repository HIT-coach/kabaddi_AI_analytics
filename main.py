import argparse
from enum import Enum
from typing import Iterator, List
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from team import TeamClassifier
from collections import deque
import matplotlib.colors as mcolors

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection-v2.pt')

# Class IDs - Update these based on Kabbadi's requirements
PLAYER_CLASS_ID = 0
REFEREE_CLASS_ID = 1
BALL_CLASS_ID = 2

STRIDE = 60

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#FFFFFF']

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Kabbadi AI video analysis.
    """
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def assign_kabaddi_roles(frame, players, players_team_id):
    """
    Assigns roles to players in a Kabaddi match based on the team and count of players identified.
    """
    unique_teams, counts = np.unique(players_team_id, return_counts=True)
    team_counts = dict(zip(unique_teams, counts))

    # Identify if there's a team with exactly one player
    attacker_team = None
    for t_id, count in team_counts.items():
        if count == 1:
            attacker_team = t_id
            break

    # If we found a team with a single player, designate that as attacker
    if attacker_team is not None and len(team_counts) == 2:
        all_teams = list(team_counts.keys())
        other_team = [t for t in all_teams if t != attacker_team][0]
        corrected_team_ids = []
        for t_id in players_team_id:
            if t_id == attacker_team:
                # This single player is the attacker
                corrected_team_ids.append(attacker_team)
            else:
                # All players of the other team are defenders
                corrected_team_ids.append(other_team)
        players_team_id = np.array(corrected_team_ids)

    return players_team_id


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

    # Step 1: Collect initial crops for team fitting
    crops = []
    for frame in tqdm(frame_generator, desc='Collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops += get_crops(frame, players)

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # Reinitialize the frame generator without stride for full analysis
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps if video_info.fps else 30.0
    time_per_frame = 1.0 / fps

    # Set up ByteTrack and dictionaries to store player states
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    player_positions_in_meters = {}
    player_speeds = {}
    # Store recent speeds for each player (for moving average)
    player_speed_history = {}

    # Moving average configuration
    speed_history_length = 15  # Use the last 5 frames for averaging
    max_human_speed = 10.0  # Maximum plausible speed in m/s

    # Compute pixel-to-meter ratio (assume full frame = pitch size)
    width_pixels = video_info.width
    height_pixels = video_info.height
    meter_per_pixel_x = 11.0 / width_pixels
    meter_per_pixel_y = 8.0 / height_pixels

    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)
        players_team_id = assign_kabaddi_roles(frame, players, players_team_id)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        detections = sv.Detections.merge([players, referees])
        color_lookup = np.array(
            players_team_id.tolist() + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        # Calculate speeds
        for i, det_id in enumerate(detections.tracker_id):
            if det_id is None:
                continue
            # Get bounding box
            x1, y1, x2, y2 = detections.xyxy[i]
            center_x_pixel = (x1 + x2) / 2.0
            center_y_pixel = (y1 + y2) / 2.0
            center_x_meter = center_x_pixel * meter_per_pixel_x
            center_y_meter = center_y_pixel * meter_per_pixel_y

            current_pos = (center_x_meter, center_y_meter)
            if det_id in player_positions_in_meters:
                old_pos = player_positions_in_meters[det_id]
                dx = current_pos[0] - old_pos[0]
                dy = current_pos[1] - old_pos[1]
                distance = np.sqrt(dx*dx + dy*dy)
                speed = distance / time_per_frame

                # Add speed to history
                if det_id not in player_speed_history:
                    player_speed_history[det_id] = deque(maxlen=speed_history_length)
                player_speed_history[det_id].append(speed)

                # Compute moving average
                avg_speed = np.mean(player_speed_history[det_id])
                player_speeds[det_id] = min(avg_speed, max_human_speed)

                # Check for implausible speeds
                if avg_speed > max_human_speed:
                    print(f"Warning: Unusual speed detected for player {det_id}: {avg_speed:.2f} m/s")
            player_positions_in_meters[det_id] = current_pos

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )

        # Annotate moving average speeds
        for i, det_id in enumerate(detections.tracker_id):
            if det_id is None:
                continue
            class_id = detections.class_id[i]
            if class_id == PLAYER_CLASS_ID:
                avg_speed = player_speeds.get(det_id, 0.0)
                x1, y1, x2, y2 = detections.xyxy[i]
                text_pos = (int((x1+x2)/2), int(y1)-10)
                speed_text = f"{avg_speed:.2f} m/s"
                cv2.putText(
                    annotated_frame,
                    speed_text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA
                )

        yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode, json_file_path: str) -> None:
    if mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kabbadi AI Video Analysis')
    parser.add_argument('--source_video_path', type=str, required=True, help='Path to the source video')
    parser.add_argument('--target_video_path', type=str, required=True, help='Path to save the annotated video')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu", "cuda")')
    parser.add_argument('--mode', type=Mode, default=Mode.TEAM_CLASSIFICATION, choices=list(Mode), help='Mode of operation')
    parser.add_argument('--json_file_path', type=str, default='output.json', help='Path to the output JSON file')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        json_file_path=args.json_file_path
    )