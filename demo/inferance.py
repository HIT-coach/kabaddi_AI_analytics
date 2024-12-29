from ultralytics import YOLO
import cv2
import numpy as np
import math
from collections import defaultdict, deque
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def calculate_velocity(positions, timestamps, pixels_per_meter):
    """Calculate instantaneous velocity using the most recent positions"""
    if len(positions) < 2:
        return 0.0

    pos1, pos2 = positions[-2], positions[-1]
    t1, t2 = timestamps[-2], timestamps[-1]

    # Calculate distance in pixels
    distance_px = calculate_distance(pos1, pos2)

    # Convert to meters
    distance_m = distance_px / pixels_per_meter

    # Calculate time difference
    time_diff = t2 - t1

    if time_diff == 0:
        return 0.0

    velocity = distance_m / time_diff

    VELOCITY_THRESHOLD = 0.2
    MAX_VELOCITY = 10.0  # Maximum reasonable velocity

    if velocity < VELOCITY_THRESHOLD:
        return 0.0
    elif velocity > MAX_VELOCITY:
        return MAX_VELOCITY

    return velocity


def get_bottom_center_point(box):
    x1, y1, x2, y2 = box.xyxy[0].astype(int)
    return (int((x1 + x2) / 2), y2)


def get_center_point(box):
    x1, y1, x2, y2 = box.xyxy[0].astype(int)
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def calculate_scale(points):
    COURT_WIDTH = 10.0
    COURT_HEIGHT = 13.0

    dist_1_2_px = calculate_distance(points["point1"], points["point2"])
    dist_3_4_px = calculate_distance(points["point3"], points["point4"])
    avg_width_pixels = (dist_1_2_px + dist_3_4_px) / 2

    dist_1_3_px = calculate_distance(points["point1"], points["point3"])
    dist_2_4_px = calculate_distance(points["point2"], points["point4"])
    avg_height_pixels = (dist_1_3_px + dist_2_4_px) / 2

    width_scale = avg_width_pixels / COURT_WIDTH
    height_scale = avg_height_pixels / COURT_HEIGHT

    pixels_per_meter = width_scale * 0.6 + height_scale * 0.4

    return pixels_per_meter


def calculate_center_line(points):
    if all(p in points for p in ["point1", "point2", "point3", "point4"]):
        top_mid = (
            (points["point1"][0] + points["point2"][0]) // 2,
            (points["point1"][1] + points["point2"][1]) // 2,
        )
        bottom_mid = (
            (points["point3"][0] + points["point4"][0]) // 2,
            (points["point3"][1] + points["point4"][1]) // 2,
        )
        return top_mid, bottom_mid
    return None, None


def point_to_line_distance(point, line_start, line_end, pixels_per_meter):
    if line_start is None or line_end is None:
        return None

    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if denominator == 0:
        return 0

    distance_px = numerator / denominator
    return distance_px / pixels_per_meter if pixels_per_meter else 0


def create_gradient_background(width, height):
    """Create a horizontal gradient from grey to white"""
    # Create an empty canvas with three color channels (RGB)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(width):
        grey_value = int(150 + (i / width * 1.5) * (255 - 150))  # 169 grey, 255 white
        gradient[:, i] = [grey_value, grey_value, grey_value]

    return gradient


def plot_graphs(player_data):
    for team, data in player_data.items():
        # Time vs Distance
        plt.figure(figsize=(10, 5))
        plt.plot(
            data["time"],
            data["distance"],
            label=f"{team} - Distance",
            color="b",
            linewidth=3,
            marker="o",
            markersize=8,
        )
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Distance (m)", fontsize=14)
        plt.title(f"{team} - Time vs Distance", fontsize=16)
        plt.legend(fontsize=12, loc="best")
        plt.grid(True, linestyle="--", linewidth=0.8)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(
            data["time"],
            data["velocity"],
            label=f"{team} - Velocity",
            color="r",
            linewidth=3,
            marker="o",
            markersize=8,
        )
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Velocity (m/s)", fontsize=14)
        plt.title(f"{team} - Time vs Velocity", fontsize=16)
        plt.legend(fontsize=12, loc="best")
        plt.grid(True, linestyle="--", linewidth=0.8)
        plt.tight_layout()
        plt.show()


def create_canvas(video_frame, graphs):
    frame_height, frame_width = video_frame.shape[:2]

    canvas_width = int(frame_width * 4 / 3)
    canvas_height = frame_height

    canvas = create_gradient_background(canvas_width, canvas_height)

    canvas[:, :frame_width] = video_frame

    graphs_width = canvas_width - frame_width
    graphs_height = canvas_height

    if graphs:
        graph_height = graphs_height // len(graphs)

        for i, graph_path in enumerate(graphs):
            graph_img = cv2.imread(graph_path, cv2.IMREAD_UNCHANGED)
            if graph_img is not None:
                graph_resized = cv2.resize(graph_img, (graphs_width, graph_height))

                y_start = i * graph_height
                y_end = y_start + graph_height
                x_start = frame_width
                x_end = canvas_width

                if graph_resized.shape[2] == 4:
                    alpha_s = graph_resized[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s

                    for c in range(0, 3):
                        canvas[y_start:y_end, x_start:x_end, c] = (
                            alpha_s * graph_resized[:, :, c]
                            + alpha_l * canvas[y_start:y_end, x_start:x_end, c]
                        )
                else:
                    canvas[y_start:y_end, x_start:x_end] = graph_resized[:, :, :3]

    return canvas


def save_graph_as_image(data, graph_type, team, save_path):
    plt.figure(figsize=(4, 3), dpi=100)
    plt.rcParams["axes.facecolor"] = "none"

    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["axes.titlecolor"] = "white"
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.labelcolor"] = "white"

    if graph_type == "distance":
        plt.plot(data["time"], data["distance"], label=f"{team} - Distance", color="b")
        plt.title(
            f"{team} - Distance", color="white", fontdict={"size": 10, "weight": "bold"}
        )
    elif graph_type == "velocity":
        plt.plot(data["time"], data["velocity"], label=f"{team} - Velocity", color="r")
        plt.title(
            f"{team} - Velocity", color="white", fontdict={"size": 10, "weight": "bold"}
        )

    plt.xlabel("Time (s)", color="white")

    plt.legend(
        fontsize=8, loc="best", frameon=False, handlelength=2, labelcolor="white"
    )
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, transparent=True)
    plt.close()


def overlay_graph_on_frame(frame, graph_image_path, position=(10, 10)):
    graph_img = cv2.imread(graph_image_path, cv2.IMREAD_UNCHANGED)
    if graph_img is not None:
        graph_width = int(frame.shape[1] * 0.3)
        aspect_ratio = graph_img.shape[0] / graph_img.shape[1]
        graph_height = int(graph_width * aspect_ratio)

        graph_img_resized = cv2.resize(graph_img, (graph_width, graph_height))

        y1, y2 = position[1], position[1] + graph_height
        x1, x2 = position[0], position[0] + graph_width

        if y2 > frame.shape[0]:
            y2 = frame.shape[0]
            graph_height = y2 - y1
            graph_img_resized = cv2.resize(graph_img, (graph_width, graph_height))

        if x2 > frame.shape[1]:
            x2 = frame.shape[1]
            graph_width = x2 - x1
            graph_img_resized = cv2.resize(graph_img, (graph_width, graph_height))

        alpha_s = graph_img_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (
                alpha_s * graph_img_resized[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c]
            )

    return frame


def draw_player_metrics(
    frame, x1, y1, x2, y2, duration, center_dist, velocity, team_color
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    padding = 8

    center_x = (x1 + x2) // 2

    time_text = f" {duration:.1f}s"
    dist_text = f"{center_dist:.1f}m"
    vel_text = f"{velocity:.1f}m/s"

    (time_w, time_h), _ = cv2.getTextSize(time_text, font, font_scale, thickness)
    (dist_w, dist_h), _ = cv2.getTextSize(dist_text, font, font_scale, thickness)
    (vel_w, vel_h), _ = cv2.getTextSize(vel_text, font, font_scale, thickness)

    box_width = max(time_w, dist_w, vel_w) + 2 * padding
    box_height = (time_h + dist_h + vel_h) + 4 * padding

    box_x = center_x - (box_width // 2)
    box_y = y1 - box_height - 15

    box_x = max(0, min(box_x, frame.shape[1] - box_width))
    box_y = max(0, min(box_y, frame.shape[0] - box_height))

    overlay = frame.copy()

    # # Draw main background
    # cv2.rectangle(
    #     overlay,
    #     (box_x, box_y),
    #     (box_x + box_width, box_y + box_height),
    #     (40, 40, 40),
    #     -1,
    # )

    # cv2.rectangle(
    #     overlay,
    #     (box_x, box_y),
    #     (box_x + box_width, box_y + box_height),
    #     team_color,
    #     2,
    # )

    # Blend overlay with main frame
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    time_color = (
        (0, 255, 0)
        if duration < 20
        else ((0, 0, 255) if duration > 30 else (0, 255, 255))
    )

    text_x = box_x + padding
    text_y = box_y + padding + time_h

    # Draw duration
    cv2.putText(
        frame,
        time_text,
        (text_x, text_y),
        font,
        font_scale,
        time_color,
        thickness,
    )

    # Draw distance
    text_y += dist_h + padding
    cv2.putText(
        frame,
        dist_text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    text_y += vel_h + padding
    vel_color = (
        int(min(velocity * 25.5, 255)),  # R increases with speed
        int(max(255 - velocity * 25.5, 0)),  # G decreases with speed
        0,  # B remains 0
    )
    cv2.putText(
        frame,
        vel_text,
        (text_x, text_y),
        font,
        font_scale,
        vel_color,
        thickness,
    )


def is_player_in_opposite_court(player_point, top_mid, bottom_mid):
    player_y = player_point[1]

    if player_y > top_mid[1]:
        print("inside the court")
        return True

    return False


def add_timer_to_frame(
    frame,
    elapsed_time,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1,
    thickness=2,
):
    if elapsed_time > 60:
        elapsed_time = 60

    remaining_time = 60 - elapsed_time
    text = f"{remaining_time:.2f}s"

    if remaining_time <= 20:
        color = (0, 0, 255)  # Red
    elif remaining_time <= 30:
        color = (0, 255, 255)  # Yellow
    else:
        color = (255, 255, 255)  # White

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = (frame.shape[1] - text_width) // 2
    text_y = 30

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    return frame


def detect_and_track(video_path, output_path=None):
    model = YOLO("../kabaddi_AI_analytics/data/best_yolov8_kabbadi.pt")
    class_names = ["team_1", "team_2", "center", "point1", "point2", "point3", "point4"]
    colors = {
        "point1": (255, 255, 0),
        "point2": (255, 0, 255),
        "point3": (0, 255, 255),
        "point4": (255, 255, 255),
        "team_1": (255, 0, 0),
        "team_2": (0, 0, 255),
    }

    player_tracks = {
        "team_1": defaultdict(
            lambda: {"positions": deque(maxlen=5), "timestamps": deque(maxlen=5)}
        ),
        "team_2": defaultdict(
            lambda: {"positions": deque(maxlen=5), "timestamps": deque(maxlen=5)}
        ),
    }

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    DETECTION_THRESHOLD_FRAMES = int(fps)
    frame_center_x = width // 2

    if output_path:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 4 // 3, height)
        )

    frame_count = 0
    player_data = {
        "team_1": {"time": [], "distance": [], "velocity": []},
        "team_2": {"time": [], "distance": [], "velocity": []},
    }

    team1_detected = False
    team2_detected = False
    team1_start_time = None
    team2_start_time = None
    team1_timer_active = False
    team2_timer_active = False
    team1_consistency_counter = 0
    team2_consistency_counter = 0
    detection_threshold_frames = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        frame_count += 1

        # Run detection
        results = model(frame, conf=0.3)
        points = {}
        players = {"team_1": [], "team_2": []}

        current_team1_detected = False
        current_team2_detected = False

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                confidence = box.conf[0]

                if class_name.startswith("point"):
                    center = get_center_point(box)
                    points[class_name] = center
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_name], 2)
                    cv2.circle(frame, center, 4, colors[class_name], -1)

                elif class_name in ["team_1", "team_2"]:
                    if class_name == "team_1":
                        current_team1_detected = True
                        if not team1_timer_active:
                            team1_start_time = frame_count / fps
                            team1_timer_active = True
                    if class_name == "team_2":
                        current_team2_detected = True
                        if not team2_timer_active:
                            team2_start_time = frame_count / fps
                            team2_timer_active = True

                    bottom_center = get_bottom_center_point(box)
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    track_id = id(box)

                    duration = 0.0
                    if class_name == "team_1" and team1_start_time is not None:
                        duration = (frame_count / fps) - team1_start_time
                    elif class_name == "team_2" and team2_start_time is not None:
                        duration = (frame_count / fps) - team2_start_time

                    player_tracks[class_name][track_id]["positions"].append(
                        bottom_center
                    )
                    player_tracks[class_name][track_id]["timestamps"].append(timestamp)
                    players[class_name].append(
                        (
                            bottom_center,
                            (x1, y1, x2, y2),
                            confidence,
                            track_id,
                            duration,
                        )
                    )

        team1_detected = current_team1_detected
        team2_detected = current_team2_detected
        if current_team1_detected:
            team1_consistency_counter += 1
        else:
            team1_consistency_counter = 0

        if current_team2_detected:
            team2_consistency_counter += 1
        else:
            team2_consistency_counter = 0

        team1_detected = team1_consistency_counter >= detection_threshold_frames
        team2_detected = team2_consistency_counter >= detection_threshold_frames

        if not team1_detected and team1_timer_active:
            team1_timer_active = False
            team1_start_time = None
            player_data["team_1"] = {"time": [], "distance": [], "velocity": []}

        if not team2_detected and team2_timer_active:
            team2_timer_active = False
            team2_start_time = None
            player_data["team_2"] = {"time": [], "distance": [], "velocity": []}

        if len(points) == 4:
            pixels_per_meter = calculate_scale(points)
            top_mid, bottom_mid = calculate_center_line(points)

            for team in ["team_1", "team_2"]:
                for player_point, bbox, confidence, track_id, duration in players[team]:
                    x1, y1, x2, y2 = bbox

                    # Calculate metrics for court
                    center_dist = point_to_line_distance(
                        player_point, top_mid, bottom_mid, pixels_per_meter
                    )
                    velocity = calculate_velocity(
                        list(player_tracks[team][track_id]["positions"]),
                        list(player_tracks[team][track_id]["timestamps"]),
                        pixels_per_meter,
                    )

                    draw_player_metrics(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        duration,
                        center_dist,
                        velocity,
                        colors[team],
                    )

                    # Update player data only if the team is detected
                    if team1_detected and team == "team_1":
                        player_data[team]["time"].append(frame_count / fps)
                        player_data[team]["distance"].append(center_dist)
                        player_data[team]["velocity"].append(velocity)

                    if team2_detected and team == "team_2":
                        player_data[team]["time"].append(frame_count / fps)
                        player_data[team]["distance"].append(center_dist)
                        player_data[team]["velocity"].append(velocity)

        active_graphs = []
        if team1_detected:
            save_graph_as_image(
                player_data["team_1"], "distance", "team_1", "team_1_distance.png"
            )
            save_graph_as_image(
                player_data["team_1"], "velocity", "team_1", "team_1_velocity.png"
            )
            active_graphs.extend(["team_1_distance.png", "team_1_velocity.png"])

        if team2_detected:
            save_graph_as_image(
                player_data["team_2"], "distance", "team_2", "team_2_distance.png"
            )
            save_graph_as_image(
                player_data["team_2"], "velocity", "team_2", "team_2_velocity.png"
            )
            active_graphs.extend(["team_2_distance.png", "team_2_velocity.png"])

        # Create the combined canvas
        canvas = create_canvas(frame, active_graphs)

        if team1_timer_active and is_player_in_opposite_court(
            player_point, top_mid, bottom_mid
        ):
            elapsed_time_team1 = (frame_count / fps) - team1_start_time
            canvas = add_timer_to_frame(canvas, elapsed_time_team1)
        if team2_timer_active and is_player_in_opposite_court(
            player_point, top_mid, bottom_mid
        ):
            elapsed_time_team2 = (frame_count / fps) - team2_start_time
            canvas = add_timer_to_frame(canvas, elapsed_time_team2)

        if output_path:
            writer.write(canvas)

        cv2.imshow("Player Tracking", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if output_path:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"M:\VS_code\Kabaddi_analytics\clip_40.mp4"
    output_path = "output_tracking_.mp4"
    detect_and_track(video_path, output_path)
