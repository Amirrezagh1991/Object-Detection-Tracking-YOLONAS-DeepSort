import numpy as np
import datetime
import cv2
import torch
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
from super_gradients.training import models


def main(args):
    # Initialize the data capture and the data writer objects
    video_cap = cv2.VideoCapture(args.video)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # Initialize the data writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)

    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load the YOLO model
    model = models.get(args.model, pretrained_weights="coco").to(device)

    # Load the COCO class labels the YOLO model was trained on
    classes_path = "./configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # Create a list of random colors to represent each class
    np.random.seed(42)  # to get the same colors
    colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

    while True:
        start = datetime.datetime.now()         # Start time to compute the FPS
        ret, frame = video_cap.read()          # Read a frame from the data

        # If there is no frame, we have reached the end of the data
        if not ret:
            print("End of the data file...")
            break

        # Perform object detection using the YOLO model on the current frame
        detect = next(iter(model.predict(frame, iou=0.5, conf=args.conf)))

        # Extract the bounding box coordinates, confidence scores, and class labels from the detection results
        bboxes_xyxy = torch.from_numpy(detect.prediction.bboxes_xyxy).tolist()
        confidence = torch.from_numpy(detect.prediction.confidence).tolist()
        labels = torch.from_numpy(detect.prediction.labels).tolist()
        # Combine the bounding box coordinates and confidence scores into a single list
        concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
        # Combine the concatenated list with the class labels into a final prediction list
        final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]

        # Initialize the list of bounding boxes and confidences
        results = []

        # Loop over the detections
        for data in final_prediction:
            # Extract the confidence (i.e., probability) associated with the detection
            confidence = data[4]

            # Filter out weak detections by ensuring the confidence is greater than the minimum confidence and with the class_id
            if args.class_id == None:
                if float(confidence) < args.conf:
                    continue
            else:
                if ((int(data[5] != args.class_id)) or (float(confidence) < args.conf)):
                    continue

            # If the confidence is greater than the minimum confidence, draw the bounding box on the frame
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])

            # Add the bounding box (x, y, w, h), confidence, and class ID to the results list
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        # Update the tracker with the new detections
        tracks = tracker.update_tracks(results, frame=frame)

        # Loop over the tracks
        for track in tracks:
            # If the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            # Get the track ID and the bounding box
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Get the color for the class
            color = colors[class_id]
            B, G, R = int(color[0]), int(color[1]), int(color[2])

            # Create text for track ID and class name
            text = str(track_id) + " - " + str(class_names[class_id])

            # Draw bounding box and text on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # End time to compute the FPS
        end = datetime.datetime.now()

        # Show the time it took to process 1 frame
        print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")

        # Calculate the frames per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # Show the frame
        cv2.imshow("Frame", frame)

        # Write the frame to the output data file
        writer.write(frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release data capture and data writer objects
    video_cap.release()
    writer.release()

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()

        # Add command line arguments
        parser.add_argument('--model', type=str, default='yolo_nas_l', help='yolo_nas_l or yolo_nas_m or yolo_nas_s')
        parser.add_argument('--video', type=str, default='./data/video/test.mp4',
                            help='path to input data or set to 0 for webcam')
        parser.add_argument('--output', type=str, default='./output/output.mp4', help='path to output data')
        parser.add_argument('--conf', type=float, default=0.50, help='confidence threshold')
        parser.add_argument('--class_id', type=int, default=None,
                            help='class id 0 for person, check coco.names for others')

        # Parse the command line arguments
        args = parser.parse_args()
        main(args)
    except SystemExit:
        pass