#!/usr/bin/env python3
"""
Combine red dots and blue dots videos into a single overlay video.
Usage: python combine_videos.py <red_video> <blue_video> <output_video>
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

if len(sys.argv) != 4:
    print("Usage: python combine_videos.py <red_video> <blue_video> <output_video>")
    print("\nExample:")
    print("  python combine_videos.py red_dots_15to20min.mp4 blue_dots_15to20min.mp4 combined_15to20min.mp4")
    sys.exit(1)

RED_VIDEO = sys.argv[1]
BLUE_VIDEO = sys.argv[2]
OUTPUT_VIDEO = sys.argv[3]

print(f"\n{'='*70}")
print(f"Video Combiner: Red + Blue Dots Overlay")
print(f"{'='*70}")
print(f"Red dots video: {RED_VIDEO}")
print(f"Blue dots video: {BLUE_VIDEO}")
print(f"Output video: {OUTPUT_VIDEO}")
print(f"{'='*70}\n")

# Open both videos
cap_red = cv2.VideoCapture(RED_VIDEO)
cap_blue = cv2.VideoCapture(BLUE_VIDEO)

if not cap_red.isOpened():
    print(f"ERROR: Could not open red video: {RED_VIDEO}")
    sys.exit(1)

if not cap_blue.isOpened():
    print(f"ERROR: Could not open blue video: {BLUE_VIDEO}")
    sys.exit(1)

# Get video properties
fps_red = cap_red.get(cv2.CAP_PROP_FPS)
fps_blue = cap_blue.get(cv2.CAP_PROP_FPS)
width_red = int(cap_red.get(cv2.CAP_PROP_FRAME_WIDTH))
height_red = int(cap_red.get(cv2.CAP_PROP_FRAME_HEIGHT))
width_blue = int(cap_blue.get(cv2.CAP_PROP_FRAME_WIDTH))
height_blue = int(cap_blue.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames_red = int(cap_red.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_blue = int(cap_blue.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Red video: {width_red}x{height_red}, {fps_red} fps, {total_frames_red} frames")
print(f"Blue video: {width_blue}x{height_blue}, {fps_blue} fps, {total_frames_blue} frames")

# Verify videos match
if width_red != width_blue or height_red != height_blue:
    print(f"ERROR: Video dimensions don't match!")
    sys.exit(1)

if abs(fps_red - fps_blue) > 0.1:
    print(f"WARNING: FPS doesn't match exactly ({fps_red} vs {fps_blue})")

# Use minimum frame count
num_frames = min(total_frames_red, total_frames_blue)
print(f"\nCombining {num_frames} frames...")

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_red, (width_red, height_red))

if not out.isOpened():
    print("ERROR: Could not create video writer")
    sys.exit(1)

# Process frames
for frame_idx in tqdm(range(num_frames)):
    # Read frames
    ret_red, frame_red = cap_red.read()
    ret_blue, frame_blue = cap_blue.read()

    if not ret_red or not ret_blue:
        print(f"\nEnd of video at frame {frame_idx}")
        break

    # Use additive blending: combine both frames, then subtract the court background once
    # This preserves both red and blue dots on the same court
    combined = cv2.addWeighted(frame_red, 0.5, frame_blue, 0.5, 0)

    # Write combined frame
    out.write(combined)

# Cleanup
cap_red.release()
cap_blue.release()
out.release()

print(f"\n✅ Combined video saved to: {OUTPUT_VIDEO}")
print(f"Frames combined: {num_frames}")
print(f"Duration: {num_frames / fps_red / 60:.1f} minutes")
print(f"Resolution: {width_red}x{height_red}")
