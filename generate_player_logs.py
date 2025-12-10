#!/usr/bin/env python3
"""
Generate Player Logs

Scans the video, detects players, transforms coordinates to the UWB/Court system,
and outputs a JSON log file. This data can be used to mathematically synchronize
video timestamps with UWB timestamps by matching movement patterns.

Usage:
    python generate_player_logs.py --video GX010018_1080p.MP4 --output player_logs.json
"""

import sys
import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from app.services.player_detector import PlayerDetector
from app.services.calibration_integration import CalibrationIntegration
from app.services.dxf_parser import parse_court_dxf

def parse_time(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def main():
    parser = argparse.ArgumentParser(description='Generate player coordinate logs from video')
    parser.add_argument('--video', type=str, default='GX010018_1080p.MP4', help='Input video path')
    parser.add_argument('--calibration', type=str, default='data/calibration/1080p/homography.json', help='Calibration file')
    parser.add_argument('--layout', type=str, default='court_2.dxf', help='DXF file for bounds')
    parser.add_argument('--output', type=str, help='Output JSON file (default: matches video name)')
    parser.add_argument('--start', type=str, default='00:00:00', help='Start time HH:MM:SS')
    parser.add_argument('--end', type=str, help='End time HH:MM:SS (optional, defaults to end of video)')
    parser.add_argument('--step', type=int, default=1, help='Process every Nth frame (default: 1)')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video {video_path} not found")
        sys.exit(1)
        
    if args.output:
        output_path = args.output
    else:
        output_path = video_path.stem + "_player_coords.json"
        
    # Initialize components
    print("Initializing Player Detector...")
    player_detector = PlayerDetector(model_name="yolo11n.pt", confidence_threshold=0.3, enable_tracking=True)
    
    print("Loading Calibration...")
    calibration = CalibrationIntegration(args.calibration)
    
    # Get Court Bounds Logic (Optional: for sanity check)
    # Note: we primarily want the RAW court coordinates (cm) that calibration.image_to_court returns.
    # Whether these are X=Long or X=Short depends on calibration.
    # Based on validation_overlap.py success, calibration returns vertical-oriented coords (0..4261, 0..7341).
    # UWB is typically (Long, Short) = (2800, 1500).
    # We will log the RAW calibration coordinates. Post-processing can handle rotation.
    
    # Video Setup
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_sec = parse_time(args.start)
    start_frame = int(start_sec * fps)
    
    if args.end:
        end_sec = parse_time(args.end)
        end_frame = int(end_sec * fps)
    else:
        end_frame = total_frames
        
    num_frames = end_frame - start_frame
    
    print(f"\nProcessing Video: {video_path}")
    print(f"Time Range: {args.start} -> {args.end if args.end else 'End'}")
    print(f"Frames: {start_frame} -> {end_frame} ({num_frames} frames)")
    print(f"Step: Every {args.step} frames")
    print(f"Output: {output_path}")
    
    # Data structure
    logs = {
        "metadata": {
            "video": str(video_path),
            "fps": fps,
            "start_time": args.start,
            "start_frame": start_frame,
            "step": args.step,
            "coord_system": "calibration_raw_cm" 
        },
        "frames": []
    }
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Use tqdm for progress
    pbar = tqdm(total=num_frames)
    
    frame_idx = start_frame
    processed_count = 0
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame = frame_idx
        
        # Skip frames if step > 1
        # But we need to read to advance iterator (slow) or use set (slow).
        # Actually logic above reads every frame.
        # If Step > 1, we should read and skip processing.
        
        should_process = (current_frame - start_frame) % args.step == 0
        
        if should_process:
            # Detect
            tracks = player_detector.track_players(frame)
            
            frame_entry = {
                "frame": current_frame,
                "time_sec": current_frame / fps,
                "players": []
            }
            
            for track in tracks:
                track_id = track.get('track_id', -1)
                bx, by = track['bottom']
                
                try:
                    # Project to Court (CM)
                    cx, cy = calibration.image_to_court(bx, by)
                    
                    # Store
                    frame_entry["players"].append({
                        "id": track_id,
                        "x": float(cx),
                        "y": float(cy)
                    })
                except:
                    pass
            
            logs["frames"].append(frame_entry)
            processed_count += 1
            
        frame_idx += 1
        pbar.update(1)
        
    pbar.close()
    
    # Save JSON
    print(f"\nSaving {len(logs['frames'])} entries to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(logs, f, indent=2) # indent=None for size? User asked for logs, indent is readable.
        
    print("Done.")

if __name__ == "__main__":
    main()
