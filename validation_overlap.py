#!/usr/bin/env python3
"""
Validation Overlap Tool (Refined Vertical)

Creates a 2-panel validation video:
- Left: Original video with YOLO detection boxes and Tag IDs.
- Right: Court view (Vertical) with:
    - BLUE dots: UWB Tags (with 200cm proximity circle)
    - RED dots: Player detections (outside proximity)
    - GREEN dots: Player detections (inside proximity / associated)
      * Closest player logic: If multiple players in circle, closest gets tag.

Usage:
    python validation_overlap.py <start_time> <end_time> [sync_offset] [video_path]
"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from bisect import bisect_left
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from app.services.player_detector import PlayerDetector
from app.services.calibration_integration import CalibrationIntegration
from app.services.dxf_parser import parse_court_dxf
from app.services.uwb_associator import UWBAssociator

# Configuration
VIDEO_PATH = "GX020018_1080p.MP4"
SYNC_ANALYSIS_FILE = "cluster_sync_analysis.json"
COURT_IMAGE_PATH = "data/calibration/court_image.png"
COURT_DXF_PATH = "court_2.dxf"

# Output Video Height (Fixed)
OUTPUT_HEIGHT = 1080

# Colors (BGR)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255) # Warning/Dropout
COLOR_BLUE = (255, 100, 0) # UWB Dot
COLOR_LIGHT_BLUE = (255, 200, 0) # Radius Circle
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

def time_to_seconds(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

class Validator:
    def __init__(self, video_path, calibration_file="data/calibration/1080p/homography.json"):
        self.video_path = video_path
        self.calibration = CalibrationIntegration(calibration_file)
        self.player_detector = PlayerDetector(
            model_name="yolo11n.pt",
            confidence_threshold=0.3,
            enable_tracking=True,
            track_buffer=50
        )
        
        # 1. Load Court Image & Rotate to Vertical (Standard Red Dots Canvas)
        self.court_img_orig = cv2.imread(COURT_IMAGE_PATH)
        if self.court_img_orig is None:
            print(f"Error: Could not load court image from {COURT_IMAGE_PATH}")
            sys.exit(1)
            
        # Rotate 90 CW to get Vertical Canvas
        self.court_canvas_vertical = cv2.rotate(self.court_img_orig, cv2.ROTATE_90_CLOCKWISE)
        self.vert_h, self.vert_w = self.court_canvas_vertical.shape[:2]
        print(f"   Vertical Canvas: {self.vert_w}x{self.vert_h}")

        # 2. Setup Geometry for UWB -> Vertical Pixel Mapping
        self.setup_geometry()
        
        # 3. Load UWB Data (Keep for local drawing)
        self.load_uwb_data()
        
        # Associator placeholder (initialized later with sync_offset)
        self.uwb_associator = None

    def setup_geometry(self):
        """
        Setup mapping from UWB World Coords (cm) -> Vertical Canvas Pixels.
        """
        geometry = parse_court_dxf(COURT_DXF_PATH)
        self.courtBounds = geometry.bounds
        
        courtWidth = self.courtBounds.max_x - self.courtBounds.min_x  # Long side (~2800)
        courtHeight = self.courtBounds.max_y - self.courtBounds.min_y # Short side (~1500)
        
        PADDING = 50
        scaleY = (self.vert_h - PADDING * 2) / courtWidth  # Map X(Long) to Height
        scaleX = (self.vert_w - PADDING * 2) / courtHeight # Map Y(Short) to Width
        
        self.actualScale = min(scaleX, scaleY)
        
        scaledLong = courtWidth * self.actualScale
        scaledShort = courtHeight * self.actualScale
        
        self.offsetY = (self.vert_h - scaledLong) / 2 # Vertical padding
        self.offsetX = (self.vert_w - scaledShort) / 2 # Horizontal padding
        
        self.radius_200cm_px = int(200 * self.actualScale)
        
        # Dot size scaling
        self.DOT_RADIUS = int(40 * (self.vert_w / 7341 * 1.7)) # Heuristic
        if self.DOT_RADIUS < 15: self.DOT_RADIUS = 15
        self.DOT_OUTLINE = 3
        
        print(f"   Scale: {self.actualScale:.6f} px/cm")
        print(f"   200cm Radius: {self.radius_200cm_px} px")

    def uwbToVerticalScreen(self, x_cm, y_cm):
        """
        Convert UWB (cm) coords to Vertical Canvas Pixels.
        """
        # Vertical Y (matches Long dimension X)
        sy = ((x_cm - self.courtBounds.min_x) * self.actualScale) + self.offsetY
        
        # Vertical X (matches Short dimension Y)
        sx = ((y_cm - self.courtBounds.min_y) * self.actualScale) + self.offsetX
        
        return int(sx), int(sy)

    def load_uwb_data(self):
        tags_dir = Path("data/tags")
        self.tag_data = {}
        for tag_file in tags_dir.glob('*.json'):
            with open(tag_file, 'r') as f:
                data = json.load(f)
                self.tag_data[data['tag_id']] = data['positions']

        if not self.tag_data:
            print("Error: No UWB tag data found.")
            sys.exit(1)
            
        first_tag_id = list(self.tag_data.keys())[0]
        self.uwb_start = datetime.fromisoformat(self.tag_data[first_tag_id][0]['datetime'])
        
        self.sorted_tag_data = {}
        for tag_id, positions in self.tag_data.items():
            sorted_positions = []
            for pos in positions:
                dt = datetime.fromisoformat(pos['datetime'])
                sorted_positions.append((dt, pos['x'], pos['y']))
            sorted_positions.sort(key=lambda x: x[0])
            self.sorted_tag_data[tag_id] = sorted_positions

    def get_uwb_positions(self, target_dt):
        positions = {}
        for tag_id, sorted_positions in self.sorted_tag_data.items():
            if not sorted_positions: continue
            timestamps = [dt for dt, x, y in sorted_positions]
            idx = bisect_left(timestamps, target_dt)
            candidates = []
            if idx > 0: candidates.append((idx - 1, abs((timestamps[idx - 1] - target_dt).total_seconds())))
            if idx < len(timestamps): candidates.append((idx, abs((timestamps[idx] - target_dt).total_seconds())))
            
            if candidates:
                best_idx, time_diff = min(candidates, key=lambda x: x[1])
                if time_diff < 1.0:
                    dt, x, y = sorted_positions[best_idx]
                    positions[tag_id] = (x, y)
        return positions

    def run(self, start_time, end_time, sync_offset, output_path):
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps)
        num_frames = end_frame - start_frame
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize UWBAssociator
        self.uwb_associator = UWBAssociator(
            tags_dir=Path("data/tags"),
            sync_offset_seconds=sync_offset,
            proximity_threshold=self.radius_200cm_px,
            remapping_cooldown=60, # 2 seconds
            canvas_width=self.vert_w,
            canvas_height=self.vert_h
        )
        
        court_scale = OUTPUT_HEIGHT / self.vert_h
        court_panel_w = int(self.vert_w * court_scale)
        TOTAL_WIDTH = 1920 + court_panel_w
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (TOTAL_WIDTH, OUTPUT_HEIGHT))
        
        print(f"\nProcessing {num_frames} frames ({start_time} - {end_time})...")

        for frame_idx in tqdm(range(num_frames)):
            ret, frame = cap.read()
            if not ret: break
            
            current_frame = start_frame + frame_idx
            video_time_sec = current_frame / fps
            u_sec = video_time_sec + sync_offset
            target_dt = self.uwb_start + timedelta(seconds=u_sec)
            
            # --- 1. Prepare Vertical Canvas ---
            canvas = self.court_canvas_vertical.copy()
            
            # --- 2. Get UWB Positions & Draw ---
            uwb = self.get_uwb_positions(target_dt)
            uwb_screen = {} # tag_id -> (sx, sy)
            
            # Draw Proximity Circles
            overlay = canvas.copy()
            for tag_id, (ux, uy) in uwb.items():
                sx, sy = self.uwbToVerticalScreen(ux, uy)
                uwb_screen[tag_id] = (sx, sy)
                cv2.circle(overlay, (sx, sy), self.radius_200cm_px, COLOR_LIGHT_BLUE, 2)
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
            
            # Draw Blue Dots (Actual UWB)
            for tag_id, (sx, sy) in uwb_screen.items():
                cv2.circle(canvas, (sx, sy), self.DOT_RADIUS, COLOR_BLUE, -1)
                cv2.circle(canvas, (sx, sy), self.DOT_RADIUS, COLOR_WHITE, self.DOT_OUTLINE)

            # --- 3. Detect Players & Project ---
            tracked_players = self.player_detector.track_players(frame)
            
            projected_players = []
            for player in tracked_players:
                bx, by = player['bottom']
                try:
                    cx, cy = self.calibration.image_to_court(bx, by)
                    if 0 <= cx < self.vert_w and 0 <= cy < self.vert_h:
                        # Append projected coords to player dict for Associator
                        player['court_x'] = float(cx)
                        player['court_y'] = float(cy)
                        
                        projected_players.append({
                            'player': player,
                            'cx': int(cx),
                            'cy': int(cy)
                        })
                except:
                    pass

            # --- 4. Association (Persistent) ---
            # Pass original player dicts (which now contain court_x/y) to Associator
            players_for_acc = [p['player'] for p in projected_players]
            
            # Associator modifies list in-place
            self.uwb_associator.associate(video_time_sec, players_for_acc)

            # --- 5. Draw Players & Annotate Video ---
            for p_idx, p_data in enumerate(projected_players):
                cx, cy = p_data['cx'], p_data['cy']
                player = p_data['player']
                
                assigned_tag = player.get('uwb_tag_id')
                
                # Logic:
                # 1. If no tag: RED
                # 2. If tag:
                #    Check if tag is currently visible (in uwb_screen)
                #    If visible: check distance -> Green/Red
                #    If invisible: Red (or Orange?) -> Keep Tag Label
                
                dist_to_tag = float('inf')
                
                if assigned_tag is not None:
                    if assigned_tag in uwb_screen:
                        tx, ty = uwb_screen[assigned_tag]
                        dist_to_tag = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                        
                        if dist_to_tag <= self.radius_200cm_px:
                            color = COLOR_GREEN
                            
                            # Draw line to tag
                            cv2.line(canvas, (cx, cy), (tx, ty), COLOR_WHITE, 2)
                        else:
                            # Tag assigned but drifted far away
                            color = COLOR_RED 
                    else:
                        # Tag assigned but not in current UWB frame (Dropout/Ghost)
                        color = COLOR_RED # Or COLOR_ORANGE if distinct
                else:
                    color = COLOR_RED
                
                # Determine Label
                if assigned_tag is not None:
                    label = f"Tag {assigned_tag}"
                else:
                    label = f"ID {player.get('track_id')}"
                
                # Draw on Vertical Canvas
                cv2.circle(canvas, (cx, cy), self.DOT_RADIUS, color, -1)
                cv2.circle(canvas, (cx, cy), self.DOT_RADIUS, COLOR_WHITE, self.DOT_OUTLINE)
                
                # Draw on Original Video
                bbox = player['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Show Tag ID if assigned (or ID if needed)
                if assigned_tag is not None:
                    # Draw text with background
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_w, text_h = text_size
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
                    
            # --- 6. Stitch ---
            canvas_resized = cv2.resize(canvas, (court_panel_w, OUTPUT_HEIGHT))
            combined = np.zeros((OUTPUT_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
            combined[0:1080, 0:1920] = frame
            combined[0:1080, 1920:TOTAL_WIDTH] = canvas_resized
            
            out.write(combined)
            
        cap.release()
        out.release()
        print(f"✅ Video Saved: {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python validation_overlap.py <start> <end> <offset> <video>")
        sys.exit(1)
        
    start = sys.argv[1]
    end = sys.argv[2]
    offset = float(sys.argv[3]) if len(sys.argv) > 3 else 1194.0
    video = sys.argv[4] if len(sys.argv) > 4 else VIDEO_PATH
    
    validator = Validator(video)
    output = f"validation_overlap_{start.replace(':','')}_{end.replace(':','')}.mp4"
    validator.run(start, end, offset, output)

if __name__ == "__main__":
    main()
