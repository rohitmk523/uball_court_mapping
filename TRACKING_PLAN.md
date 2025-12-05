# Basketball Court Player Tracking & Tagging System

## Overview
Court-centric video system that combines YOLO detection, UWB tag positioning, and player tracking to create a tagged player visualization.

---

## Current Phase (COMPLETED ✅)

### What We Have:
1. **Red Dots** (Player Detection)
   - YOLO detects players in 1080p video
   - Projects to court canvas using calibrated homography
   - Coordinate logging for analysis
   - 29.97 fps processing

2. **Blue Dots** (UWB Tags)
   - UWB tag positions on court
   - 15-20 Hz sampling rate
   - Real-world ground truth positions

3. **Combined Visualization**
   - Overlay red + blue dots on court
   - Vertical court orientation (4261x7341)
   - High-res output for analysis

### Key Achievement:
Fixed player projection by rotating court image 90° CW to match calibration space.

---

## Phase 2: Hybrid ByteTrack + UWB Matching (NEXT)

### Goal:
Maintain consistent player IDs across frames using ByteTrack, with periodic UWB validation.

### Architecture:

#### 1. Player Detection & Initial Tracking
```
Video Frame → YOLO Detection → ByteTrack → Tracked Players
                                     ↓
                              Assign Track IDs
```

#### 2. UWB Association (Every 2 seconds)
```
Tracked Players (from ByteTrack) ──┐
                                    ├─→ Hungarian Algorithm → Optimal Assignment
UWB Tags (ground truth positions) ─┘
```

#### 3. Player Categorization
- **Tagged Players** (within 200cm of UWB tag)
  - Color: **GREEN**
  - ID: UWB Tag ID (e.g., "Tag 42")
  - Nearest player gets the tag if multiple within radius

- **Untagged Tracked Players** (no nearby UWB tag)
  - Color: **RED**
  - ID: ByteTrack ID (e.g., "P1", "P2")
  - Continue tracking with ByteTrack

- **Lost Track** (ByteTrack confidence drops)
  - Try to re-associate with UWB if within 200cm
  - Otherwise start new track

#### 4. ID Switch Handling
- **Detection**: When tracked position drifts >1m from expected position
- **Resolution**: Re-match all players with UWB tags using Hungarian algorithm
- **Logging**: Record all ID switches for post-processing analysis

### Implementation Details:

#### Initial ID Assignment
```python
def initialize_ids(detections, uwb_tags):
    """
    Assign initial IDs by matching detections with nearest UWB tags.

    - For each detection, find nearest UWB tag within 200cm
    - If multiple detections near same tag, assign to nearest
    - Unmatched detections get sequential IDs (P1, P2, ...)
    """
    pass
```

#### Periodic Re-matching (Every 2 seconds = ~60 frames at 29.97 fps)
```python
def periodic_rematch(tracked_players, uwb_tags):
    """
    Force re-match with UWB to prevent drift accumulation.

    - Build cost matrix: distance between each tracked player and UWB tag
    - Use Hungarian algorithm for optimal assignment
    - Update player IDs based on matches
    - Log any ID changes for analysis
    """
    pass
```

#### Tagging Logic
```python
def assign_tags(tracked_players, uwb_tags, threshold_cm=200):
    """
    Determine which players are tagged.

    For each UWB tag:
        - Find all players within 200cm radius
        - If multiple players: assign to nearest
        - Change player color to GREEN
        - Display tag ID above player

    Untagged players remain RED with ByteTrack ID
    """
    pass
```

### Data Logging:
Record for post-processing analysis:
- Frame number
- Timestamp
- All detections (bbox, confidence, position)
- All tracks (ByteTrack ID, position, confidence)
- All UWB tags (ID, position, timestamp)
- Tag assignments (player → tag)
- ID switch events (old_id → new_id, reason, frame)

### Handling Edge Cases:

#### Case 1: Multiple Players Cross Paths
- **Scenario**: 3 players cross, all IDs get swapped
- **Detection**: After crossing, tracked positions don't match UWB
- **Solution**: Log event, continue with potentially wrong IDs
- **Post-processing**: Re-run with stricter matching after analyzing logs

#### Case 2: Player Temporarily Off-Court
- **Scenario**: Player goes out of frame (no UWB data off-court)
- **Solution**: ByteTrack continues tracking if player returns quickly (<2 sec)
- **If longer**: Lose track, re-initialize ID when player returns

#### Case 3: ByteTrack Confidence Drops
- **Scenario**: Occlusion, jersey similarity causes tracking uncertainty
- **Solution**: Check if position matches any UWB tag within 200cm
- **If yes**: Re-assign ID based on UWB
- **If no**: Continue with low confidence, mark for review

#### Case 4: New Player Enters Frame
- **Scenario**: Player wasn't detected initially, now appears
- **Detection**: ByteTrack assigns new track ID
- **Association**: Check distance to all UWB tags
- **If match**: Assign UWB tag ID
- **If no match**: Keep as untagged tracked player

### Performance Considerations:
- ByteTrack: ~30 fps (real-time capable)
- Hungarian algorithm: O(n³) but n ≤ 10 players, negligible
- Re-matching overhead: 2 seconds → minimal impact
- Logging: Write to JSON buffer, flush every 100 frames

### Output Visualization:
```
Court Canvas (Vertical 4261x7341)
├── Court lines (white)
├── UWB tags (blue dots with radius circle)
├── Tagged players (green bounding box + tag ID)
├── Untagged players (red bounding box + track ID)
└── Info panel (frame, timestamp, player count, tagged count)
```

---

## Phase 3: Kalman Filtering for UWB Prediction (FUTURE)

### Goal:
Improve matching accuracy by predicting UWB positions between samples.

### Why Needed:
- UWB samples at 15-20 Hz (every ~50-65ms)
- Video processes at 29.97 fps (every ~33ms)
- Gap between UWB samples → need interpolation

### Approach:

#### 1. UWB State Tracking
```python
class UWBTracker:
    """
    Kalman filter for each UWB tag.

    State: [x, y, vx, vy]  # position + velocity

    Predict: Use constant velocity model
    Update: Correct with actual UWB measurement
    """
```

#### 2. Prediction Pipeline
```
UWB Sample (t=0ms) → Kalman Update → State [x, y, vx, vy]
                           ↓
Video Frame (t=33ms) → Kalman Predict → Estimated [x', y']
                           ↓
Video Frame (t=66ms) → Kalman Predict → Estimated [x'', y'']
                           ↓
UWB Sample (t=100ms) → Kalman Update → Correct & update state
```

#### 3. Integration with ByteTrack
```python
def match_with_predicted_uwb(tracked_players, uwb_trackers, frame_time):
    """
    Match tracked players with Kalman-predicted UWB positions.

    - Predict UWB positions at current video frame time
    - Build cost matrix with predicted positions
    - Use Hungarian algorithm for optimal assignment
    - Smoother matching than raw UWB samples
    """
    pass
```

### Benefits:
- ✅ Smoother ID assignments (no jitter between UWB samples)
- ✅ Better handling of UWB sampling gaps
- ✅ Can detect anomalies (player moves differently than predicted)
- ✅ Helps with occlusion recovery (predicted position guides search)

### Implementation:
- Use `filterpy` library for Kalman filters
- Tune process noise (how much we trust velocity model)
- Tune measurement noise (how much we trust UWB readings)
- Handle missed UWB samples (coast on prediction)

---

## Phase 4: Post-Processing & Refinement (OPTIONAL)

### After generating full game video with tracking:

#### 1. ID Switch Analysis
- Review logged ID switch events
- Identify patterns (same location, specific game events)
- Create rules to prevent similar switches in future

#### 2. Trajectory Smoothing
- Use full game UWB trajectory to validate tracking
- Identify implausible jumps (player "teleports")
- Re-run matching with stricter constraints on suspicious segments

#### 3. Manual Correction (if needed)
- Provide GUI tool to manually correct persistent ID errors
- Export corrected track data for final video generation

---

## Technical Stack:

### Current:
- **Detection**: YOLOv8n (ultralytics)
- **Projection**: OpenCV homography
- **Visualization**: OpenCV
- **Data**: JSON coordinate logs

### Phase 2 Additions:
- **Tracking**: ByteTrack (from `boxmot` or custom implementation)
- **Matching**: SciPy Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
- **Logging**: Enhanced JSON with tracking metadata

### Phase 3 Additions:
- **Filtering**: FilterPy Kalman filters
- **Prediction**: NumPy for state prediction

---

## Success Metrics:

### Phase 2:
- ✅ Player IDs remain consistent for >5 seconds without UWB correction
- ✅ <5% ID switches per minute in typical game footage
- ✅ >95% correct tag assignments when player within 200cm

### Phase 3:
- ✅ Smoother tracking (less jitter in player positions)
- ✅ <2% ID switches per minute (improved from Phase 2)
- ✅ Better occlusion recovery (track doesn't break during brief occlusions)

---

## Next Steps:

1. ✅ Generate baseline videos (red, blue, combined) for 15-20 min segment
2. ✅ Analyze sync offset between video and UWB
3. 🔄 Implement ByteTrack integration (Phase 2)
4. 🔄 Test on 5-minute segment
5. 🔄 Add Kalman filtering (Phase 3)
6. 🔄 Process full game video
7. 🔄 Analyze results and tune parameters

---

## Notes:

### SAM2 Discussion:
- SAM2 has tracking capabilities but adds unnecessary complexity
- ByteTrack is more specialized for multi-object tracking
- SAM2 better suited for segmentation tasks (pixel-level masks)
- **Decision**: Use ByteTrack for tracking, skip SAM2 for now

### Color Coding:
- **GREEN**: Tagged player (within 200cm of UWB tag, displays tag ID)
- **RED**: Untagged tracked player (displays ByteTrack ID)
- **BLUE**: UWB tag position (with 200cm radius circle)

### Coordinate Systems:
- **Video**: 1920x1080 pixels (1080p)
- **Court Canvas (Horizontal)**: 7341x4261 pixels
- **Court Canvas (Vertical)**: 4261x7341 pixels (after 90° CW rotation)
- **UWB**: Real-world coordinates in cm (court-centric)

---

*Last Updated: 2025-12-04*
*Status: Current Phase Complete, Phase 2 Planning Done*
