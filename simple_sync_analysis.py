#!/usr/bin/env python3
"""
Simple synchronization analysis between video timecode and UWB log timestamps.
Only uses GX020018_1080p.MP4 and session_1763778442483.log
"""
from datetime import datetime

# Video information
VIDEO_TIMECODE_START = "22:23:01;26"  # From video metadata
VIDEO_FPS = 29.97
VIDEO_DURATION_SECONDS = 3139.136

# Parse video timecode (HH:MM:SS;FF format, drop-frame)
def timecode_to_seconds(timecode_str):
    """Convert timecode HH:MM:SS;FF to total seconds"""
    parts = timecode_str.replace(';', ':').split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    frames = int(parts[3])

    total_seconds = (hours * 3600) + (minutes * 60) + seconds + (frames / VIDEO_FPS)
    return total_seconds

video_timecode_seconds = timecode_to_seconds(VIDEO_TIMECODE_START)
video_timecode_ms = video_timecode_seconds * 1000

print("="*70)
print("VIDEO ANALYSIS")
print("="*70)
print(f"Video timecode at start: {VIDEO_TIMECODE_START}")
print(f"Timecode in seconds: {video_timecode_seconds:.3f} seconds")
print(f"Timecode in milliseconds: {video_timecode_ms:.0f} ms")
print(f"Video duration: {VIDEO_DURATION_SECONDS:.3f} seconds ({VIDEO_DURATION_SECONDS/60:.2f} minutes)")
print()

# Analyze log file
LOG_FILE = "session_1763778442483.log"

print("="*70)
print("UWB LOG ANALYSIS")
print("="*70)

# Read first and last entries
with open(LOG_FILE, 'r') as f:
    lines = f.readlines()

first_line = lines[0].strip()
last_line = lines[-1].strip()

print(f"Total log entries: {len(lines):,}")
print()

# Parse first entry
first_parts = first_line.split('|')
first_datetime_str = first_parts[0].strip()
first_timestamp = int(first_parts[-1].split('=')[1].strip())
first_datetime = datetime.strptime(first_datetime_str, '%Y-%m-%d %H:%M:%S.%f')

print(f"First log entry:")
print(f"  Date/Time: {first_datetime_str}")
print(f"  UWB Timestamp: {first_timestamp}")
print()

# Parse last entry
last_parts = last_line.split('|')
last_datetime_str = last_parts[0].strip()
last_timestamp = int(last_parts[-1].split('=')[1].strip())
last_datetime = datetime.strptime(last_datetime_str, '%Y-%m-%d %H:%M:%S.%f')

print(f"Last log entry:")
print(f"  Date/Time: {last_datetime_str}")
print(f"  UWB Timestamp: {last_timestamp}")
print()

# Calculate log duration
log_duration = (last_datetime - first_datetime).total_seconds()
uwb_timestamp_diff = last_timestamp - first_timestamp

print(f"Log duration (by date/time): {log_duration:.3f} seconds ({log_duration/60:.2f} minutes)")
print(f"UWB timestamp difference: {uwb_timestamp_diff}")
print()

# Analyze timestamp groups
print("="*70)
print("TIMESTAMP PATTERN ANALYSIS")
print("="*70)

# Sample first 100 entries to find patterns
timestamps = []
for line in lines[:100]:
    parts = line.split('|')
    ts = int(parts[-1].split('=')[1].strip())
    timestamps.append(ts)

unique_ts = sorted(set(timestamps))
print(f"Unique timestamp values in first 100 entries: {len(unique_ts)}")
print(f"Timestamp range: {min(timestamps)} to {max(timestamps)}")
print(f"First few unique values: {unique_ts[:10]}")
print()

# Check if there are multiple timestamp groups
if len(unique_ts) > 10:
    print("Multiple distinct timestamp values detected!")
    print("This suggests different tags may have different time bases.")
else:
    print(f"Found {len(unique_ts)} distinct timestamp values")

print()
print("="*70)
print("SYNCHRONIZATION NOTES")
print("="*70)
print()
print("Video timecode (22:23:01;26) represents:")
print(f"  - {video_timecode_seconds/3600:.2f} hours from timecode zero")
print(f"  - This is the camera's internal clock when recording started")
print()
print("UWB Timestamps represent:")
print(f"  - Milliseconds from when the UWB system started")
print(f"  - Different tags may have different time bases (system resets)")
print()
print("To synchronize:")
print("  1. Use the log date/time stamps (e.g., '2025-11-22 02:26:40.578')")
print("  2. Find a common event visible in both video and court positions")
print("  3. Calculate offset: (video_playback_time - log_date_time)")
print()
print(f"Video starts at: playback time 0:00:00")
print(f"Log starts at: {first_datetime_str}")
print(f"Log ends at: {last_datetime_str}")
print(f"Log covers: {log_duration/60:.1f} minutes")
print(f"Video covers: {VIDEO_DURATION_SECONDS/60:.1f} minutes")
