import json
from datetime import datetime, timedelta, timezone

def calculate_offset():
    # 1. Video Creation Time from MP4 Metadata
    # "creation_time": "2025-11-22T02:27:30.000000Z"
    video_creation_str = "2025-11-22T02:27:30.000000Z"
    video_start = datetime.strptime(video_creation_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    video_start = video_start.replace(tzinfo=timezone.utc)

    # 2. Log Start Time from Log File Content
    # First line: 2025-11-22 02:26:40.578
    log_start_str = "2025-11-22 02:26:40.578"
    log_start = datetime.strptime(log_start_str, "%Y-%m-%d %H:%M:%S.%f")
    # Assuming log is also in UTC or matches the camera time zone. 
    # Usually GoPro uses UTC for creation_time metadata.
    # We will assume UWB log time is consistent with the camera time (or we are finding the relative difference).
    log_start = log_start.replace(tzinfo=timezone.utc)

    # 3. Calculate Offset
    # If Video starts AFTER Log: Offset is positive (Log Time + Offset = Video Time)
    # The user wants "sync offset", usually meaning: Video Time = Log Time - Offset OR Log Time = Video Time + Offset
    # Let's find the difference:
    # Difference = Video_Start - Log_Start
    
    diff = video_start - log_start
    offset_seconds = diff.total_seconds()

    print(f"Video Start (UTC): {video_start}")
    print(f"Log Start: {log_start}")
    print(f"Difference (Video - Log): {diff}")
    print(f"Offset in Seconds: {offset_seconds}")

    return offset_seconds

if __name__ == "__main__":
    calculate_offset()
