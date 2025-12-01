
class Synchronizer:
    def __init__(self, fps=30.0):
        self.fps = fps
        self.sync_frame = None
        self.sync_timestamp = None

    def set_sync_point(self, frame_index, uwb_timestamp):
        """
        Set the synchronization point.
        :param frame_index: Video frame number (0-indexed)
        :param uwb_timestamp: UWB timestamp (microseconds)
        """
        self.sync_frame = frame_index
        self.sync_timestamp = uwb_timestamp
        print(f"Sync point set: Frame {frame_index} = Timestamp {uwb_timestamp}")

    def get_timestamp_for_frame(self, frame_index):
        """
        Calculate UWB timestamp for a given video frame.
        """
        if self.sync_frame is None or self.sync_timestamp is None:
            raise ValueError("Sync point not set")

        # Calculate time difference in seconds
        frame_diff = frame_index - self.sync_frame
        time_diff_seconds = frame_diff / self.fps

        # Convert to microseconds (1 second = 1,000,000 microseconds)
        time_diff_micros = int(time_diff_seconds * 1_000_000)

        return self.sync_timestamp + time_diff_micros

    def get_frame_for_timestamp(self, uwb_timestamp):
        """
        Calculate video frame for a given UWB timestamp.
        """
        if self.sync_frame is None or self.sync_timestamp is None:
            raise ValueError("Sync point not set")

        # Calculate time difference in microseconds
        time_diff_micros = uwb_timestamp - self.sync_timestamp
        
        # Convert to seconds
        time_diff_seconds = time_diff_micros / 1_000_000.0

        # Convert to frames
        frame_diff = int(time_diff_seconds * self.fps)

        return self.sync_frame + frame_diff
