#!/Users/rohitkale/miniconda3/envs/court_tracking/bin/python

"""
Interactive Calibration Tool
Standalone GUI for selecting correspondence points between court and video.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

class CalibrationTool:
    def __init__(self, video_path, court_image_path, output_dir):
        self.video_path = Path(video_path)
        self.court_image_path = Path(court_image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Point storage
        self.court_points = []
        self.video_points = []
        self.current_mode = 'court'  # 'court' or 'video'

        # Load images
        self.load_images()

        # Setup GUI
        self.setup_gui()

    def load_images(self):
        """Load video frame and court image."""
        # Load video frame
        # Prefer loading the pre-undistorted image if it exists
        undistorted_path = Path("video_frame_100.jpg")
        if undistorted_path.exists():
            print(f"Loading undistorted frame from {undistorted_path}...")
            frame = cv2.imread(str(undistorted_path))
            if frame is None:
                 raise ValueError(f"Could not read {undistorted_path}")
            self.video_frame_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print("Undistorted frame not found, loading from video...")
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise ValueError("Could not read frame 100 from video")
            
            self.video_frame_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        print(f"Original Video frame: {self.video_frame_orig.shape}")

        # Load court image
        print("Loading court image...")
        court_img = cv2.imread(str(self.court_image_path))
        if court_img is None:
            raise ValueError(f"Could not load court image: {self.court_image_path}")

        self.court_image_orig = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)
        
        # Rotate court image 90 degrees clockwise as requested
        self.court_image_orig = cv2.rotate(self.court_image_orig, cv2.ROTATE_90_CLOCKWISE)
        print("Rotated court image 90 degrees clockwise")
        
        print(f"Original Court image: {self.court_image_orig.shape}")

        # Resize for display (max width 1280)
        max_width = 1280
        
        # Thicken lines for court image (assuming black lines on white/transparent)
        # Create a copy for processing
        court_processed = self.court_image_orig.copy()
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(court_processed, cv2.COLOR_RGB2GRAY)
        
        # Check if lines are dark (mean > 127 implies light background)
        if np.mean(gray) > 127:
            # Light background, dark lines -> Erode to thicken dark lines
            kernel = np.ones((5,5), np.uint8) # 5x5 kernel for significant thickening
            court_processed = cv2.erode(court_processed, kernel, iterations=2)
            print("Applied erosion to thicken dark lines")
        else:
            # Dark background, light lines -> Dilate to thicken light lines
            kernel = np.ones((5,5), np.uint8)
            court_processed = cv2.dilate(court_processed, kernel, iterations=2)
            print("Applied dilation to thicken light lines")

        # Resize court
        h, w = court_processed.shape[:2]
        if w > max_width:
            self.scale_court = max_width / w
            new_size = (max_width, int(h * self.scale_court))
            self.court_image_display = cv2.resize(court_processed, new_size)
        else:
            self.scale_court = 1.0
            self.court_image_display = court_processed.copy()
            
        # Resize video
        h, w = self.video_frame_orig.shape[:2]
        if w > max_width:
            self.scale_video = max_width / w
            new_size = (max_width, int(h * self.scale_video))
            self.video_frame_display = cv2.resize(self.video_frame_orig, new_size)
        else:
            self.scale_video = 1.0
            self.video_frame_display = self.video_frame_orig.copy()
            
        print(f"Display Court scale: {self.scale_court:.4f}")
        print(f"Display Video scale: {self.scale_video:.4f}")

    def setup_gui(self):
        """Setup the GUI window."""
        self.root = tk.Tk()
        self.root.title("Basketball Court Calibration Tool")
        
        # Set window size (maximize for Mac)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{int(screen_width*0.9)}x{int(screen_height*0.9)}+0+0")

        # Configure root grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main_frame grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)  # Canvas row

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="1. Click on COURT (left), then VIDEO (right)\n"
                 "2. Select 10-15 correspondence points\n"
                 "3. Click SAVE when done",
            font=('Arial', 10, 'bold'),
            foreground='blue',
            justify='center'
        )
        instructions.grid(row=0, column=0, columnspan=2, pady=5, sticky='ew')

        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Ready - Click on COURT first",
            font=('Arial', 10),
            foreground='green',
            anchor='center'
        )
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5, sticky='ew')

        # Point count label
        self.point_count_label = ttk.Label(
            main_frame,
            text="Points: 0",
            font=('Arial', 10, 'bold'),
            anchor='center'
        )
        self.point_count_label.grid(row=2, column=0, columnspan=2, pady=5, sticky='ew')

        # Create figure with two subplots
        self.fig, (self.ax_court, self.ax_video) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle('Calibration Tool - Click to Select Points', fontsize=12, fontweight='bold')

        # Display images
        self.ax_court.imshow(self.court_image_display)
        self.ax_court.set_title('COURT (Click First)', fontsize=10, fontweight='bold', color='blue')
        self.ax_court.axis('off')

        self.ax_video.imshow(self.video_frame_display)
        self.ax_video.set_title('VIDEO Frame 100 (Click Second)', fontsize=10, fontweight='bold', color='green')
        self.ax_video.axis('off')
        
        self.fig.tight_layout()

        # Embed matplotlib in tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=10, sticky='nsew')

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)

        # Buttons
        self.undo_btn = ttk.Button(
            button_frame,
            text="⟲ UNDO",
            command=self.undo_point,
            width=15
        )
        self.undo_btn.grid(row=0, column=0, padx=5)

        self.save_btn = ttk.Button(
            button_frame,
            text="💾 SAVE",
            command=self.save_calibration,
            width=15,
            state='disabled'
        )
        self.save_btn.grid(row=0, column=1, padx=5)

        self.delete_btn = ttk.Button(
            button_frame,
            text="🗑 DELETE ALL",
            command=self.delete_all_points,
            width=15
        )
        self.delete_btn.grid(row=0, column=2, padx=5)

        self.quit_btn = ttk.Button(
            button_frame,
            text="❌ QUIT",
            command=self.quit_app,
            width=15
        )
        self.quit_btn.grid(row=0, column=3, padx=5)

        # Store canvas reference
        self.canvas = canvas

    def on_click(self, event):
        """Handle mouse click on image."""
        if event.inaxes is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Determine which image was clicked
        if event.inaxes == self.ax_court:
            if self.current_mode == 'court':
                # Add court point (convert back to original scale)
                orig_x = x / self.scale_court
                orig_y = y / self.scale_court
                self.court_points.append([float(orig_x), float(orig_y)])
                
                # Draw on display (using display coordinates)
                self.draw_point(self.ax_court, x, y, len(self.court_points), 'blue')
                
                self.current_mode = 'video'
                self.status_label.config(
                    text=f"Good! Now click corresponding point on VIDEO",
                    foreground='green'
                )
            else:
                messagebox.showwarning(
                    "Wrong Order",
                    "Please click on VIDEO (right) first to complete the pair!"
                )

        elif event.inaxes == self.ax_video:
            if self.current_mode == 'video':
                # Add video point (convert back to original scale)
                orig_x = x / self.scale_video
                orig_y = y / self.scale_video
                self.video_points.append([float(orig_x), float(orig_y)])
                
                # Draw on display (using display coordinates)
                self.draw_point(self.ax_video, x, y, len(self.video_points), 'red')
                
                self.current_mode = 'court'
                self.status_label.config(
                    text=f"Point {len(self.video_points)} added! Click on COURT for next point",
                    foreground='blue'
                )
                self.update_point_count()
            else:
                messagebox.showwarning(
                    "Wrong Order",
                    "Please click on COURT (left) first!"
                )

        self.canvas.draw()

    def draw_point(self, ax, x, y, num, color):
        """Draw a numbered point on the axis."""
        # Very small point for precision
        ax.plot(x, y, 'o', color=color, markersize=3, markeredgecolor='white', markeredgewidth=0.5)
        # Text further away to not obstruct view
        ax.text(x, y-15, str(num), color=color, fontsize=6, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6))

    def update_point_count(self):
        """Update point count label and enable/disable save button."""
        count = len(self.video_points)
        self.point_count_label.config(text=f"Points: {count}")

        if count >= 10:
            self.save_btn.config(state='normal')
            self.point_count_label.config(foreground='green')
        else:
            self.save_btn.config(state='disabled')
            self.point_count_label.config(foreground='orange')

    def undo_point(self):
        """Undo the last point pair."""
        if len(self.video_points) > 0 and len(self.court_points) > 0:
            self.video_points.pop()
            self.court_points.pop()
            self.current_mode = 'court'
            self.redraw_all_points()
            self.status_label.config(
                text="Last point removed. Click on COURT",
                foreground='blue'
            )
        elif len(self.court_points) > len(self.video_points):
            # Court point added but no video point yet
            self.court_points.pop()
            self.current_mode = 'court'
            self.redraw_all_points()
            self.status_label.config(
                text="Last point removed. Click on COURT",
                foreground='blue'
            )
        else:
            messagebox.showinfo("Info", "No points to undo")

    def delete_all_points(self):
        """Delete all points."""
        if messagebox.askyesno("Confirm", "Delete all points and start over?"):
            self.court_points.clear()
            self.video_points.clear()
            self.current_mode = 'court'
            self.redraw_all_points()
            self.status_label.config(
                text="All points cleared. Click on COURT to start",
                foreground='blue'
            )

    def redraw_all_points(self):
        """Redraw all points."""
        # Clear axes
        self.ax_court.clear()
        self.ax_video.clear()

        # Redisplay images
        self.ax_court.imshow(self.court_image_display)
        self.ax_court.set_title('COURT (Click First)', fontsize=12, fontweight='bold', color='blue')
        self.ax_court.axis('off')

        self.ax_video.imshow(self.video_frame_display)
        self.ax_video.set_title('VIDEO Frame 100 (Click Second)', fontsize=12, fontweight='bold', color='green')
        self.ax_video.axis('off')

        # Redraw all points (convert original to display coords)
        for i, (cx, cy) in enumerate(self.court_points, 1):
            disp_x = cx * self.scale_court
            disp_y = cy * self.scale_court
            self.draw_point(self.ax_court, disp_x, disp_y, i, 'blue')

        for i, (vx, vy) in enumerate(self.video_points, 1):
            disp_x = vx * self.scale_video
            disp_y = vy * self.scale_video
            self.draw_point(self.ax_video, disp_x, disp_y, i, 'red')

        self.canvas.draw()
        self.update_point_count()

    def save_calibration(self):
        """Compute homography and save calibration."""
        if len(self.court_points) < 10:
            messagebox.showwarning("Not Enough Points", "Please select at least 10 points")
            return

        if len(self.court_points) != len(self.video_points):
            messagebox.showerror("Error", "Mismatch in point pairs!")
            return

        try:
            # Compute homography using RANSAC
            court_pts = np.array(self.court_points, dtype=np.float32)
            video_pts = np.array(self.video_points, dtype=np.float32)

            H, mask = cv2.findHomography(court_pts, video_pts, cv2.RANSAC, 5.0)

            if H is None:
                messagebox.showerror("Error", "Could not compute homography matrix")
                return

            # Count inliers
            inliers = np.sum(mask)
            outliers = len(mask) - inliers

            # Prepare calibration data
            calibration = {
                "homography": H.tolist(),
                "court_points": self.court_points,
                "video_points": self.video_points,
                "timestamp": datetime.now().isoformat(),
                "num_points": len(self.court_points),
                "inliers": int(inliers),
                "outliers": int(outliers),
                "video_frame": 100
            }

            # Save to file
            output_file = self.output_dir / "homography.json"
            with open(output_file, 'w') as f:
                json.dump(calibration, f, indent=2)

            messagebox.showinfo(
                "Success!",
                f"Calibration saved successfully!\n\n"
                f"Points: {len(self.court_points)}\n"
                f"Inliers: {inliers}\n"
                f"Outliers: {outliers}\n"
                f"File: {output_file}"
            )

            print(f"\n{'='*60}")
            print("CALIBRATION SAVED")
            print(f"{'='*60}")
            print(f"Output file: {output_file}")
            print(f"Points: {len(self.court_points)}")
            print(f"Inliers: {inliers} ({inliers/len(self.court_points)*100:.1f}%)")
            print(f"Outliers: {outliers}")
            print(f"\nHomography matrix:")
            print(H)
            print(f"{'='*60}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save calibration:\n{str(e)}")
            print(f"Error: {e}")

    def quit_app(self):
        """Quit the application."""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.root.quit()
            self.root.destroy()

    def run(self):
        """Run the GUI."""
        self.root.mainloop()


def main():
    """Main entry point."""
    # Paths
    video_path = "GX020018.MP4"
    court_image_path = "data/calibration/court_image.png"
    output_dir = "data/calibration"

    print("="*60)
    print("BASKETBALL COURT CALIBRATION TOOL")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Court: {court_image_path}")
    print(f"Output: {output_dir}")
    print("="*60)
    print("\nStarting GUI...")

    try:
        tool = CalibrationTool(video_path, court_image_path, output_dir)
        tool.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()