"""
python -m ipdb scripts/make_demo_video.py
made with claude-3.7-thinking in cursor over many iterations
"""

import os
from pprint import pprint
import sys
import ipdb
import data.load_viddiff_dataset as lvd
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, ImageClip, VideoClip
from moviepy.video.fx.resize import resize
from moviepy.video.fx.margin import margin


dir_results = Path("results/demo_videos")
dir_results.mkdir(parents=True, exist_ok=True)

def select_video_pair(action="ballsports_3"):
    """
    Select a video pair for the given action.
    """
    # load the dataset
    dataset = lvd.load_viddiff_dataset(splits=['medium'])
    dataset = dataset.filter(lambda example: action in example['action'])
    videos = lvd.load_all_videos(dataset, do_tqdm=True, cache=True, cache_dir="cache/cache_data")

    # Create action-specific results directory
    action_dir = dir_results / action
    action_dir.mkdir(exist_ok=True)

    # Process each pair of videos
    for i in range(len(videos[0])):
        # Get first frame from each video
        frame1 = videos[0][i]['video'][0]  # First frame of video 1
        frame2 = videos[1][i]['video'][0]  # First frame of video 2

        # Convert frames to PIL Images
        img1 = Image.fromarray(frame1)
        img2 = Image.fromarray(frame2)

        # Create a new image with width = sum of widths, height = max height
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        combined_img = Image.new('RGB', (total_width, max_height))

        # Paste the images side by side
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width, 0))

        # Save the combined image
        save_path = action_dir / f"thumbnail_{i}.png"
        combined_img.save(save_path)
    
    ### now get the differences
    # get the unique differences
    diffs = dataset['differences_annotated']
    diffs_unique = dict()
    for diff in diffs:
        for k, v in diff.items():
            if v is not None:
                diffs_unique[k] = v['description']
    
    diffs_gt = []
    for row in dataset:
        this_row = []
        diffs = row['differences_gt']
        for k, v in diffs.items():
            if v is not None: 
                this_row.append(v)
        diffs_gt.append(this_row)

    # get the unique differences
    df = pd.DataFrame(diffs_gt)

    # example filter for indexes
    idxs = np.where((df[4]!= 'c') & (df[5]!= 'c'))[0]
    # get the video paths
    for idx in idxs:
        path0 = videos[0][idx]['path']
        fps0 = videos[0][idx]['fps']
        frames_trim0 = videos[0][idx]['frames_trim']
        path1 = videos[1][idx]['path']
        fps1 = videos[1][idx]['fps']
        frames_trim1 = videos[1][idx]['frames_trim']

        print(f"idx: {idx}")
        print(path0)
        print(fps0)
        print(frames_trim0)
        print(path1)
        print(fps1)
        print(frames_trim1)
        print()
        print()
    pprint(diffs_unique)
    print(df)
    ipdb.set_trace()
    pass

def find_filenames():
    dataset = lvd.load_viddiff_dataset(splits=['medium','easy','hard'])
    # for row in dataset:
    #     if 'deadlift' in row['action_description']:
    #         print("deadlift action name is", row['action'])
    #         break
    action_names = ['fitness_2','ballsports_1', 'music_0', 'diving_0', 'surgery_0', 'fitness_1']

    seen_actions = dict(zip(action_names, [0,]*len(action_names)))
    fnames, fpss, frames_trims = [], [], []
    for i, row in enumerate(dataset):
        action = row['action']
        if action not in action_names:
            continue

        if seen_actions[action] > 0:
            continue

        seen_actions[action] += 1
        
        vids = row['videos']
        fnames.append(vids[0]['path'])
        fpss.append(vids[0]['fps_original'])
        frames_trims.append(vids[0]['frames_trim'])

    print(fnames)
    print(fpss)
    print(frames_trims)
    # ipdb.set_trace()
    # pass

def create_demo_video(fname0, fname1, output_path, texts, duration=None, 
                     frames_trim0=None, frames_trim1=None,
                     text_color=(0, 0, 0),  # Changed default to black
                     font_style="FONT_HERSHEY_SIMPLEX", title_box=False):
    """
    Create a demo video showcasing the differences between two videos.
    
    Args:
        fname0: Path to first video
        fname1: Path to second video
        output_path: Path to save the output video
        texts: List of text descriptions about the differences
        duration: Optional duration to trim videos to (in seconds)
        frames_trim0: Optional list [start_frame, end_frame, None] to trim first video
        frames_trim1: Optional list [start_frame, end_frame, None] to trim second video
        text_color: RGB tuple for text color, default is black (0, 0, 0)
        font_style: Font style to use (see available_fonts for options)
    """
    # Available font options in OpenCV
    available_fonts = {
        "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
        "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
        "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
        "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
        "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
        "FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
        "FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        "FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    }
    
    # Use the selected font or default to SIMPLEX if invalid
    font = available_fonts.get(font_style)#, cv2.FONT_HERSHEY_SIMPLEX)
    
    # Load the videos
    video0 = VideoFileClip(fname0)
    video1 = VideoFileClip(fname1)
    
    # Convert frame trimming to time if provided
    if frames_trim0 and len(frames_trim0) >= 2 and frames_trim0[0] is not None and frames_trim0[1] is not None:
        start_time0 = frames_trim0[0] / video0.fps
        end_time0 = frames_trim0[1] / video0.fps
        video0 = video0.subclip(start_time0, end_time0)
    # Otherwise trim by duration if provided
    elif duration:
        video0 = video0.subclip(0, min(duration, video0.duration))
        
    if frames_trim1 and len(frames_trim1) >= 2 and frames_trim1[0] is not None and frames_trim1[1] is not None:
        start_time1 = frames_trim1[0] / video1.fps
        end_time1 = frames_trim1[1] / video1.fps
        video1 = video1.subclip(start_time1, end_time1)
    # Otherwise trim by duration if provided
    elif duration:
        video1 = video1.subclip(0, min(duration, video1.duration))
    
    # Ensure both videos have the same height for side-by-side display
    target_height = min(video0.h, video1.h)
    video0_resized = resize(video0, height=target_height)
    video1_resized = resize(video1, height=target_height)
    
    # Extract first frames
    frame0 = video0_resized.get_frame(0)
    frame1 = video1_resized.get_frame(0)
    
    # Function to add text overlay using PIL for better quality text
    def add_text_overlay(frame, show_diff_texts=True, is_intro=False, title_box=False):
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Convert the OpenCV image (BGR) to PIL Image (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Create a drawing context
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load fonts - use system fonts if available, otherwise fall back to default
        try:
            # Try these common system fonts first - adjust paths for your OS
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/Library/Fonts/Arial.ttf",  # macOS
                "C:/Windows/Fonts/Arial.ttf",  # Windows
                "C:/Windows/Fonts/Calibri.ttf",  # Windows alternative
                "C:/Windows/Fonts/Verdana.ttf",  # Windows alternative
            ]
            
            title_font = None
            for path in font_paths:
                try:
                    title_font = ImageFont.truetype(path, 80)  # Title font size - keeping the same
                    regular_font = ImageFont.truetype(path, 28)  # Regular text size - reduced by 30%
                    small_font = ImageFont.truetype(path, 21)  # Small text size - reduced by 30%
                    break
                except IOError:
                    continue
            
            if title_font is None:
                raise IOError("No system fonts found")
            
        except IOError:
            # Fall back to default font
            print("Warning: Could not load font, using default")
            title_font = ImageFont.load_default()
            regular_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        if is_intro:
            # For phase 1: bigger, centered text without Video A/B labels
            text_lines = ["Video Action Differencing (VidDiff)", "Let's compare these videos"]
            
            # Measure text sizes
            text_widths = [draw.textlength(line, font=title_font) for line in text_lines]
            text_height = title_font.getbbox(text_lines[0])[3]  # Get height from bounding box
            line_spacing = text_height // 2  # Space between lines
            
            max_width = max(text_widths)
            total_height = (text_height * len(text_lines)) + (line_spacing * (len(text_lines) - 1))
            
            # Calculate positions for centered text
            x_center = w // 2
            y_center = h // 2
            y_start = y_center - (total_height // 2)
            
            # Draw semi-transparent background box if requested
            if title_box:
                padding = 40
                box_left = x_center - (max_width // 2) - padding
                box_right = x_center + (max_width // 2) + padding
                box_top = y_start - padding
                box_bottom = y_start + total_height + padding
                
                # Create a semi-transparent overlay
                overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                draw_overlay = ImageDraw.Draw(overlay)
                draw_overlay.rectangle(
                    [(box_left, box_top), (box_right, box_bottom)],
                    fill=(0, 0, 0, 180)  # Black with 70% opacity
                )
                
                # Paste the overlay onto the main image
                pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
                draw = ImageDraw.Draw(pil_img)
                
                text_color = (255, 255, 255)  # White
            else:
                text_color = (0, 0, 0)  # Black
            
            # Draw each line of text
            for i, line in enumerate(text_lines):
                x = x_center - (text_widths[i] // 2)  # Center each line
                y = y_start + (i * (text_height + line_spacing))
                draw.text((x, y), line, font=title_font, fill=text_color)
            
        else:
            # # For other phases: text in top left corner
            # text_lines = ["Video Action Differencing (VidDiff)", "Let's compare these videos"]
            
            # # Adjusted spacing for smaller font
            # line_spacing = 35  # Reduced for smaller font
            
            # # Draw header text
            # for i, line in enumerate(text_lines):
            #     draw.text((20, 15 + i * line_spacing), line, font=regular_font, fill=(0, 0, 0))
            
            # Draw Video A and Video B labels with transparent background boxes
            video_a_text = "Video A"
            video_b_text = "Video B"

            # Create font for video labels (using existing font object)
            label_font = regular_font

            # Calculate text dimensions
            video_a_width = draw.textlength(video_a_text, font=label_font)
            video_b_width = draw.textlength(video_b_text, font=label_font)
            text_height = label_font.getbbox(video_a_text)[3]

            # Add padding for the background boxes
            box_padding = 10

            # Create semi-transparent overlay for Video A label
            overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)

            # Video A box position (adjust coordinates as needed based on your layout)
            video_a_x = w // 4 - video_a_width // 2
            video_a_y = 30  # Adjust this value as needed
            draw_overlay.rectangle(
                [(video_a_x - box_padding, video_a_y - box_padding), 
                 (video_a_x + video_a_width + box_padding, video_a_y + text_height + box_padding)],
                fill=(0, 0, 0, 180)  # Black with 70% opacity
            )

            # Video B box position
            video_b_x = 3 * w // 4 - video_b_width // 2
            video_b_y = 30  # Adjust this value as needed
            draw_overlay.rectangle(
                [(video_b_x - box_padding, video_b_y - box_padding), 
                 (video_b_x + video_b_width + box_padding, video_b_y + text_height + box_padding)],
                fill=(0, 0, 0, 180)  # Black with 70% opacity
            )

            # Paste the overlay onto the main image
            pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(pil_img)

            # Draw the text in white for better visibility
            draw.text((video_a_x, video_a_y), video_a_text, font=label_font, fill=(255, 255, 255))
            draw.text((video_b_x, video_b_y), video_b_text, font=label_font, fill=(255, 255, 255))
        
        # Add difference texts only if requested
        if show_diff_texts:
            # Center the difference texts at the bottom
            diff_y_start = h - 120  # Starting y position
            diff_spacing = 30  # Spacing between lines
            
            # Calculate max width of all difference texts
            diff_widths = []
            for text in texts:
                text_width = draw.textlength(text, font=regular_font)
                diff_widths.append(text_width)
            
            max_diff_width = max(diff_widths)
            
            # Create transparent background box for difference texts
            box_padding_x = 20  # Horizontal padding
            box_padding_y = 10  # Vertical padding
            
            box_left = (w // 2) - (max_diff_width // 2) - box_padding_x
            box_right = (w // 2) + (max_diff_width // 2) + box_padding_x
            
            # Calculate approximate text height
            text_height = regular_font.getbbox("Ayg")[3]  # Including descenders like 'g'
            
            # Calculate exact coordinates for box
            box_top = diff_y_start - box_padding_y
            
            # If there are exactly 2 lines (as in the screenshot)
            if len(texts) == 2:
                # Make sure to extend enough to cover the text height of the second line
                box_bottom = diff_y_start + diff_spacing + text_height + box_padding_y
            else:
                # For other numbers of lines
                last_line_y = diff_y_start + (len(texts) - 1) * diff_spacing
                box_bottom = last_line_y + text_height + box_padding_y
            
            # Create a semi-transparent overlay
            overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.rectangle(
                [(box_left, box_top), (box_right, box_bottom)],
                fill=(0, 0, 0, 180)  # Black with 70% opacity
            )
            
            # Paste the overlay onto the main image
            pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(pil_img)
            
            # Draw text in white (for better visibility on dark background)
            text_color = (255, 255, 255)
            
            for i, text in enumerate(texts):
                text_width = draw.textlength(text, font=regular_font)
                diff_x = (w // 2) - (text_width // 2)
                diff_y = diff_y_start + (i * diff_spacing)
                
                draw.text((diff_x, diff_y), text, font=regular_font, fill=text_color)
        else:
            # For other texts without a box, keep original text color
            text_color = (0, 0, 0)
        
        # Convert back to OpenCV format (BGR)
        result_img = np.array(pil_img)
        if result_img.shape[2] == 4:  # If RGBA, convert to RGB first
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)
        result_frame = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        return result_frame
    
    # PART 1: Show first frames side by side
    frame_duration = 3
    combined_frame = np.hstack([frame0, frame1])
    combined_frame = add_text_overlay(combined_frame, show_diff_texts=False, is_intro=True, title_box=True)
    frame_clip = ImageClip(combined_frame).set_duration(frame_duration)
    
    # PART 2: Play video0 with frame1 as still image
    def process_video0_frame(frame):
        frame_with_text = np.zeros((target_height, video0_resized.w + video1_resized.w, 3), dtype=np.uint8)
        frame_with_text[:, :video0_resized.w] = frame
        frame_with_text[:, video0_resized.w:] = frame1  # Show frame 0 of video B
        return add_text_overlay(frame_with_text, show_diff_texts=False, is_intro=False)
    
    video0_with_text = video0_resized.fl_image(process_video0_frame)
    
    # PART 3: Play video1 with frame0 as still image
    def process_video1_frame(frame):
        frame_with_text = np.zeros((target_height, video0_resized.w + video1_resized.w, 3), dtype=np.uint8)
        frame_with_text[:, :video0_resized.w] = frame0  # Show frame 0 of video A
        frame_with_text[:, video0_resized.w:] = frame
        return add_text_overlay(frame_with_text, show_diff_texts=False, is_intro=False)
    
    video1_with_text = video1_resized.fl_image(process_video1_frame)
    
    # PART 4: Create side-by-side clips more efficiently
    # Use already trimmed videos instead of loading them again
    video0_base = video0
    video1_base = video1
    
    video0_base_resized = video0_resized
    video1_base_resized = video1_resized
    
    # Function to process frames side by side
    def process_side_by_side_frame(t):
        # Get frames at the same time point
        try:
            frame0 = video0_base_resized.get_frame(t)
            frame1 = video1_base_resized.get_frame(min(t, video1_base_resized.duration - 0.001))
        except:
            # Fallback in case of time out of bounds
            frame0 = video0_base_resized.get_frame(0)
            frame1 = video1_base_resized.get_frame(0)
        
        combined = np.hstack([frame0, frame1])
        return add_text_overlay(combined, show_diff_texts=True, is_intro=False)  # Show difference texts
    
    # Create a dummy clip with the right duration
    side_by_side_duration = min(video0_base_resized.duration, video1_base_resized.duration)
    side_by_side_base = ImageClip(process_side_by_side_frame(0))
    side_by_side_base = side_by_side_base.set_duration(side_by_side_duration)
    side_by_side_base = side_by_side_base.set_make_frame(lambda t: process_side_by_side_frame(t))
    
    # Create 3 copies of the side-by-side clip
    side_by_side_clips = [side_by_side_base] * 3
    
    # Concatenate all clips
    final_clip = concatenate_videoclips([
        frame_clip,
        video0_with_text,
        video1_with_text,
        *side_by_side_clips
    ])
    
    # Write the final video
    final_clip.write_videofile(str(output_path), codec='libx264', fps=24, preset='ultrafast', bitrate='1000k')
    
    # Close all clips to release resources
    video0.close()
    video1.close()
    final_clip.close()
    
    print(f"Demo video created successfully: {output_path}")
    
    # Return the list of available fonts for reference
    return output_path

def create_clean_background(fname0, fname1, frames_trim0, frames_trim1, fps0=30.0, fps1=30.0):
    """
    Create a clean background from the first frames of two videos without any text overlay.
    
    Args:
        fname0, fname1: Paths to the two videos
        frames_trim0, frames_trim1: Frame trimming for each video
        fps0, fps1: Frame rates for each video
    
    Returns:
        A numpy array containing the composite image
    """
    # Load the source videos
    clip0 = VideoFileClip(fname0)
    clip1 = VideoFileClip(fname1)
    
    # Calculate the frame time based on trimming
    if frames_trim0 and frames_trim0[0] is not None:
        time0 = frames_trim0[0] / fps0
    else:
        time0 = 0
        
    if frames_trim1 and frames_trim1[0] is not None:
        time1 = frames_trim1[0] / fps1
    else:
        time1 = 0
    
    # Get first frames from each clip
    frame0 = clip0.get_frame(time0)
    frame1 = clip1.get_frame(time1)
    
    # Get dimensions
    h0, w0 = frame0.shape[:2]
    h1, w1 = frame1.shape[:2]
    
    # Resize to consistent height
    common_height = 360  # Same as in create_demo_video
    frame0 = resize_frame(frame0, height=common_height)
    frame1 = resize_frame(frame1, height=common_height)
    
    # Get new dimensions
    h0, w0 = frame0.shape[:2]
    h1, w1 = frame1.shape[:2]
    
    # Create composite image (side by side layout)
    total_width = w0 + w1 + 10  # 10px gap between videos
    composite = np.zeros((common_height, total_width, 3), dtype=np.uint8)
    
    # Place the frames
    composite[:, :w0] = frame0
    composite[:, w0+10:w0+10+w1] = frame1
    
    # Close video clips
    clip0.close()
    clip1.close()
    
    return composite

def resize_frame(frame, width=None, height=None):
    """Resize a frame to specified width or height while preserving aspect ratio"""
    h, w = frame.shape[:2]
    
    if width is None and height is None:
        return frame
    
    if width is None:
        aspect = w / h
        width = int(height * aspect)
    elif height is None:
        aspect = h / w
        height = int(width * aspect)
    
    return cv2.resize(frame, (width, height))

def create_video_collage(fnames, fpss, frames_trims, output_path, background_fname0, background_fname1, 
                        background_frames_trim0, background_frames_trim1, num_loops=3):
    # Import PIL for better text rendering
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create clean background from first frames of comparison videos
    background_frame = create_clean_background(
        background_fname0, 
        background_fname1, 
        background_frames_trim0, 
        background_frames_trim1
    )
    
    # Apply 70% opacity to the background
    background_frame = (background_frame * 0.7).astype(np.uint8)
    
    # Get dimensions from background frame
    canvas_height, canvas_width = background_frame.shape[:2]
    print(f"Background dimensions: {canvas_width}x{canvas_height}")
    
    # Get target FPS from a reference video
    reference_clip = VideoFileClip(background_fname0)
    target_fps = reference_clip.fps
    reference_clip.close()
    
    # Load and trim all videos
    video_clips = []
    video_paths = []
    
    print("\n----- Loading Video Clips -----")
    for i, (fname, fps, frames_trim) in enumerate(zip(fnames, fpss, frames_trims)):
        try:
            path = fname
            print(f"Loading video {len(video_clips)+1}: {path}")
            
            # Check if the path is a directory or a file
            is_dir = Path(path).is_dir() or (not path.endswith('.mp4') and Path(f"{path}").is_dir())
            
            if is_dir:
                print(f"  - Path is a directory of images")
                dir_path = path if Path(path).is_dir() else f"{path}"
                
                # For directories, use the function from load_viddiff_dataset
                if frames_trim and len(frames_trim) >= 2 and frames_trim[0] is not None and frames_trim[1] is not None:
                    frames_slice = slice(frames_trim[0], frames_trim[1])
                else:
                    frames_slice = None
                
                # Load frames from directory
                all_frames = lvd._load_video_from_directory_of_images(dir_path, frames_trim=frames_slice)
                print(f"  - Loaded {len(all_frames)} frames from directory")
                
                # Limit to 3 seconds (based on fps)
                max_frames = int(3.0 * fps)
                if len(all_frames) > max_frames:
                    all_frames = all_frames[:max_frames]
                    print(f"  - Limited to {max_frames} frames (3 seconds at {fps} fps)")
                
                # Create a fixed copy of the frames for this closure
                current_frames = all_frames.copy()
                current_fps = fps
                
                # Fix the closure issue with a factory function
                def create_frame_getter(frames_array, frames_fps):
                    # This factory binds the current values to a new function
                    def get_frame(t):
                        frame_idx = min(int(t * frames_fps), len(frames_array) - 1)
                        return frames_array[frame_idx]
                    return get_frame
                
                # Create a closure with fixed references to the current frames
                make_frame_function = create_frame_getter(current_frames, current_fps)
                
                clip_duration = len(current_frames) / current_fps
                clip = VideoClip(make_frame_function, duration=clip_duration)
                clip = clip.set_fps(current_fps)
                print(f"  - Created video clip with {len(current_frames)} frames")
                
            else:
                # For MP4 files, use VideoFileClip
                if not path.endswith('.mp4'):
                    video_path = f"{path}.mp4"
                else:
                    video_path = path
                
                clip = VideoFileClip(video_path)
                
                # Trim based on frames_trim
                if frames_trim and len(frames_trim) >= 2 and frames_trim[0] is not None and frames_trim[1] is not None:
                    start_time = frames_trim[0] / fps
                    end_time = frames_trim[1] / fps
                    clip = clip.subclip(start_time, end_time)
                    print(f"  - Trimmed from {start_time}s to {end_time}s")
            
            # Limit to 3 seconds maximum for all clips
            if clip.duration > 3.0:
                clip = clip.subclip(0, 3.0)
                print(f"  - Limited to 3.0s duration")
            
            # Resize for collage - make height slightly smaller for better spacing
            original_size = (clip.w, clip.h)
            width_factor = canvas_width / 1280
            target_height = int(140 * width_factor)  # Reduced size for better spacing
            clip = resize(clip, height=target_height)
            print(f"  - Resized from {original_size} to {clip.w}x{clip.h}")
            
            video_clips.append(clip)
            video_paths.append(path)
            print(f"  - Successfully loaded clip {len(video_clips)}, duration: {clip.duration:.2f}s")
            
        except Exception as e:
            print(f"  - Error loading video {i+1}: {e}")
    
    print(f"\nTotal video clips for collage: {len(video_clips)}")
    for i, (clip, path) in enumerate(zip(video_clips, video_paths)):
        print(f"Clip {i+1}: {path}, {clip.w}x{clip.h}, duration: {clip.duration:.2f}s")
    
    # Create the collage
    def make_collage_frame(t):
        # Start with the background frame as the canvas (70% opacity)
        canvas = background_frame.copy()
        
        # Fine-tuned positions based on the actual result
        positions_and_scales = [
            # (x_percent, y_percent, scale_factor, rotation_angle)
            (0.125, 0.48, 0.95, 0),         # Top left
            (0.45, 0.48, 0.95, 0),         # Top middle 
            (0.75, 0.48, 0.95, 0),         # Top right
            (0.28, 0.8, 0.95, 0),         # Bottom left (centered between top left and middle)
            (0.61, 0.8, 0.95, 0),         # Bottom middle (centered between top middle and right)
            (0.86, 0.8, 0.95, 0),         # Bottom right (slightly offset and lower)
        ]
        
        # Place each video frame at its position
        for i, clip in enumerate(video_clips):
            if i >= len(positions_and_scales):  # Skip if we have more clips than positions
                break
                
            # Get frame for this time point (with looping)
            clip_t = t % clip.duration
            try:
                frame = clip.get_frame(clip_t)
                h, w = frame.shape[:2]
                
                # Get the position, scale, and rotation for this clip
                x_percent, y_percent, scale, rotation = positions_and_scales[i]
                x = int(canvas_width * x_percent)
                y = int(canvas_height * y_percent)
                
                # Apply scaling
                if scale != 1.0:
                    scaled_h = int(h * scale)
                    scaled_w = int(w * scale)
                    frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
                    h, w = scaled_h, scaled_w
                
                # Apply rotation if needed
                if rotation != 0:
                    # Get the rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1)
                    # Apply the rotation
                    frame = cv2.warpAffine(frame, rotation_matrix, (w, h))
                
                # Calculate placement coordinates (centered on the position)
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = x1 + w
                y2 = y1 + h
                
                # Ensure coordinates are within canvas bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(canvas_width, x2)
                y2 = min(canvas_height, y2)
                
                # Calculate corresponding region in the frame
                frame_x1 = 0 if x1 >= 0 else -x1
                frame_y1 = 0 if y1 >= 0 else -y1
                frame_x2 = frame_x1 + (x2 - x1)
                frame_y2 = frame_y1 + (y2 - y1)
                
                # Apply subtle border/shadow effect for visual depth
                border_size = int(min(h, w) * 0.02)  # 2% of video size
                if border_size > 0 and frame.shape[0] > 2*border_size and frame.shape[1] > 2*border_size:
                    # Create a slightly darkened border
                    mask = np.ones_like(frame)
                    cv2.rectangle(mask, (border_size, border_size), 
                                 (frame.shape[1]-border_size, frame.shape[0]-border_size), 
                                 (0.85, 0.85, 0.85), -1)
                    frame = (frame * mask).astype(np.uint8)
                
                # Ensure frame region is valid
                if frame_y2 <= frame.shape[0] and frame_x2 <= frame.shape[1]:
                    # Place the frame on the canvas
                    canvas[y1:y2, x1:x2] = frame[frame_y1:frame_y2, frame_x1:frame_x2]
                
            except Exception as e:
                print(f"Error in frame rendering for clip {i}: {e}")
        
        # Add title text with PIL for better font rendering
        title_text = "The VidDiffBench benchmark has real-world skilled actions"
        
        # Create a transparent PIL Image for text
        text_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        # Try to load a nicer font, fall back to default if not available
        try:
            # Calculate font size - increased by 50% from current size
            font_size = int(14 * (canvas_width/1280) * 2 * 1.0)
            
            # Try multiple common fonts that might be available
            font_options = [
                "Arial", 
                "DejaVuSans",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "Helvetica",
                "FreeSans"
            ]
            
            font = None
            for font_name in font_options:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except (OSError, IOError):
                    continue
                    
            if font is None:
                # If no font works, use default font
                font = ImageFont.load_default()
        except Exception:
            # Fall back to default font if anything goes wrong
            font = ImageFont.load_default()
            
        # Calculate text size and position
        text_bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (canvas_width - text_width) // 2
        text_y = 15  # Y position kept higher as in the previous change
        
        # Draw semi-transparent background for text
        bg_padding = 25  # Slightly increased padding for the larger text
        background_coords = (
            text_x - bg_padding, 
            text_y - bg_padding,
            text_x + text_width + bg_padding,
            text_y + text_height + bg_padding
        )
        draw.rectangle(background_coords, fill=(0, 0, 0, 150))  # Black with 60% opacity
        
        # Draw the text with updated content and smaller size
        draw.text((text_x, text_y), title_text, font=font, fill=(255, 255, 255, 255))
        
        # Convert PIL image to numpy array and blend with canvas
        text_array = np.array(text_img)
        
        # Blend the text overlay with the canvas
        for c in range(3):  # RGB channels
            canvas[:, :, c] = canvas[:, :, c] * (1 - text_array[:, :, 3]/255.0) + \
                              text_array[:, :, c] * (text_array[:, :, 3]/255.0)
        
        return canvas
    
    # Create a clip with fixed 3-second duration
    fixed_duration = 3.0
    collage_clip = VideoClip(make_frame=make_collage_frame, duration=fixed_duration)
    collage_clip = collage_clip.set_fps(target_fps)
    
    # Loop the collage clip exactly 3 times
    looped_clips = [collage_clip] * num_loops
    final_collage = concatenate_videoclips(looped_clips)
    
    # Write to file with the same parameters as the original video
    final_collage.write_videofile(
        str(output_path), 
        codec='libx264', 
        fps=target_fps, 
        preset='medium',
        bitrate='5000k',
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    )
    
    # Close all clips
    for clip in video_clips:
        clip.close()
    final_collage.close()
    
    return output_path

def combine_videos(video1_path, video2_path, output_path):
    """Combine two videos sequentially with proper encoding"""
    clip1 = VideoFileClip(str(video1_path))
    clip2 = VideoFileClip(str(video2_path))
    
    # Get properties from the first clip to ensure consistency
    target_width = clip1.w
    target_height = clip1.h
    target_fps = clip1.fps
    
    # Ensure second clip matches the first clip's properties
    if clip2.w != target_width or clip2.h != target_height:
        clip2 = clip2.resize(width=target_width, height=target_height)
    
    print(f"Clip 1: {clip1.w}x{clip1.h} @ {clip1.fps}fps, duration: {clip1.duration}s")
    print(f"Clip 2: {clip2.w}x{clip2.h} @ {clip2.fps}fps, duration: {clip2.duration}s")
    
    # Explicitly create a sequence with compatible properties
    final_clip = concatenate_videoclips([clip1, clip2], method="compose")
    
    # Middle ground compression settings
    final_clip.write_videofile(
        str(output_path),
        codec='libx264',
        fps=min(target_fps, 24),  # Reduce fps if higher than 24
        preset='medium',  # Less aggressive preset
        bitrate='1500k',  # Higher bitrate for better quality
        ffmpeg_params=[
            '-pix_fmt', 'yuv420p',
            '-crf', '26',  # Lower CRF for better quality (18-28 is typical range)
            '-maxrate', '2000k',
            '-bufsize', '2500k'
        ]
    )
    
    clip1.close()
    clip2.close()
    final_clip.close()
    
    return output_path

if __name__ == "__main__":
    ## choosing kicking pair for the main video
    # select_video_pair()

    ## find filenames for other actions
    # find_filenames()

    # Create base comparison video (first stage)
    fname0 = "data/src_EgoExo4D/takes/iiith_soccer_053_6/frame_aligned_videos/downscaled/448/cam02.mp4"
    frames_trim0 = [25, 93, None]

    fname1 = "data/src_EgoExo4D/takes/iiith_soccer_061_6/frame_aligned_videos/downscaled/448/cam02.mp4"
    frames_trim1 = [211, 264, None]
    comparison_path = dir_results / "comparison_0.mp4"
    texts = ["Video A has more hip rotation", "Video A kicks the ball harder"]

    # Create the first stage video
    comparison_path = create_demo_video(
        fname0, 
        fname1, 
        comparison_path, 
        texts, 
        duration=5,
        frames_trim0=frames_trim0,
        frames_trim1=frames_trim1,
        text_color=(0, 0, 0),
        font_style="FONT_HERSHEY_DUPLEX",
        title_box=True
    )
    
    # Create the second stage video (collage)
    fnames = [
               'data/src_humman/p100071_a000701/kinect_color/kinect_009', 
               'data/src_FineDiving/FINADivingWorldCup2021_Women10m_semifinal_r3/17', 
               'data/src_EgoExo4D/takes/unc_basketball_03-31-23_02_8/frame_aligned_videos/downscaled/448/cam01.mp4', 
               'data/src_jigsaws/Knot_Tying/video/Knot_Tying_B001_capture1.mp4', 
               'data/src_EgoExo4D/takes/upenn_0726_Piano_1_2/frame_aligned_videos/downscaled/448/gp02.mp4',
               'data/src_humman/p000560_a000048/kinect_color/kinect_000',
               ]
    fpss = [8.0, 18.0, 30.0, 30.0, 30.0, 8.0]
    frames_trims = [[0, 18, None], [16, 95, None], [73, 205, None], [26, 727, None], [407, 882, None], [0, 26, None]]
    
    collage_path = dir_results / "collage.mp4"
    
    # Now pass the original video source files and frame trims for the background
    create_video_collage(
        fnames, 
        fpss, 
        frames_trims, 
        collage_path, 
        fname0,  # Original source video 1
        fname1,  # Original source video 2
        frames_trim0,  # Frame trim for source video 1 
        frames_trim1,  # Frame trim for source video 2
        num_loops=3
    )
    
    # Create outro with the same style as intro but different text
    # Load the first frames from both videos
    video0 = VideoFileClip(fname0)
    video1 = VideoFileClip(fname1)
    
    # Get frames at the specified frame trim positions
    if frames_trim0 and frames_trim0[0] is not None:
        frame0_time = frames_trim0[0] / video0.fps
    else:
        frame0_time = 0
        
    if frames_trim1 and frames_trim1[0] is not None:
        frame1_time = frames_trim1[0] / video1.fps
    else:
        frame1_time = 0
        
    frame0 = video0.get_frame(frame0_time)
    frame1 = video1.get_frame(frame1_time)
    
    # Get smallest height for consistency
    target_height = min(frame0.shape[0], frame1.shape[0])
    
    # Resize frames if needed
    if frame0.shape[0] != target_height:
        frame0 = resize_frame(frame0, height=target_height)
    if frame1.shape[0] != target_height:
        frame1 = resize_frame(frame1, height=target_height)
    
    # Create combined frame
    combined_frame = np.hstack([frame0, frame1])
    
    # Use same function as in create_demo_video to add text overlay
    def add_text_overlay_outro(frame):
        frame = frame.copy()
        
        # Convert to PIL for better text rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load fonts
        try:
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/Library/Fonts/Arial.ttf",
                "C:/Windows/Fonts/Arial.ttf",
                "C:/Windows/Fonts/Calibri.ttf",
                "C:/Windows/Fonts/Verdana.ttf",
            ]
            
            title_font = None
            for path in font_paths:
                try:
                    title_font = ImageFont.truetype(path, 80)
                    break
                except IOError:
                    continue
            
            if title_font is None:
                raise IOError("No system fonts found")
            
        except IOError:
            # Fall back to default font
            title_font = ImageFont.load_default()
        
        # Outro text
        text_lines = ["Video Action Differencing (ICLR 2025)", "jmhb0.github.io/viddiff/"]
        
        # Measure text sizes
        text_widths = [draw.textlength(line, font=title_font) for line in text_lines]
        text_height = title_font.getbbox(text_lines[0])[3]
        line_spacing = text_height // 2
        
        max_width = max(text_widths)
        total_height = (text_height * len(text_lines)) + (line_spacing * (len(text_lines) - 1))
        
        # Center text
        h, w = frame.shape[:2]
        x_center = w // 2
        y_center = h // 2
        y_start = y_center - (total_height // 2)
        
        # Draw semi-transparent background box
        padding = 40
        box_left = x_center - (max_width // 2) - padding
        box_right = x_center + (max_width // 2) + padding
        box_top = y_start - padding
        box_bottom = y_start + total_height + padding
        
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(
            [(box_left, box_top), (box_right, box_bottom)],
            fill=(0, 0, 0, 180)  # Black with 70% opacity
        )
        
        pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
        draw = ImageDraw.Draw(pil_img)
        
        text_color = (255, 255, 255)  # White
        
        # Draw each line of text
        for i, line in enumerate(text_lines):
            x = x_center - (text_widths[i] // 2)
            y = y_start + (i * (text_height + line_spacing))
            draw.text((x, y), line, font=title_font, fill=text_color)
        
        # Convert back to OpenCV format
        result_img = np.array(pil_img)
        if result_img.shape[2] == 4:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)
        result_frame = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        return result_frame
    
    # Create the outro with the same duration as the intro plus one second
    outro_frame = add_text_overlay_outro(combined_frame)
    frame_duration = 4  # One second longer than intro
    outro_clip = ImageClip(outro_frame).set_duration(frame_duration)
    
    # Save outro to a temporary file
    outro_path = dir_results / "outro.mp4"
    outro_clip.write_videofile(
        str(outro_path),
        codec='libx264',
        fps=24,
        preset='ultrafast',
        bitrate='1000k'
    )
    
    # Close video objects
    video0.close()
    video1.close()
    outro_clip.close()
    
    # Intermediate path for combining comparison and collage
    intermediate_path = dir_results / "intermediate.mp4"
    combine_videos(comparison_path, collage_path, intermediate_path)
    
    # Final output combining all three videos
    final_output_path = dir_results / "final_demo.mp4"
    combine_videos(intermediate_path, outro_path, final_output_path)
    
    print(f"Final demo video created: {final_output_path}")

    
    