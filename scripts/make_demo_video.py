"""
python -m ipdb scripts/make_demo_video.py
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

from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, ImageClip
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
    for row in dataset:
        if 'deadlift' in row['action_description']:
            print("deadlift action name is", row['action'])
            break
    action_names = ['fitness_2','ballsports_1', 'music_0', 'diving_0', 'surgery_0']
    seen_actions = set()
    fnames, fpss, frames_trims = [], [], []
    for i, row in enumerate(dataset):
        if row['action'] in seen_actions:
            continue
        seen_actions.add(row['action'])
        if row['action'] in action_names:
            vids = row['videos']
            fnames.append(vids[0]['path'])
            fpss.append(vids[0]['fps_original'])
            frames_trims.append(vids[0]['frames_trim'])
    print(fnames)
    print(fpss)
    print(frames_trims)
    ipdb.set_trace()
    pass

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
            text_lines = ["Video Action Differencing (VidDiff):", "let's compare these videos"]
            
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
            # For other phases: text in top left corner
            text_lines = ["Video Action Differencing (VidDiff):", "let's compare these videos"]
            
            # Adjusted spacing for smaller font
            line_spacing = 35  # Reduced for smaller font
            
            # Draw header text
            for i, line in enumerate(text_lines):
                draw.text((20, 15 + i * line_spacing), line, font=regular_font, fill=(0, 0, 0))
            
            # Add Video A and B labels at the bottom
            video_a_text = "Video A"
            video_b_text = "Video B"
            
            # Calculate positions for video labels
            video_a_width = draw.textlength(video_a_text, font=regular_font)
            video_b_width = draw.textlength(video_b_text, font=regular_font)
            
            video_a_x = (w // 4) - (video_a_width // 2)
            video_b_x = (3 * w // 4) - (video_b_width // 2)
            
            draw.text((video_a_x, h - 45), video_a_text, font=regular_font, fill=(0, 0, 0))
            draw.text((video_b_x, h - 45), video_b_text, font=regular_font, fill=(0, 0, 0))
        
        # Add difference texts only if requested
        if show_diff_texts:
            # Center the difference texts at the bottom
            diff_y_start = h - 120  # Moved up a bit for smaller font
            diff_spacing = 30  # Reduced spacing for smaller font
            
            for i, text in enumerate(texts):
                text_width = draw.textlength(text, font=regular_font)
                diff_x = (w // 2) - (text_width // 2)
                diff_y = diff_y_start + (i * diff_spacing)
                
                draw.text((diff_x, diff_y), text, font=regular_font, fill=(0, 0, 0))
        
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


if __name__ == "__main__":
    # select_video_pair()

    # find filenames for other actions
    find_filenames()

    # create video
    # fname0 = "data/src_EgoExo4D/takes/iiith_soccer_027_6/frame_aligned_videos/downscaled/448/cam02.mp4"
    # fname1 = "data/src_EgoExo4D/takes/iiith_soccer_040_6/frame_aligned_videos/downscaled/448/cam02.mp4"
    
    
    # first stage videos
    fname0 = "data/src_EgoExo4D/takes/iiith_soccer_053_6/frame_aligned_videos/downscaled/448/cam02.mp4"
    frames_trim0 = [25, 93, None]

    fname1 = "data/src_EgoExo4D/takes/iiith_soccer_061_6/frame_aligned_videos/downscaled/448/cam02.mp4"
    frames_trim1 = [211, 264, None]
    output_path = dir_results / "comparison_0.mp4"
    texts = ["Video A has more hip rotation", "Video A kicks the ball harder"]

    # second stage videos
    fnames = ['data/src_humman/p100071_a000701/kinect_color/kinect_009', 'data/src_FineDiving/FINADivingWorldCup2021_Women10m_semifinal_r3/17', 'data/src_jigsaws/Knot_Tying/video/Knot_Tying_B001_capture1.mp4', 'data/src_EgoExo4D/takes/unc_basketball_03-31-23_02_8/frame_aligned_videos/downscaled/448/cam01.mp4', 'data/src_EgoExo4D/takes/upenn_0726_Piano_1_2/frame_aligned_videos/downscaled/448/gp02.mp4']
    fpss = [8.0, 12.0, 30.0, 30.0, 30.0]
    frames_trims = [[0, 18, None], [16, 95, None], [26, 727, None], [73, 205, None], [407, 882, None]]
    
    # Example with custom text color and font
    output_path = create_demo_video(
        fname0, 
        fname1, 
        output_path, 
        texts, 
        duration=5,
        frames_trim0=frames_trim0,
        frames_trim1=frames_trim1,
        # text_color=(0, 255, 255),  # Yellow text
        text_color=(0, 0, 0),  # Yellow text
        # font_style="FONT_HERSHEY_COMPLEX"
        font_style="FONT_HERSHEY_DUPLEX",
        title_box=True
    )

    
    
    # fond options
    # FONT_HERSHEY_SIMPLEX: Normal size sans-serif font
    # FONT_HERSHEY_PLAIN: Small size sans-serif font
    # FONT_HERSHEY_DUPLEX: More complex normal size sans-serif font
    # FONT_HERSHEY_COMPLEX: Normal size serif font
    # FONT_HERSHEY_TRIPLEX: More complex normal size serif font
    # FONT_HERSHEY_COMPLEX_SMALL: Smaller version of complex font
    # FONT_HERSHEY_SCRIPT_SIMPLEX: Hand-writing style font
    # FONT_HERSHEY_SCRIPT_COMPLEX: More complex variant of the script font
    # These options give you good flexibility to match the style of your project while keeping the video file size small.

    
    