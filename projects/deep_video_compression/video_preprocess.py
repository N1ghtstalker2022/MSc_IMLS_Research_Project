import cv2
import os
import shutil


# Function to convert video to images
def convert_video_to_images(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Converting {video_path} to images...")
    else:
        print(f"Folder {output_folder} already exists. Skipping conversion.")
        return

    # Open the video file
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    count = 0

    # Read frames from the video and save them as images
    while success:
        # Save the frame as an image
        image_path = os.path.join(output_folder, f"img{count}.png")
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Save as PNG

        # Read the next frame
        success, frame = video.read()
        count += 1

    # Release the video object
    video.release()


# Define a function to extract the numerical part from each string
def extract_number(string):
    return int(string.lstrip("img").rstrip(".png"))


def create_frame_groups():
    frames_folder = (
        "/scratch/zczqyc4/360-videos"  # Path to the folder containing the frames
    )
    group_size = 7
    output_folder = "/scratch/zczqyc4/360-videos-grouped"

    items = os.listdir(frames_folder)
    frame_directories = [
        item for item in items if os.path.isdir(os.path.join(frames_folder, item))
    ]

    frame_directories.sort()  # Sort frames in ascending order
    for frame_directory in frame_directories:
        sub_output_folder = os.path.join(output_folder, frame_directory)
        if os.path.exists(sub_output_folder):
            print(f"Folder {sub_output_folder} already exists.")

        cur_frames = os.listdir(os.path.join(frames_folder, frame_directory))
        cur_frames.sort(key=extract_number)  # Sort frames in ascending order

        for i in range(len(cur_frames) - group_size + 1):
            # Get the current group of frames
            group = cur_frames[i : i + group_size]

            # Create a folder for the group
            if os.path.exists(
                os.path.join(output_folder, frame_directory, f"group{i + 1}")
            ):
                print(
                    f"Folder {os.path.join(output_folder, frame_directory, f'group{i + 1}')} already exists."
                )
                continue
            group_folder = os.path.join(output_folder, frame_directory, f"group{i + 1}")
            os.makedirs(group_folder)

            # Copy the frames to the group folder
            for j, frame in enumerate(group):
                frame_path = os.path.join(frames_folder, frame_directory, frame)
                new_frame_name = f"img{j + 1}.png"  # New name for the frame
                new_frame_path = os.path.join(group_folder, new_frame_name)
                shutil.copy(frame_path, new_frame_path)


if __name__ == "__main__":
    # # Specify the folder path containing the videos
    # videos_folder = "/scratch/zczqyc4/360-videos"
    #
    # # Iterate through the videos in the folder
    # for filename in os.listdir(videos_folder):
    #     if filename.endswith(".mp4"):
    #         # Specify the output folder path for the images
    #         output_folder = os.path.join(videos_folder, filename.split(".")[0])
    #
    #         video_path = os.path.join(videos_folder, filename)
    #         # Convert the video to images
    #         convert_video_to_images(video_path, output_folder)

    create_frame_groups()
