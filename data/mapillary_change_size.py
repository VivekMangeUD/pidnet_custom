from PIL import Image
import os

def resize_images(folder_path, output_folder, target_size=(2048, 1024)):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you can add more image extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Open the image file
            with Image.open(input_path) as img:
                # Resize the image
                resized_img = img.resize(target_size)

                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)

if __name__ == "__main__":
    # folder1_path = "/home/aqua/navigation_class/final_project_ws/src/PIDNet/data/mapillary/training/test_script_labels"
    # output_folder_path = "/home/aqua/navigation_class/final_project_ws/src/PIDNet/data/mapillary/training/test_script_images_labels"

    # folder1_path = "/home/aqua/navigation_class/final_project_ws/src/PIDNet/data/mapillary/training/images"
    # output_folder_path = "/home/aqua/navigation_class/final_project_ws/src/PIDNet/data/mapillary/training/images"

    folder1_path = "/home/aqua/navigation_class/final_project_ws/src/PIDNet/data/mapillary/training/labels"
    output_folder_path = "/home/aqua/navigation_class/final_project_ws/src/PIDNet/data/mapillary/training/labels"
    resize_images(folder1_path, output_folder_path)