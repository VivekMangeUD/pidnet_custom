import os

def create_train_lst(images_folder, labels_folder, output_file):
    images_list = sorted(os.listdir(images_folder))
    labels_list = sorted(os.listdir(labels_folder))
    print('image list ', len(images_list))
    print('label list ', len(labels_list))
    

    with open(output_file, 'w') as f:
        for image_name, label_name in zip(images_list, labels_list):
            image_path = os.path.join("test_images", image_name)
            label_path = os.path.join("test_labels", label_name)
            f.write(f"{image_path} {label_path}\n")

if __name__ == "__main__":
    # Specify the folder paths for images and labels
    images_folder = 'test_images'
    labels_folder = 'test_labels'

    # Specify the output train.lst file
    output_file = 'test.lst'

    # Generate train.lst file
    create_train_lst(images_folder, labels_folder, output_file)

    print(f"File list has been written to {output_file}.")

