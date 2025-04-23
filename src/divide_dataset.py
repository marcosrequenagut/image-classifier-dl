import os 
import shutil
import random

def divide_dataset(path, fraction):
    # Load the images
    images = [image for image in os.listdir(path) if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Shuffle the images
    random.shuffle(images)

    # Number of images in each directory
    num_images = int(len(images) * (fraction / 100))

    # Create the main directories for training and testing
    main_train_dir = "data2_train"
    main_test_dir = "data2_test"

    os.makedirs(main_train_dir, exist_ok=True)
    os.makedirs(main_test_dir, exist_ok=True)

    # Create subdirectories for each category inside train and test
    category_name = os.path.basename(path)  # Get the category name (e.g., 'cloudy')
    train_category_dir = os.path.join(main_train_dir, category_name)
    test_category_dir = os.path.join(main_test_dir, category_name)

    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(test_category_dir, exist_ok=True)

    # Move the images into each subdirectory
    for i, image in enumerate(images):
        if i < num_images:
            shutil.move(os.path.join(path, image), os.path.join(train_category_dir, image))
        else:
            shutil.move(os.path.join(path, image), os.path.join(test_category_dir, image))

    print(f"Divided images: {num_images} en {train_category_dir}, {len(images) - num_images} en {test_category_dir}")

# Example usage for each category
divide_dataset(r"data\cloudy", fraction = 80)
divide_dataset(r"data\desert", fraction = 80)
divide_dataset(r"data\green_area", fraction = 80)
divide_dataset(r"data\water", fraction = 80)
