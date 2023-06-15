import random
import numpy as np
from PIL import Image, ImageDraw

# Define image size
img_size = (500, 500)
# Define box size
box_size = (50, 50)

poses = np.zeros((0,2))
for k in range(50):
    # Create a new image with white background
    img = Image.new('RGB', img_size, "white")
    draw = ImageDraw.Draw(img)


    # Generate random position for the box. Make sure the box doesn't go outside of the image.
    x = np.random.randint(0, img_size[0] - box_size[0])
    y = np.random.randint(0, img_size[1] - box_size[1])


    x_c = x + int(box_size[0]/2)
    y_c = y + int(box_size[1]/2)
    poses = np.concatenate((poses, np.array([[x_c, y_c]])), axis=0)

    pos = (
        x,
        y,
        x + box_size[0],
        y + box_size[1]
    )

    # Draw the box
    draw.rectangle(pos, fill='blue')

    # Save the image
    img.save("data/image_with_box_{}.png".format(k))

    print("Image saved as 'image_with_box_{}.png'".format(k))
np.save('data/poses.npy', poses)