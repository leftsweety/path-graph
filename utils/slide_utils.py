import openslide
import matplotlib.pyplot as plt

def show_slide_info(slide_path, level=0):
    slide = openslide.OpenSlide(slide_path)

    # Print basic properties of the slide
    print("Number of levels: ", slide.level_count)
    print("Dimensions of each level: ", slide.level_dimensions)
    print("Downsampling factors for each level: ", slide.level_downsamples)

    # Read the selected level
    slide_image = slide.read_region((0, 0), level, (512, 512))
    # Convert to a format suitable for displaying with matplotlib
    slide_image = slide_image.convert("RGB")

    # Display the slide
    plt.figure(figsize=(6, 6))
    plt.imshow(slide_image)
    plt.axis('off')
    plt.title(f"Slide Level {level}")
    plt.show()

MATCH_RESPONSE_LABEL = {
    'Non-responder': 0,
    'Responder': 1
}

def match_label(label):
    return MATCH_RESPONSE_LABEL[label]