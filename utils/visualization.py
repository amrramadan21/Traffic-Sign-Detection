import matplotlib.pyplot as plt

def compare_filters(image, gaussian, median):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15,5))

    # Original
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')

    # Gaussian
    plt.subplot(1,3,2)
    plt.imshow(gaussian)
    plt.title("Gaussian")
    plt.axis('off')

    # Median
    plt.subplot(1,3,3)
    plt.imshow(median)
    plt.title("Median")
    plt.axis('off')

    plt.show()