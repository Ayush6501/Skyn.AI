import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image, ImageOps

pd.options.mode.chained_assignment = None  # default='warn'

# main


def skin_detection(img_path):
    original = read_image(img_path)
    images = image_conversions2(original)
    height, width = skin_predict2(images)
    dframe, dframe_removed = dataframe2(images)
    skin_cluster_row, skin_cluster_label = skin_cluster(dframe)
    cluster_label_mat = cluster_matrix(
        dframe, dframe_removed, skin_cluster_label, height, width)
    final_segment2(images, cluster_label_mat)
    # display_all_images(images)
    # # write_all_images(images)
    # skin_cluster_row = np.delete(skin_cluster_row, 1)
    # skin_cluster_row = np.delete(skin_cluster_row, 2)
    return np.delete(skin_cluster_row, -1)
    # return images["final_segment"]


# Plot Histogram and Threshold values
def plot_histogram(histogram, bin_edges, Totsu, Tmax, Tfinal):
    plt.figure()
    plt.title("Image Histogram")
    plt.xlabel("pixel value")
    plt.ylabel("pixel frequency")
    plt.xlim([0, 256])
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.axvline(x=Tmax, label="Tmax", color='red', linestyle="--")
    plt.axvline(x=Totsu, label="Totsu", color='green', linestyle="--")
    plt.axvline(x=Tfinal, label="Tfinal", color='yellow', linestyle="-")
    plt.legend()
    plt.show()

# display an image plus label and wait for key press to continue


def display_image(image, name):
    window_name = name
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def display_image2(image, name):
    # Convert the image to a Pillow image if it isn't already
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Display the image
    image.show(title=name)

# Display all images


def display_all_images(images):
    for key, value in images.items():
        display_image2(value, key)

# write all images


def write_all_images(images):
    for key, value in images.items():
        cv2.imwrite(key+'.jpg', value)

# read in image into openCV


def read_image(dir):
    maxwidth, maxheight = 400, 500
    image_path = dir

    # Open the image using Pillow
    img = Image.open(image_path)

    # Calculate the resizing factor
    f1 = maxwidth / img.width
    f2 = maxheight / img.height
    f = min(f1, f2)  # resizing factor

    # Resize the image
    new_size = (int(img.width * f), int(img.height * f))
    img = img.resize(new_size, Image.LANCZOS)

    # Convert to BGR format (if needed for further OpenCV processing)
    img_BGR = img.convert("RGB")

    return img_BGR


def thresholding(images):
    histogram, bin_edges = np.histogram(
        images["grayscale"].ravel(), 256, [0, 256])
    Totsu, threshold_image_otsu = cv2.threshold(
        images["grayscale"], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    Tmax = np.where(histogram[:] == max(histogram[:]))[0][0]
    Tfinal = round((Tmax + Totsu)/2) if Tmax > 10 else round((Tmax + Totsu)/4)

    plot_histogram(histogram, bin_edges, Totsu, Tmax, Tfinal)

    threshold_type = (cv2.THRESH_BINARY if Tmax <
                      220 else cv2.THRESH_BINARY_INV)
    Tfinal, threshold_image = cv2.threshold(
        images["grayscale"], Tfinal, 255, threshold_type)

    masked_img = cv2.bitwise_and(
        images["BGR"], images["BGR"], mask=threshold_image)
    return masked_img


# Grayscle and Thresholding and HSV & YCrCb color space conversions
def image_conversions(img_BGR):
    images = {
        "BGR": img_BGR,
        "grayscale": cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    }
    images["thresholded"] = thresholding(images)
    images["HSV"] = cv2.cvtColor(
        images["thresholded"], cv2.COLOR_BGR2HSV)
    images["YCrCb"] = cv2.cvtColor(
        images["thresholded"], cv2.COLOR_BGR2YCrCb)
    # display_all_images(images)
    return images


from PIL import Image, ImageOps
import numpy as np

def image_conversions2(img_BGR):
    # Convert to grayscale
    grayscale = ImageOps.grayscale(img_BGR)

    # Convert grayscale image to a numpy array for further processing
    grayscale_np = np.array(grayscale)

    # Calculate histogram and bin edges
    histogram, bin_edges = np.histogram(grayscale_np.ravel(), 256, [0, 256])

    # Calculate Otsu's threshold using numpy (since cv2 is not used anymore)
    Totsu = otsu_threshold(grayscale_np)

    Tmax = np.argmax(histogram)
    Tfinal = round((Tmax + Totsu) / 2) if Tmax > 10 else round((Tmax + Totsu) / 4)

    plot_histogram(histogram, bin_edges, Totsu, Tmax, Tfinal)

    # Determine the threshold type based on Tmax
    threshold_type = 'BINARY' if Tmax < 220 else 'BINARY_INV'

    # Apply the final threshold
    threshold_image = ImageOps.invert(grayscale) if threshold_type == 'BINARY_INV' else grayscale.point(lambda p: 255 if p > Tfinal else 0)
    thresholded_np = np.array(threshold_image)

    # Create a mask and apply it to the original image
    masked_img = Image.fromarray(np.array(img_BGR) * (thresholded_np[:, :, None] > 0).astype(np.uint8))

    # Convert images to HSV and YCrCb
    hsv = masked_img.convert("HSV")
    ycrcb = masked_img.convert("YCbCr")

    # Store all images in a dictionary
    images = {
        "BGR": img_BGR,
        "grayscale": grayscale,
        "thresholded": threshold_image,
        "HSV": hsv,
        "YCrCb": ycrcb
    }

    return images

def otsu_threshold(grayscale_np):
    """Calculate Otsu's threshold manually using numpy."""
    pixel_counts, bin_edges = np.histogram(grayscale_np.ravel(), bins=256, range=(0, 256))
    total_pixels = grayscale_np.size
    sum_total = np.dot(np.arange(256), pixel_counts)
    sumB, wB, wF, max_variance, threshold = 0, 0, 0, 0, 0

    for i in range(256):
        wB += pixel_counts[i]
        if wB == 0:
            continue
        wF = total_pixels - wB
        if wF == 0:
            break
        sumB += i * pixel_counts[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        variance_between = wB * wF * (mB - mF) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            threshold = i

    return threshold



# Predict skin pixels
def skin_predict(images):
    height, width = images["grayscale"].shape
    images["skin_predict"] = np.empty_like(images["grayscale"])
    images["skin_predict"][:] = images["grayscale"]

    for i in range(height):
        for j in range(width):
            if((images["HSV"].item(i, j, 0) <= 170) and (140 <= images["YCrCb"].item(i, j, 1) <= 170) and (90 <= images["YCrCb"].item(i, j, 2) <= 120)):
                images["skin_predict"][i, j] = 255
            else:
                images["skin_predict"][i, j] = 0
    return height, width


def skin_predict2(images):
    # Convert HSV and YCrCb images to numpy arrays for pixel-wise processing
    hsv_np = np.array(images["HSV"])
    ycrcb_np = np.array(images["YCrCb"])

    # Initialize the skin_predict image with the same dimensions as grayscale
    skin_predict = np.zeros_like(np.array(images["grayscale"]))

    # Iterate through each pixel and apply the skin color detection logic
    for i in range(hsv_np.shape[0]):
        for j in range(hsv_np.shape[1]):
            if (hsv_np[i, j, 0] <= 170 and
                140 <= ycrcb_np[i, j, 1] <= 170 and
                90 <= ycrcb_np[i, j, 2] <= 120):
                skin_predict[i, j] = 255
            else:
                skin_predict[i, j] = 0

    # Store the skin prediction result in the images dictionary
    images["skin_predict"] = Image.fromarray(skin_predict)

    return skin_predict.shape


def dataframe2(images):
    dframe = pd.DataFrame()

    # Convert the HSV and YCrCb images to numpy arrays
    hsv_np = np.array(images["HSV"])
    ycrcb_np = np.array(images["YCrCb"])
    skin_predict_np = np.array(images["skin_predict"])

    # Reshape the HSV and YCrCb channels for DataFrame
    dframe['H'] = hsv_np.reshape([-1, 3])[:, 0]
    dframe['Cr'] = ycrcb_np.reshape([-1, 3])[:, 1]
    dframe['Cb'] = ycrcb_np.reshape([-1, 3])[:, 2]

    # Convert the thresholded image to grayscale using Pillow
    gray = ImageOps.grayscale(images["thresholded"])
    gray_np = np.array(gray)

    # Get the y-x coordinates of all pixels
    yx_coords = np.column_stack(np.where(gray_np >= 0))
    dframe['Y'] = yx_coords[:, 0]
    dframe['X'] = yx_coords[:, 1]

    # Flatten the skin_predict array and add it to the DataFrame
    dframe['I'] = skin_predict_np.flatten()

    # Remove black pixels - which are already segmented
    dframe_removed = dframe[dframe['H'] == 0]
    dframe = dframe[dframe['H'] != 0]

    return dframe, dframe_removed




def dataframe(images):
    dframe = pd.DataFrame()
    dframe['H'] = images["HSV"].reshape([-1, 3])[:, 0]

    # Getting the y-x coordinated
    gray = cv2.cvtColor(images["thresholded"], cv2.COLOR_BGR2GRAY)
    yx_coords = np.column_stack(np.where(gray >= 0))
    dframe['Y'] = yx_coords[:, 0]
    dframe['X'] = yx_coords[:, 1]

    dframe['Cr'] = images["YCrCb"].reshape([-1, 3])[:, 1]
    dframe['Cb'] = images["YCrCb"].reshape([-1, 3])[:, 2]
    dframe['I'] = images["skin_predict"].reshape(
        [1, images["skin_predict"].size])[0]

    # Remove Black pixels - which are already segmented
    dframe_removed = dframe[dframe['H'] == 0]
    dframe.drop(dframe[dframe['H'] == 0].index, inplace=True)
    return dframe, dframe_removed

# cluster skin pixels using K-means


def skin_cluster(dframe):
    # K-means
    kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=5,
        max_iter=100,
        random_state=69
    )
    kmeans.fit(dframe)

    # Get the cluster centers
    km_cc = kmeans.cluster_centers_

    # Find the skin cluster label - which has the highest average 'I' value
    skin_cluster_row = km_cc[np.argmax(km_cc[:, -1])]
    skin_cluster_label = np.argmax(km_cc[:, -1])

    # Add cluster labels to the dataframe
    dframe['cluster'] = kmeans.labels_

    return skin_cluster_row, skin_cluster_label


# Append removed pixels to the dataframe and get cluster matrix
def cluster_matrix(dframe, dframe_removed, skin_cluster_label, height, width):
    dframe_removed['cluster'] = np.full((len(dframe_removed.index), 1), -1)
    dframe = pd.concat([dframe, dframe_removed], ignore_index=False).sort_index()
    dframe['cluster'] = (dframe['cluster'] ==
                         skin_cluster_label).astype(int) * 255
    cluster_label_mat = np.asarray(
        dframe['cluster'].values.reshape(height, width), dtype=np.uint8)
    return cluster_label_mat

# final segmentation


def final_segment(images, cluster_label_mat):
    final_segment_img = cv2.bitwise_and(
        images["BGR"], images["BGR"], mask=cluster_label_mat)
    images["final_segment"] = final_segment_img
    # display_image(final_segment_img, "final segmentation")


def final_segment2(images, cluster_label_mat):
    # Convert images and mask to numpy arrays
    img_bgr_np = np.array(images["BGR"])
    mask_np = np.array(cluster_label_mat)

    # Ensure the mask is binary (0 or 255)
    mask_binary = (mask_np > 0).astype(np.uint8) * 255

    # Apply the mask to the BGR image
    final_segment_np = img_bgr_np * mask_binary[:, :, np.newaxis]

    # Convert the result back to a Pillow image
    images["final_segment"] = Image.fromarray(final_segment_np)

    return images


skin_detection("./public/test images/brendon_urie_3.jpeg")
