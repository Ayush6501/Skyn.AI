import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None  # default='warn'


def skin_detection(img_path):
    images = image_conversions(read_image(img_path))
    height, width = images["grayscale"].shape

    dframe, dframe_removed = prepare_dataframe(images)
    skin_cluster_row, skin_cluster_label = cluster_skin_pixels(dframe)

    cluster_label_mat = create_cluster_matrix(dframe, dframe_removed, skin_cluster_label, height, width)
    final_segmentation(images, cluster_label_mat)

    display_all_images(images)

    return np.delete(skin_cluster_row, [1, 2, -1])


def plot_histogram(histogram, bin_edges, Totsu, Tmax, Tfinal):
    plt.figure()
    plt.title("Image Histogram")
    plt.xlabel("pixel value")
    plt.ylabel("pixel frequency")
    plt.xlim([0, 256])
    plt.plot(bin_edges[:-1], histogram)
    plt.axvline(x=Tmax, label="Tmax", color='red', linestyle="--")
    plt.axvline(x=Totsu, label="Totsu", color='green', linestyle="--")
    plt.axvline(x=Tfinal, label="Tfinal", color='yellow', linestyle="-")
    plt.legend()
    plt.show()


def display_image(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_all_images(images):
    for key, value in images.items():
        display_image(value, key)


def read_image(img_path):
    maxwidth, maxheight = 400, 500
    img_BGR = cv2.imread(img_path, cv2.IMREAD_COLOR)
    f = min(maxwidth / img_BGR.shape[1], maxheight / img_BGR.shape[0])
    return cv2.resize(img_BGR, (int(img_BGR.shape[1] * f), int(img_BGR.shape[0] * f)))


def thresholding(images):
    histogram, bin_edges = np.histogram(images["grayscale"].ravel(), 256, [0, 256])
    Totsu, _ = cv2.threshold(images["grayscale"], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    Tmax = np.argmax(histogram)
    Tfinal = round((Tmax + Totsu) / 2) if Tmax > 10 else round((Tmax + Totsu) / 4)

    plot_histogram(histogram, bin_edges, Totsu, Tmax, Tfinal)

    threshold_type = cv2.THRESH_BINARY_INV if Tmax > 220 else cv2.THRESH_BINARY
    _, threshold_image = cv2.threshold(images["grayscale"], Tfinal, 255, threshold_type)

    return cv2.bitwise_and(images["BGR"], images["BGR"], mask=threshold_image)


def image_conversions(img_BGR):
    grayscale = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    thresholded = thresholding({"BGR": img_BGR, "grayscale": grayscale})

    return {
        "BGR": img_BGR,
        "grayscale": grayscale,
        "thresholded": thresholded,
        "HSV": cv2.cvtColor(thresholded, cv2.COLOR_BGR2HSV),
        "YCrCb": cv2.cvtColor(thresholded, cv2.COLOR_BGR2YCrCb),
        "skin_predict": predict_skin_pixels(thresholded, grayscale)
    }


def predict_skin_pixels(thresholded, grayscale):
    height, width = grayscale.shape
    skin_predict = np.zeros_like(grayscale)

    HSV = cv2.cvtColor(thresholded, cv2.COLOR_BGR2HSV)
    YCrCb = cv2.cvtColor(thresholded, cv2.COLOR_BGR2YCrCb)

    skin_mask = (
            (HSV[..., 0] <= 170) &
            (140 <= YCrCb[..., 1]) & (YCrCb[..., 1] <= 170) &
            (90 <= YCrCb[..., 2]) & (YCrCb[..., 2] <= 120)
    )
    skin_predict[skin_mask] = 255

    return skin_predict


def prepare_dataframe(images):
    H = images["HSV"][..., 0].ravel()
    Y, X = np.where(cv2.cvtColor(images["thresholded"], cv2.COLOR_BGR2GRAY) >= 0)
    Cr = images["YCrCb"][..., 1].ravel()
    Cb = images["YCrCb"][..., 2].ravel()
    I = images["skin_predict"].ravel()

    dframe = pd.DataFrame({'H': H, 'Y': Y, 'X': X, 'Cr': Cr, 'Cb': Cb, 'I': I})
    dframe_removed = dframe[dframe['H'] == 0]
    dframe = dframe[dframe['H'] != 0]

    return dframe, dframe_removed


def cluster_skin_pixels(dframe):
    kmeans = KMeans(init="random", n_clusters=3, n_init=5, max_iter=100, random_state=42)
    kmeans.fit(dframe)

    skin_cluster_row = max(kmeans.cluster_centers_, key=lambda x: x[-1])
    skin_cluster_label = np.argmax(np.allclose(kmeans.cluster_centers_, skin_cluster_row))

    dframe['cluster'] = kmeans.labels_

    return skin_cluster_row, skin_cluster_label


def create_cluster_matrix(dframe, dframe_removed, skin_cluster_label, height, width):
    dframe_removed['cluster'] = -1
    dframe = pd.concat([dframe, dframe_removed]).sort_index()

    cluster_label_mat = (dframe['cluster'] == skin_cluster_label).astype(np.uint8).values.reshape(height, width) * 255
    return cluster_label_mat


def final_segmentation(images, cluster_label_mat):
    images["final_segment"] = cv2.bitwise_and(images["BGR"], images["BGR"], mask=cluster_label_mat)


# Example usage
# print(skin_detection("images/Optimized-selfieNig-cropped.jpg"))
