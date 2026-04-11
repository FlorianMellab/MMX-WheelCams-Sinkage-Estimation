import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from edge_detection_front import read_image, visualize, preprocess, perform_morphological_transform
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import scipy.io
from scipy.stats import pearsonr

"This script segments an image based on texture and is optimzed for front wheel images. This script works by applying a bank of Gabor filters, extracting pixel-wise responses, and clustering them with K-means. The resulting texture groups are visualized and can be refined with morphological operations and saved as a mask for further analysis."

def build_gabor_filters(ksize, sigmas, lambdas, thetas):
    filters = []
    filter_params = []
    for theta in thetas:
        for sigma in sigmas:
            for lamda in lambdas:
                real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma=0.5, psi=0)
                imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma=0.5, psi=np.pi/2)
                filters.append((real, imag))
                filter_params.append((sigma, lamda, theta))
    return filters, filter_params

def rank_gabor_filters_AUC(img, mask, filters, filter_params):
    """
    Computes ROC_AUC for each Gabor filter magnitude response.
    Returns a sorted list of filters with their scores.
    """
    y_true = mask.flatten().astype(np.uint8)
    results = []

    for i, ((real_kernel, imag_kernel), params) in enumerate(zip(filters, filter_params)):
        sigma, lamda, theta = params

        # Filter image with real+imag kernels
        real_resp = cv2.filter2D(img, cv2.CV_32F, real_kernel)
        imag_resp = cv2.filter2D(img, cv2.CV_32F, imag_kernel)

        # Magnitude response
        magnitude = np.sqrt(real_resp**2 + imag_resp**2).flatten()

        # Normalize for ROC–AUC stability
        magnitude = MinMaxScaler().fit_transform(magnitude.reshape(-1,1)).ravel()

        # Compute AUC
        auc = roc_auc_score(y_true, magnitude)

        results.append({
            "index": i,
            "sigma": sigma,
            "lambda": lamda,
            "theta": theta,
            "auc": auc
        })

        # print(f"Filter {i:02d}: sigma={sigma}, lambda={lamda}, theta={theta:.2f} → AUC={auc:.4f}")

    # Sort by AUC descending
    results = sorted(results, key=lambda x: x["auc"], reverse=True)
    return results

def rank_gabor_filters_pearson(img, mask, filters, filter_params):
    """
    Computes Pearson correlation between each Gabor filter magnitude response
    and the ground-truth mask.
    Returns a sorted list of filters with their Pearson coefficients.
    """
    y_true = mask.flatten().astype(np.uint8)
    results = []

    for i, ((real_kernel, imag_kernel), params) in enumerate(zip(filters, filter_params)):
        sigma, lamda, theta = params

        # Filter image with real+imag kernels
        real_resp = cv2.filter2D(img, cv2.CV_32F, real_kernel)
        imag_resp = cv2.filter2D(img, cv2.CV_32F, imag_kernel)

        # Magnitude response
        magnitude = np.sqrt(real_resp**2 + imag_resp**2).flatten()

        # Normalize magnitude for stability
        magnitude = MinMaxScaler().fit_transform(magnitude.reshape(-1,1)).ravel()

        # Compute Pearson correlation
        pearson_coef, _ = pearsonr(y_true, magnitude)

        results.append({
            "index": i,
            "sigma": sigma,
            "lambda": lamda,
            "theta": theta,
            "pearson_coef": pearson_coef
        })

        # print(f"Filter {i:02d}: sigma={sigma}, lambda={lamda}, theta={theta:.2f} → Pearson r={pearson_coef:.4f}")

    # Sort by absolute Pearson coefficient descending
    results = sorted(results, key=lambda x: abs(x["pearson_coef"]), reverse=True)
    return results

if __name__ == "__main__":
    img = read_image(r"Donatien segmentation mask/PP_F_1154_57.3487.png")
    # img = read_image("images/high_sinkage_front.png")
    img_clahe = preprocess(img)
    print(img.shape)
    img_clahe = cv2.GaussianBlur(img_clahe, (7, 7), 2)  #Smoothing to ignore fine textures
    # img_clahe = cv2.resize(img_clahe, (0,0), fx=0.5, fy=0.5)
    # visualize(img)

    #Gabor filter parameters
    ksize = 21
    sigmas = (50,100)
    lambdas = (100,200,300)
    thetas = np.arange(0, np.pi, np.pi/8)

    #Clustering parameters
    k = 5  # number of texture groups
    n_init = 5 #K_means is run n_init times and the best partition is kept

    filters, filter_params = build_gabor_filters(ksize, sigmas, lambdas, thetas)

    mask = np.load(r"Donatien segmentation mask/mask.npy")
    plt.figure(figsize=(6,6))
    plt.imshow(mask, cmap='gray')
    plt.title("Wheel Mask")
    plt.axis('off')
    plt.show()
    rankings = rank_gabor_filters_pearson(img_clahe, mask, filters, filter_params)

    #     # Get PCA components (PC1, PC2, PC3)
    # pcs = pca.components_[:3]  # shape: (3, n_features)

    # # Generate feature names dynamically
    # feature_names = [f"gabor_{i}" for i in range(features.shape[1])]

    # # Threshold
    # threshold = 0.01

    # for i, pc in enumerate(pcs, start=1):
    #     low_weight_indices = np.where(np.abs(pc) < threshold)[0]
    #     low_weight_features = [feature_names[idx] for idx in low_weight_indices]
    #     print(f"PC{i} - features with |weight| < {threshold}:")
    #     print(low_weight_features)
    #     print(f"Number of features below threshold: {len(low_weight_features)}\n")

    # low_weight_indices = np.array([3, 9, 10, 11, 16, 17, 18, 19, 22, 23, 25, 26, 27, 33, 34, 35, 40, 41, 42, 43, 49, 50, 51, 59])
    
    # print("Low-weighted Gabor filters and their parameters:")

    # for idx in low_weight_indices:
    #     sigma, lamda, theta = filter_params[idx]
    #     theta_deg = np.degrees(theta)
    #     print(f"Filter {idx}: sigma={sigma}, lambda={lamda}, theta={theta_deg:.2f} deg")
    
    # useful_indices = [i for i in range(len(filters)) if i not in low_weight_indices]
    # filters = [filters[i] for i in useful_indices]


    #Apply filters and record responses
    responses = []
    for (real_kernel, imag_kernel) in filters:
        real_response = cv2.filter2D(img_clahe, ddepth=cv2.CV_32F, kernel=real_kernel)
        imag_response = cv2.filter2D(img_clahe, ddepth=cv2.CV_32F, kernel=imag_kernel)
        magnitude = np.sqrt(real_response**2+imag_response**2)
        responses.append(magnitude.reshape(-1))

    #Combine responses into a feature matrix (n_pixels x n_features)
    features = np.array(responses).T

    pca = PCA(n_components = 3)
    X_pca = pca.fit_transform(features)
    auc_pc1 = roc_auc_score(mask.flatten(), X_pca[:,0])
    pearson_pc1 = pearsonr(X_pca[:,0], mask.flatten())

    print("auc pc1", auc_pc1)
    print("pearson_pc1", pearson_pc1)

    kmeans = KMeans(n_clusters=k, random_state=15, n_init=n_init) 
    labels = kmeans.fit_predict(features)
    segmented = labels.reshape(img_clahe.shape)

    #Clustering on PC1 only
    # kmeans = KMeans(n_clusters=k, random_state=15, n_init=n_init)
    # labels = kmeans.fit_predict(X_pca[:, 0].reshape(-1,1))
    # segmented = labels.reshape(img_clahe.shape)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance captured:", np.sum(pca.explained_variance_ratio_))

    # fig = plt.figure(figsize=(7, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
    #         c=labels, s=5, cmap='viridis')

    # ax.set_title("PCA of Pixel Texture Features (3D) with K-means Clustering")
    # ax.set_xlabel("PC1")
    # ax.set_ylabel("PC2")
    # ax.set_zlabel("PC3")

    # plt.show()

    # #Visualize a texture group
    # texture_group = 1 
    # mask = (segmented == texture_group).astype(np.uint8)
    # kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))  # Elliptical kernel keeps more natural shape
    # kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,8))  # Elliptical kernel keeps more natural shape
    # mask = perform_morphological_transform(mask, kernel_closing, kernel_opening)

    plt.figure(figsize=(15, 6))

    # Original image
    plt.subplot(1, k+1, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot each texture group
    for i in range(k):
        mask = (segmented == i).astype(np.uint8)
        plt.subplot(1, k+1, i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Texture Group {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Plot each texture group
    for i in range(k):
        mask = (segmented == i).astype(np.uint8)
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # Elliptical kernel keeps more natural shape
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # Elliptical kernel keeps more natural shape
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_closing, iterations=1)
        opening  = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening, iterations=1)
        mask = opening
        plt.subplot(1, k+1, i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Texture Group {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    #Save the obtained mask for further processing
    # mask_texture = (segmented == k-2).astype(np.uint8)
    # plt.imshow(mask_texture, cmap='gray')
    # plt.title(f'Texture 1')
    # plt.axis('off')
    # plt.show()
    # texture_mask = {}
    # texture_mask["image"] = img_clahe
    # texture_mask["filters"] = filters
    # texture_mask["clustering"] = (k, n_init)
    # texture_mask["mask"] = mask_texture

    # print(texture_mask)
    # np.save("texture_mask.npy", mask_texture)
    
    #AND comibnation of 2 groups
    # mask_texture_2 = (segmented == k-3).astype(np.uint8)
    # plt.imshow(mask_texture_2, cmap='gray')
    # plt.title(f'Texture 2')
    # plt.axis('off')
    # plt.show()

    # mask_texture_final = mask_texture | mask_texture_2

    # plt.imshow(mask_texture_final, cmap='gray')
    # plt.title(f'Texture OR combination')
    # plt.axis('off')
    # plt.show()

    for i in range(k):
        plt.imshow((segmented == k-1-i).astype(np.uint8), cmap='gray')
        plt.title(f'Texture group {i}')
        plt.axis('off')
        plt.show()

    #Visualize principle component overlay with original image. Should see clear texture identification.
    for i in range(3):
        pc1_image = X_pca[:,i].reshape(img_clahe.shape)
        plt.imshow(pc1_image, cmap='jet')
        plt.title("PC1 Scores Overlay")
        plt.colorbar()
        plt.show()

    # -----------------------------
    # 2️⃣ Compute PC1 weights
    # -----------------------------
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(features)
    pc1_weights = np.abs(pca.components_[0])  # absolute contribution of each filter

    # Rank filters by PC1 weight
    pc1_ranking = np.argsort(pc1_weights)[::-1]  # descending order

    N = 10
    top_filters_pc1 = pc1_ranking[:N]

    print("Top filters based on PC1 contribution:")
    for rank, idx in enumerate(top_filters_pc1, start=1):
        sigma, lamda, theta = filter_params[idx]
        theta_deg = np.degrees(theta)
        weight = pc1_weights[idx]
        print(f"{rank:02d}: Filter {idx} → sigma={sigma}, lambda={lamda}, theta={theta_deg:.1f} deg, PC1 weight={weight:.4f}")


    # -----------------------------
    # Step 1: Compute PC1 weights
    # -----------------------------
    pca = PCA(n_components=1)
    X_pca_full = pca.fit_transform(features)
    pc1_weights = np.abs(pca.components_[0])  # absolute contribution of each filter

    # Rank filters by PC1 weight
    pc1_ranking = np.argsort(pc1_weights)[::-1]  # descending order

    # -----------------------------
    # Step 2: Select top N filters
    # -----------------------------
    N = 10
    top_pc1_indices = pc1_ranking[:N]

    # -----------------------------
    # Step 3: Build feature matrix with top 10 PC1 filters
    # -----------------------------
    features_top_pc1 = features[:, top_pc1_indices]

    # Optional: check shapes
    print("Feature matrix with top PC1 filters:", features_top_pc1.shape)

    # -----------------------------
    # Step 4: Run K-means clustering
    # -----------------------------
    n_init = 5
    kmeans = KMeans(n_clusters=k, random_state=15, n_init=n_init)
    labels_top_pc1 = kmeans.fit_predict(features_top_pc1)

    segmented_top_pc1 = labels_top_pc1.reshape(img_clahe.shape)

    # -----------------------------
    # Step 5: Visualize results
    # -----------------------------
    plt.figure(figsize=(15, 6))
    plt.subplot(1, k+1, 1)
    plt.imshow(img_clahe, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    for i in range(k):
        mask_i = (segmented_top_pc1 == i).astype(np.uint8)
        plt.subplot(1, k+1, i+2)
        plt.imshow(mask_i, cmap='gray')
        plt.title(f'Cluster {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ----------------------------
# Ask user whether to save mask
# ----------------------------
user_input = input("Do you want to save the texture mask? (Y/N): ").strip().upper()

if user_input == 'Y':
    filename = input("Enter a filename (leave empty for 'mask'): ").strip()
    if filename == '':
        filename = 'mask'
    
    cluster_id = int(input(f"\nEnter the cluster number (0-{k-1}) corresponding to the wheel: ").strip())

    # Extract the chosen cluster mask
    mask_to_save = (segmented_top_pc1 == cluster_id).astype(np.uint8)

    scipy.io.savemat(f'{filename}.mat', {
        'image': img_clahe,
        'params': (ksize, sigmas, lambdas, thetas),
        'filters': filters,
        'clustering': (k, n_init),
        'mask': mask_to_save
    })

    print(f"Saved as {filename}.mat")

else:
    print("Result not saved.")