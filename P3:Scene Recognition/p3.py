import os
import numpy as np
import cv2
import gc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))
    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows
    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def get_tiny_image(img, output_size):
    feature = None
    # To do    
    tiny = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
    tiny = tiny.astype(np.float32).flatten()
    
    tiny -= tiny.mean()
   
    norm = np.linalg.norm(tiny)
    if norm > 0:
        tiny /= norm

    feature = tiny
    return feature



def predict_knn(feature_train, label_train, feature_test, k):
    label_test_pred = None
    # To do    
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs.fit(feature_train)

    distances, indices = nbrs.kneighbors(feature_test)

    label_train = np.array(label_train)
    preds = []

    for inds in indices:
        neighbor_labels = label_train[inds]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        pred_label = values[np.argmax(counts)]
        preds.append(pred_label)

    label_test_pred = np.array(preds)
    return label_test_pred



def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    confusion, accuracy = None, None
    
    output_size = (16, 16)
        
    label_to_idx = {label: i for i, label in enumerate(label_classes)}
    
    label_train_indices = np.array([label_to_idx[l] for l in label_train_list])
    label_test_indices = np.array([label_to_idx[l] for l in label_test_list])
    
    
    train_features = []
    for img_path in img_train_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        feat = get_tiny_image(img, output_size)
        train_features.append(feat)
    train_features = np.vstack(train_features)
    
    
    test_features = []
    for img_path in img_test_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        feat = get_tiny_image(img, output_size)
        test_features.append(feat)
    test_features = np.vstack(test_features)
    
    
    k = 5
    pred_indices = predict_knn(train_features, label_train_indices, test_features, k)
       
    num_classes = len(label_classes)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    for true_idx, pred_idx in zip(label_test_indices, pred_indices):
        confusion_mat[true_idx, pred_idx] += 1.0
    
    
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    confusion_mat = confusion_mat / row_sums
       
    correct = np.sum(label_test_indices == pred_indices)
    accuracy = correct / len(label_test_indices)
    
    confusion = confusion_mat
    return confusion, accuracy


def compute_dsift(img, stride, size):
    dense_feature = None
    # To do      
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)

    h, w = img_gray.shape
    keypoints = []
  
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(size)))

    if len(keypoints) == 0:
        dense_feature = np.zeros((0, 128), dtype=np.float32)
        return dense_feature

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(img_gray, keypoints)

    if descriptors is None:
        dense_feature = np.zeros((0, 128), dtype=np.float32)
    else:
        dense_feature = descriptors.astype(np.float32)

    return dense_feature


def build_visual_dictionary(dense_feature_list, dict_size):
    vocab = None
    # To do
    all_features = [f for f in dense_feature_list if f is not None and f.size > 0]
    if len(all_features) == 0:
        vocab = np.zeros((dict_size, 128), dtype=np.float32)
        np.save("vocab.npy", vocab)
        return vocab

    all_features = np.vstack(all_features).astype(np.float32)
   
    max_samples = 100000
    if all_features.shape[0] > max_samples:
        idx = np.random.choice(all_features.shape[0], max_samples, replace=False)
        all_features = all_features[idx]

    kmeans = KMeans(
        n_clusters=dict_size,
        n_init=10,
        max_iter=300,
        random_state=0
    )
    kmeans.fit(all_features)

    vocab = kmeans.cluster_centers_.astype(np.float32)

    np.save("vocab.npy", vocab)
    return vocab

def compute_bow(feature, vocab):
    bow_feature = None
    # To do  
    dict_size = vocab.shape[0]

    if feature is None or feature.size == 0:
        hist = np.zeros((dict_size,), dtype=np.float32)
    else:       
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(vocab)
        _, indices = nn.kneighbors(feature)
        word_indices = indices.flatten()
        hist = np.bincount(word_indices, minlength=dict_size).astype(np.float32)
   
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm

    bow_feature = hist
    return bow_feature

def classify_knn_bow(label_classes, label_train_list, img_train_list,
                     label_test_list, img_test_list, vocab=None):
    confusion, accuracy = None, None
    # To do
    stride = 8
    patch_size = 16
    dict_size = 100
    k = 5

    if vocab is None:
        dense_feature_list = []
        for img_path in img_train_list:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            dense = compute_dsift(img, stride, patch_size)
            dense_feature_list.append(dense)
        vocab = build_visual_dictionary(dense_feature_list, dict_size)

    train_features = []
    for img_path in img_train_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        dense = compute_dsift(img, stride, patch_size)
        bow = compute_bow(dense, vocab)
        train_features.append(bow)
    train_features = np.vstack(train_features)

    test_features = []
    for img_path in img_test_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        dense = compute_dsift(img, stride, patch_size)
        bow = compute_bow(dense, vocab)
        test_features.append(bow)
    test_features = np.vstack(test_features)

    label_test_pred = predict_knn(train_features, label_train_list, test_features, k)

    num_classes = len(label_classes)
    label_to_idx = {label: i for i, label in enumerate(label_classes)}
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.float32)

    label_test_arr = np.array(label_test_list)
    for true_label, pred_label in zip(label_test_arr, label_test_pred):
        i = label_to_idx[true_label]
        j = label_to_idx[pred_label]
        confusion_mat[i, j] += 1.0

    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    confusion_mat = confusion_mat / row_sums

    accuracy = float(np.mean(label_test_arr == label_test_pred))
    
    confusion = confusion_mat
    return confusion, accuracy

def predict_svm(feature_train, label_train, feature_test):
    label_test_pred = None
    # To do    
    X_train = np.asarray(feature_train, dtype=np.float32)
    X_test = np.asarray(feature_test, dtype=np.float32)
    y_train = np.array(label_train)

    classes = np.unique(y_train)
    scores_list = []

    for c in classes:
        y_binary = np.where(y_train == c, 1, -1)
        clf = LinearSVC(C=1.0, max_iter=2000)
        clf.fit(X_train, y_binary)
        scores = clf.decision_function(X_test) 
        scores_list.append(scores.reshape(-1, 1))

    scores_matrix = np.hstack(scores_list)  
    best_indices = np.argmax(scores_matrix, axis=1)
    preds = classes[best_indices]

    label_test_pred = np.array(preds)
    return label_test_pred



def classify_svm_bow(label_classes, label_train_list, img_train_list,
                     label_test_list, img_test_list, vocab=None):
    confusion, accuracy = None, None
    # To do
    stride = 8
    patch_size = 16
    dict_size = 100

    if vocab is None:
        dense_feature_list = []
        for img_path in img_train_list:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            dense = compute_dsift(img, stride, patch_size)
            dense_feature_list.append(dense)
        vocab = build_visual_dictionary(dense_feature_list, dict_size)

    train_features = []
    for img_path in img_train_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        dense = compute_dsift(img, stride, patch_size)
        bow = compute_bow(dense, vocab)
        train_features.append(bow)
    train_features = np.vstack(train_features)

    test_features = []
    for img_path in img_test_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        dense = compute_dsift(img, stride, patch_size)
        bow = compute_bow(dense, vocab)
        test_features.append(bow)
    test_features = np.vstack(test_features)

    label_test_pred = predict_svm(train_features, label_train_list, test_features)

    num_classes = len(label_classes)
    label_to_idx = {label: i for i, label in enumerate(label_classes)}
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.float32)

    label_test_arr = np.array(label_test_list)
    for true_label, pred_label in zip(label_test_arr, label_test_pred):
        i = label_to_idx[true_label]
        j = label_to_idx[pred_label]
        confusion_mat[i, j] += 1.0

    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    confusion_mat = confusion_mat / row_sums

    accuracy = float(np.mean(label_test_arr == label_test_pred))

    confusion = confusion_mat
    return confusion, accuracy



def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    vocab = None
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    confusion, accuracy = classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    del confusion, accuracy
    gc.collect()

    # vocab = np.load("vocab.npy")
    confusion, accuracy = classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    del confusion, accuracy
    gc.collect()

    # vocab = np.load("vocab.npy")
    confusion, accuracy = classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
