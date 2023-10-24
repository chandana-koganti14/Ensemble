import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from Unet import ResUNet, AttentionUNet
import numpy as np
import cv2
import csv
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from ensem import Ensemble
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from Unet import ResUNet, AttentionUNet,VanillaUNet

def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img
def predict_masks(input_image, model):
    input_image = preprocess_image(input_image)
    with torch.no_grad():
        ensemble_output = model(input_image.unsqueeze(0))
        ensemble_output = torch.sigmoid(ensemble_output)  
    return ensemble_output
def calculate_metrics(ground_truth_mask_array, predicted_mask_bin):
    iou = jaccard_score(ground_truth_mask_array.flatten(), predicted_mask_bin.flatten(), average='weighted')
    dice_coefficient = f1_score(ground_truth_mask_array.flatten(), predicted_mask_bin.flatten(), average='weighted')
    precision = precision_score(ground_truth_mask_array.flatten(), predicted_mask_bin.flatten(), average='weighted', zero_division=1.0)
    recall = recall_score(ground_truth_mask_array.flatten(), predicted_mask_bin.flatten(), average='weighted', zero_division=1.0)

    return iou, dice_coefficient, precision, recall
def calculate_accuracy(ground_truth_mask_array, predicted_mask_bin):
    correct_pixels = np.sum(ground_truth_mask_array == predicted_mask_bin)
    total_pixels = ground_truth_mask_array.size
    accuracy = correct_pixels / total_pixels
    return accuracy
def save_probability_map_as_image(probability_map, output_path):
    probability_map = (probability_map * 255).astype(np.uint8)
    cv2.imwrite(output_path, probability_map)

def load_ground_truth_masks_for_class(class_index, test_image_filename):
    test_image_filename_no_space = test_image_filename.replace(" ", "")
    ground_truth_filename = os.path.splitext(test_image_filename_no_space)[0] + ".tif"
    ground_truth_path = os.path.join(r"C:\Users\ADMIN\Desktop\tumor\test_a", ground_truth_filename)

    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth mask '{ground_truth_path}' not found for class {class_index}.")

    ground_truth_mask = Image.open(ground_truth_path)
    ground_truth_mask_array = np.array(ground_truth_mask) > 0
    return ground_truth_mask_array
def load_ground_truth_mask(test_image_filename):
    ground_truth_filename = test_image_filename.replace("image_", "mask_")
    ground_truth_path = os.path.join(r"C:\Users\ADMIN\Desktop\tumor\test_a", ground_truth_filename)
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth mask '{ground_truth_path}' not found for '{test_image_filename}'.")
    ground_truth_mask = Image.open(ground_truth_path)
    return ground_truth_mask
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict segmentation masks using an ensemble model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the ensemble model (e.g., 'ensemble.pth').")
    parser.add_argument("--model-type", type=str, required=True, choices=["resunet", "attentionunet","vanilla", "ensemble"], help="Type of model to evaluate.")
    parser.add_argument("--test-folder", type=str, required=True, help="Path to the folder containing test images.")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the folder where you want to save the test images.")
    args = parser.parse_args()
    resunet_model = ResUNet(in_channels=3, out_channels=1)
    attentionunet_model = AttentionUNet(in_channels=3, out_channels=1)
    vanilla_model=VanillaUNet(in_channels=3, out_channels=1)
    if args.model_type == "ensemble":
        resunet_model = ResUNet(in_channels=3, out_channels=1)
        attentionunet_model = AttentionUNet(in_channels=3, out_channels=1)
        vanilla_model = VanillaUNet(in_channels=3, out_channels=1)
        ensemble_model = Ensemble()
        ensemble_model.load_state_dict(torch.load(args.model))
        ensemble_model.eval()
        model = ensemble_model  
    else:
        if args.model_type == "resunet":
            model = ResUNet(in_channels=3, out_channels=1)
            model.load_state_dict(torch.load(args.model))  
        elif args.model_type == "attentionunet":
            model = AttentionUNet(in_channels=3, out_channels=1)
            model.load_state_dict(torch.load(args.model))  
        elif args.model_type == "vanilla":
            model = VanillaUNet(in_channels=3, out_channels=1)
            model.load_state_dict(torch.load(args.model))
        model.eval()
    test_image_folder = args.test_folder
    test_image_filenames = os.listdir(test_image_folder)
    iou_scores = []
    dice_coefficient_scores = []
    precision_scores = []
    recall_scores = []
    acc=[]
    for test_image_filename in test_image_filenames:
        test_image_path = os.path.join(test_image_folder, test_image_filename)
        ground_truth_mask_filename = test_image_filename.replace("image_", "mask_")
        ground_truth_mask_path = os.path.join(r"C:\Users\ADMIN\Desktop\tumor\test_a", ground_truth_mask_filename)
        if not os.path.exists(ground_truth_mask_path):
            raise FileNotFoundError(f"Ground truth mask '{ground_truth_mask_path}' not found.")
        ground_truth_mask = Image.open(ground_truth_mask_path)
        threshold=0.5
        ground_truth_mask_array = np.array(ground_truth_mask) > 0
        predicted_mask = predict_masks(test_image_path, model)
        class_index = 0 
        predicted_mask = predicted_mask[0, class_index, :, :].cpu().numpy()
        predicted_mask = cv2.resize(predicted_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        predicted_mask_bin = (predicted_mask > threshold).astype(int)
        iou, dice_coefficient, precision, recall = calculate_metrics(ground_truth_mask_array, predicted_mask_bin)
        accuracy = calculate_accuracy(ground_truth_mask_array, predicted_mask_bin)
        iou_scores.append(iou)
        dice_coefficient_scores.append(dice_coefficient)
        precision_scores.append(precision)
        recall_scores.append(recall)
        acc.append(accuracy)
        overlay_image = ImageOps.grayscale(ground_truth_mask)
        overlay_image = Image.blend(overlay_image, ImageOps.grayscale(ground_truth_mask), alpha=0.7)
        output_image_path = os.path.join(args.output_folder, f"{os.path.splitext(test_image_filename)[0]}_predicted_mask.png")
        overlay_image = overlay_image.convert("L")  
        ground_truth_mask = ground_truth_mask.convert("L")  
        ground_truth_mask = ground_truth_mask.resize(overlay_image.size)
        overlay_image = Image.blend(overlay_image, ground_truth_mask, alpha=0.7)
        overlay_image.save(output_image_path)
        output_probabilities_path = os.path.join(args.output_folder, f"{os.path.splitext(test_image_filename)[0]}_predicted_probabilities.csv")
        predicted_mask_flatten = predicted_mask.flatten()
        with open(output_probabilities_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Probability"])
            writer.writerows([[prob] for prob in predicted_mask_flatten])
        output_mask_npy_path = os.path.join(args.output_folder, f"{os.path.splitext(test_image_filename)[0]}_predicted_mask.npy")
        np.save(output_mask_npy_path, predicted_mask)
    avg_iou = np.mean(iou_scores)
    avg_dice_coefficient = np.mean(dice_coefficient_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_acc=np.mean(acc)
    print(f"Average IoU (Jaccard Score): {avg_iou:.4f}")
    print(f"Average Dice Coefficient: {avg_dice_coefficient:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")
