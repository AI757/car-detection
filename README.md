# Car Classification Using ResNet on Jetson Inference

This project focuses on training a deep learning model to classify cars using NVIDIA Jetson Inference and ResNet18. The project involved multiple stages of data collection, model training, and testing.

Initially, the goal was to classify **car brands**, but due to high variation and low accuracy, the focus was shifted to classifying **car body types** (e.g., coupe, convertible, SUV), which yielded better and more consistent results.

## Dataset

This project utilized **three different datasets** to evaluate and improve model performance across different classification goals:

1. **Cars Brands in Egypt**  
   - [Source](https://www.kaggle.com/datasets/mohamedaziz15/cars-brands-in-egypt)  
   - Used for initial training and testing of **car brand classification**.

2. **American Car Brands Dataset**  
   - A second dataset containing **American car brands** was used to diversify the model’s exposure to different brand styles and improve generalization.

3. **Car Body Types Dataset**  
   - A dataset was used with labels such as **SUV, coupe, convertible, sedan**, etc.  
   - This dataset focused on **body type classification**, which proved to be more accurate and consistent than brand recognition with limited data.

All data was combined and organized under the directory: `cars`.

## Environment Setup

All training and inference was performed inside the NVIDIA Jetson Inference Docker container.

### 1. Start Docker Container

```bash
cd ~/jetson-inference/
./docker/run.sh

### 2. Navigate to the Training Directory

```bash
cd python/training/classification

### 3. Train the Model

```bash
python3 train.py --model-dir=models/cars data/cars

### 4. Convert Model to ONNX Format

```bash
python3 onnx_export.py --model-dir=models/cars

### 5. Set Environment Variables

```bash
NET=models/cars
DATASET=data/cars

### 6.Test the Model 

```bash
imagenet.py \
  --model=$NET/resnet18.onnx \
  --labels=$DATASET/labels.txt \
  --input_blob=input_0 \
  --output_blob=output_0 \
  $DATASET/test/Convertible/convertible1.jpg output.jpg

## Results
First test on a Porsche 911 GT3 RS:

Prediction: 2.5% Ram Truck

Second test:

Prediction: 56% Convertible

Accuracy was inconsistent when classifying by brand due to high visual variability between models of the same brand.

Model was retrained using body type labels from the custom dataset.

Accuracy improved and results were significantly more consistent.

Simplifying the classification target improved generalization.

## Lessons Learned 

Using multiple datasets (Egyptian brands, American brands, and car body types) enabled a deeper evaluation of different classification strategies.

Brand classification proved challenging due to high intra-class variation.

Body type classification was more effective, especially with limited data.

Curated, focused datasets lead to better model performance.


