<div align="center" id="gridexp">

<img src="example-images/logo.png" width="125px">

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/GridExplainer-7C3AED?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAtCAMAAAANxBKoAAAAaVBMVEX59/VimMCSlZiOkZT//PkzfLL9+vdZk707gbXP0dL49vTD1uNdlb5mnMNuocaxyt2JjJDR1tmDrcx1pcieoKKoqquZm51Sj7wqd6+70eD39fOur7DW3N/l6euQttFJirqgoqT39vS1trf7RgueAAABRElEQVRIx+2U3ZKDIAyFRYqogH9Yka3Wte//kEtQx7Uijhc7e9NvnBwvMngSE4Lgg5c0Tdd3f2p0A4LI8iZ7KiSEQOxZAZ2Nz+4OvPbfGVVLCGnzTFFKueYmqiwpMMZF87U/m5I4jkkuOUJIMGEilwkOwxB/X8qux0tnO7IX35RzijQCmX3XDic6B3qmmXkWSYDS0UExKKUGXWVSZpJJkKps6rpu7n4nHDHE/U6uVenpoONsRWIC2TsnzmwxALoSmYFBEKZKwFUl00DfMaAvgW6WwO8blSEGw8e+N1X+YbZxEq7ZrondbEOCC1iD422Qm0172BWbN230zKBtpB2+5HgGhxZYnRiMEytnc1I2v6tMr/yds+y3s8fzDhZTBwt3Bys7dqaDT0P3AF6zOJzYa/AWTWG6FKNZHB0cLUE6i2WRD//DD2PKKGImtJtKAAAAAElFTkSuQmCC" />
</div>



</div>




# AI Explanation Tool, GridExplainer: Visual Model Explanation via Grid-Based Occlusion

GridExplainer is a simple yet powerful interpretability tool for deep learning image models. It provides intuitive, visual explanations by measuring how hiding different image segments affects model confidence.

---

## Purpose
While advanced tools like SHAP, LIME, and Grad-CAM provide rich, gradient or perturbation-based explanations, they often rely on:

- Complex mathematical abstractions

- Internal model access (e.g., gradients, layers)

- Outputs that are hard to explain to non-technical users


GridExplainer was developed with this key idea:
<pre>
❓ “Why not visually show which regions matter most — by simply hiding them and observing the effect?”
</pre>

This approach results in transparent, grid-based visualizations that mimic human reasoning and are easy to interpret.

---

## Why GridExplainer Makes Sense

1. Direct Visual Feedback
Instead of analyzing internal activations, GridExplainer asks:

- “If I hide this part of the image, does the model get less confident?”

This produces concrete, verifiable evidence — no guesswork.

2. Alignment with Human Intuition
Humans tend to say:

- “It’s a dog because of the ears and the snout.”

GridExplainer systematically occludes parts of the image, simulating how people visually diagnose features.

3. Transparency over Complexity
While it may not offer theoretical optimality, it offers trust, clarity, and usability, which are essential in:

- Medical AI

- Legal tech

- Business intelligence

- Public-facing models


---

## How It Works

1. Grid segmentation: The input image is split into an N×N grid.

2. Occlusion-based perturbation: Each cell is masked (by mean value) one at a time.

3. Importance scoring: We record how much the prediction confidence drops.

4. Visualization: A heatmap grid is overlaid showing the impact of each cell.


---

## Technologies Used

- TensorFlow / Keras
- ResNet50 (pre-trained ImageNet model)
- Matplotlib / OpenCV / NumPy
- SHAP dataset (for test images)
- Google Colab-compatible

---

## Installation

```bash
 pip install shap opencv-python tensorflow matplotlib

```

---

## Quick Start

```bash
from tensorflow.keras.applications.resnet50 import ResNet50
from grid_explainer import GridExplainer

# Load model and image
model = ResNet50(weights="imagenet")
X, y = shap.datasets.imagenet50()
img = X[49]

# Explain
explainer = GridExplainer(model, grid_size=8)
explainer.explain(img)
explainer.visualize()

```

You can use any image sample between 0-50, and you can also try different samples that are not in the dataset.

---


## Notes

- Works on any image classification model (even black-box ones).

- The cell masking strategy is currently "mean replacement" but can be extended.

- Designed for interpretability and education, not production-scale attribution.


---

## Sample Outputs

Here are examples of the GridExplainer in action. Each grid cell shows the relative importance of that segment for the model’s prediction:

- You can interpret the numbers on each cell as "how important this region was" — the higher the number, the bigger the drop in confidence when occluded.

<table>
    <tr>
    <td>Original Image</td>
    <td>Grid Explanation</td>
  </tr>
  
  <tr>
    <td><img src="example-images/i_1.png" width="500"></td>
    <td><img src="example-outputs/o_1.png" width="500"></td>
  </tr>

 <tr>
    <td><img src="example-images/i_2.png" width="500"></td>
    <td><img src="example-outputs/o_2.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_4.png" width="500"></td>
    <td><img src="example-outputs/o_4.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_8.png" width="500"></td>
    <td><img src="example-outputs/o_8.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_23.png" width="500"></td>
    <td><img src="example-outputs/o_23.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_26.png" width="500"></td>
    <td><img src="example-outputs/o_26.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_28.png" width="500"></td>
    <td><img src="example-outputs/o_28.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_1.png" width="500"></td>
    <td><img src="example-outputs/o_1.png" width="500"></td>
  </tr>
  
   <tr>
    <td><img src="example-images/i_31.png" width="500"></td>
    <td><img src="example-outputs/o_31.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_35.png" width="500"></td>
    <td><img src="example-outputs/o_35.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_39.png" width="500"></td>
    <td><img src="example-outputs/o_39.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_41.png" width="500"></td>
    <td><img src="example-outputs/o_41.png" width="500"></td>
  </tr>

   <tr>
    <td><img src="example-images/i_49.png" width="500"></td>
    <td><img src="example-outputs/o_49.png" width="500"></td>
  </tr>

</table>

















