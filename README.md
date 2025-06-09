

# DCGAN for Cat and Dog Image Generation

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using PyTorch to generate images of cats and dogs. The architecture is based on best practices from the original DCGAN paper and is designed to train on a folder of labeled images.

---

## 📌 Features

* PyTorch-based DCGAN implementation
* Generator and Discriminator architectures defined clearly
* Customizable training parameters
* Progress tracking with saved generated images
* Visualization of training loss and generated results
* Support for GPU acceleration

---

## 🗂️ Dataset Structure

This project uses datasets structured like this:

```
/path/to/dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

It uses `torchvision.datasets.ImageFolder` for loading the dataset. Make sure each category (cats, dogs) is placed in a separate folder under the training directory.

---

## 🔧 Requirements

* Python 3.7+
* PyTorch
* torchvision
* matplotlib
* numpy

You can install the required packages using:

```bash
pip install torch torchvision matplotlib numpy
```

---

## ⚙️ Configuration

You can modify the training parameters at the top of the script:

```python
DATA_ROOT = 'path/to/dataset'
IMAGE_SIZE = 64
Z_DIM = 100
BATCH_SIZE = 128
NUM_EPOCHS = 5
LR_G = 0.0002
LR_D = 0.0002
```

The output images and plots are saved to the `./generated_images` directory.

---

## 🚀 Running the Code

To train the model:

```bash
python dcgan_dog_cat.py
```

> Make sure to adjust the `DATA_ROOT` variable to point to your dataset path.

The training will:

* Display the generator and discriminator architectures
* Save generated images during training (every 500 iterations)
* Save loss plots and final generated images

---

## 📈 Output Samples

* **Loss Curve**: `loss_plot.png`
* **Final Generated Images**: `final_generated_images.png`
* **Intermediate Generated Images**: `generated_image_epoch_XXXX_iter_XXXXXX.png`

These are all saved in the `./generated_images` folder.

---

## 🧠 Model Architecture

### Generator

* Based on `nn.ConvTranspose2d` layers
* Uses `BatchNorm2d` and `ReLU` activations
* Outputs 64x64 RGB images

### Discriminator

* Based on `nn.Conv2d` layers
* Uses `BatchNorm2d` and `LeakyReLU`
* Outputs a scalar probability

---

## 📎 Notes

* Make sure your dataset is large and diverse enough for the GAN to learn useful representations.
* Increase `NUM_EPOCHS` for better results.
* Use GPU for faster training (`cuda:0` is automatically detected if available).



## 📄 License

This project is provided under the [MIT License](https://opensource.org/licenses/MIT).

