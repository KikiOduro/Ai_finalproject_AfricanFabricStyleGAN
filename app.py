import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
import time

class AdaIN(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.channels = channels
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.scale_transform = layers.Dense(self.channels)
        self.bias_transform = layers.Dense(self.channels)

    def call(self, x, w):
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
        normalized = (x - mean) / (std + self.epsilon)

        style_scale = self.scale_transform(w)
        style_bias = self.bias_transform(w)

        style_scale = tf.reshape(style_scale, (-1, 1, 1, self.channels))
        style_bias = tf.reshape(style_bias, (-1, 1, 1, self.channels))

        return style_scale * normalized + style_bias 

# Set up the Streamlit page
st.set_page_config(page_title="Batiq", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #FFFFFF;
            color: #333;
        }
        .stApp {
            background-color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background-color: #FFFFFF;
        }
        h1 {
            color: #000000;
            text-align: center;
            font-size: 3rem;
        }
        h2, h3, h4, h5, h6 {
            color: #000000;
            text-align: center;
        }
        .css-1cpxqw2.edgvbvh10 {
            color: #000000;
        }
        .st-bb {
            color: #000000;
        }
        .st-ag {
            color: #000000;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #FFFFFF;
            color: #333;
            text-align: center;
            padding: 10px;
        }
        .about-us {
            background-color: #FFFFFF;
            padding: 20px;
        }
        .about-us-text {
            font-size: 1.5rem;
            color: #000000;
        }
        .image-section {
            background-color: #FFFFFF;
            padding: 10px;
        }
        .generate-button {
            background-color: #FE654F;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            display: block;
            margin: 20px auto;
            text-align: center;
        }
        .small-image {
            width: 100px;
            height: auto;
        }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<h1>Batiq!</h1>", unsafe_allow_html=True)

# About Us Section
st.markdown("## About Us")
st.markdown("<div class='about-us'>", unsafe_allow_html=True)
about_us_cols = st.columns([1, 2], gap='medium')
with about_us_cols[0]:
    about_us_image = Image.open("/Users/akuaoduro/Desktop/africa_website/fasebook_post.png")
    st.image(about_us_image, use_column_width=True)
with about_us_cols[1]:
    st.markdown("""
        <div class="about-us-text">
            Welcome to Batiq. Explore the rich heritage of African Batik prints. 
            Our artisans create unique and authentic designs that bring the beauty of African culture to life. 
            Join us in celebrating tradition and innovation.
        </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Image Sections
st.markdown("<div class='image-section'>", unsafe_allow_html=True)
image_paths = [
    "/Users/akuaoduro/Desktop/africa_website/neneh-255.jpeg",
    "/Users/akuaoduro/Desktop/africa_website/esther-802.jpeg",
    "/Users/akuaoduro/Desktop/africa_website/afs-67204-0E1H1O38T907OUF69BWUWEECTPRZ1ZT-l.jpeg",
    "/Users/akuaoduro/Desktop/africa_website/afs-52585-009-grace-workshop-FOSC32A5S6YI7-o.jpeg"
]
image_cols = st.columns(4)
for col, img_path in zip(image_cols, image_paths):
    image = Image.open(img_path)
    col.image(image, use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Load the generator model
@st.cache_resource
def load_generator_model():
    model_path = '/Users/akuaoduro/Desktop/stylegan/StyleGAN_Models/generator_model.h5'
    with tf.keras.utils.custom_object_scope({'AdaIN': AdaIN}):
        model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Function to generate images
def generate_images(generator, num_images, latent_dim):
    # Use current time to seed randomness
    seed = int(time.time()) % 1000
    tf.random.set_seed(seed)
    
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images + 1) / 2  # Rescale to [0, 1]
    return generated_images

# Function to resize images
def resize_images(images, target_size):
    resized_images = []
    for img in images:
        img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.LANCZOS3)  # Higher quality resize
        img = tf.clip_by_value(img, 0, 1)  # Ensure values are within [0, 1]
        resized_images.append(img)
    return np.array(resized_images)

# Generate Batiq Button
st.markdown("<div class='generate-button'>", unsafe_allow_html=True)
if st.button("Generate Batiq"):
    generator = load_generator_model()
    generated_images = generate_images(generator, 1, 512)  # Generate 1 image
    resized_images = resize_images(generated_images, (256, 256))  # Resize images to 256x256 for better quality

    st.markdown("<h2>Generated Batiq</h2>", unsafe_allow_html=True)
    for i in range(resized_images.shape[0]):
        st.image(resized_images[i], width=100, caption="Generated Batiq")  # Display resized images at 100px width
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<footer class='footer'><p>Built with love from Lady and Kiki</p></footer>", unsafe_allow_html=True)
