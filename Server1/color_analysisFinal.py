#!/usr/bin/env python
# coding: utf-8

# In[10]:

import cv2
import dlib
import numpy as np
import google.generativeai as genai
import pandas as pd
import os
import io
from PIL import  Image
import re
import imageio


genai.configure(api_key="AIzaSyBNEFplAk_-EsZd25G387WRky7d4tzGXPI")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Load the face detector and shape predictor models



def get_average_color(image, region):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [region], 255)
    mean_color = cv2.mean(image, mask=mask)
    return mean_color[:3]  # Return BGR


def extract_facial_colors(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected.")
        return

    face = faces[0]
    landmarks = predictor(gray, face)

    # Define regions based on landmarks
    forehead = np.array([
        (landmarks.part(19).x, landmarks.part(19).y),
        (landmarks.part(24).x, landmarks.part(24).y),
        (landmarks.part(24).x, landmarks.part(24).y - 30),
        (landmarks.part(19).x, landmarks.part(19).y - 30)
    ])

    nose = np.array([
        (landmarks.part(27).x, landmarks.part(27).y),
        (landmarks.part(33).x, landmarks.part(33).y),
        (landmarks.part(31).x, landmarks.part(31).y),
        (landmarks.part(35).x, landmarks.part(35).y)
    ])

    left_cheek = np.array([
        (landmarks.part(2).x, landmarks.part(2).y),
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(28).x, landmarks.part(28).y),
        (landmarks.part(3).x, landmarks.part(3).y)
    ])

    right_cheek = np.array([
        (landmarks.part(14).x, landmarks.part(14).y),
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(28).x, landmarks.part(28).y),
        (landmarks.part(13).x, landmarks.part(13).y)
    ])

    lips = np.array([
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y),
        (landmarks.part(64).x, landmarks.part(64).y),
        (landmarks.part(60).x, landmarks.part(60).y)
    ])

    # Get the average color of each region
    forehead_color = get_average_color(image, forehead)
    nose_color = get_average_color(image, nose)
    left_cheek_color = get_average_color(image, left_cheek)
    right_cheek_color = get_average_color(image, right_cheek)
    lips_color = get_average_color(image, lips)

    # Convert BGR to RGB for analysis
    forehead_color = forehead_color[::-1]
    nose_color = nose_color[::-1]
    left_cheek_color = left_cheek_color[::-1]
    right_cheek_color = right_cheek_color[::-1]
    lips_color = lips_color[::-1]

    return forehead_color, nose_color, left_cheek_color, right_cheek_color, lips_color


def remove_empty_lines(text):
    """removes empty lines in a text"""
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)


def analyze_skin_tone_and_suggest_lipsticks(forehead_color, nose_color, left_cheek_color, right_cheek_color, lips_color):
    # Create the prompt with the color codes
    prompt = f"""
    Analyze the following skin tone color codes and provide a color theory analysis. Determine whether the skin type is autumn, spring, winter, or summer. Additionally, suggest suitable lipstick shades for the skin tone.

    Forehead Color (RGB): {forehead_color}
    Nose Color (RGB): {nose_color}
    Left Cheek Color (RGB): {left_cheek_color}
    Right Cheek Color (RGB): {right_cheek_color}
    Lips Color (RGB): {lips_color}

    am I a summer, spring, winter or fall person
    reply according to this format:
    season:
    name
    matching colors: 
    name, hex code, name, hex code
    best lip color: 
    name, hex code, name, hex code
    don't add any stars 
    put the all matching colors and all the hair colors in one line
    don't forget to separate the name of the color and the code with a , 
    don't put an extra , at the end of any line
    """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    response=response.candidates[0].content.parts[0].text
    response_text = remove_empty_lines(response)
#     print(response_text)

    matches = re.findall(r'season:\s*(.*)\nname:\s*(.*)\nmatching colors:\s*(.*)\nbest lip color:\s*(.*)', response_text, re.IGNORECASE)

    # Process each match
    data = []
    for match in matches:
        season, name, colors_text, lip_color = match
        # Split colors by comma and strip whitespace
        colors = [color.strip() for color in colors_text.split(',')]
        # Split each color into name and hex code
        colors = [re.split(r'\s+#', color) for color in colors]
        colors1 = [color.strip() for color in lip_color.split(',')]
        # Split each color into name and hex code
        colors1 = [re.split(r'\s+#', color) for color in colors1]
        # Create a dictionary for each match
        item = {
            'season': season,
            'name': name,
            'matching colors': colors,
            'best lip color':colors1
        }
        data.append(item)
    
#     print(data)
    return data[0]
#     return response.candidates[0].content.parts[0].text



def process_image (image_path):

    # image_path = Image.open(io.BytesIO(file.read()))
    # image2 = Image.open(image1.stream)
    # image1=Image.open(image_path)
    colors = extract_facial_colors(image_path)
    if colors:
        forehead_color, nose_color, left_cheek_color, right_cheek_color, lips_color = colors
        analysis = analyze_skin_tone_and_suggest_lipsticks(forehead_color, nose_color, left_cheek_color, right_cheek_color, lips_color)
        # print(analysis)
        return analysis


# In[11]:





# In[ ]:




