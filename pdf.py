from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Make sure it's set in the .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename.split('.')[0]
        save_folder = os.path.join('uploads', filename)
        os.makedirs(save_folder, exist_ok=True)

        file.save(os.path.join(save_folder, file.filename))

        # Extract images from PDF
        pdf_path = os.path.join(save_folder, file.filename)
        extract_images_from_pdf(pdf_path, save_folder)

        # Process images and get responses
        responses = process_images(save_folder)

        return jsonify(responses)

def extract_images_from_pdf(pdf_path, save_folder):
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            base_image = pdf_document.extract_image(img[0])
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page_num}_image{image_index}.{image_ext}"
            image_path = os.path.join(save_folder, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

def image_format(image_path):
    img = Path(image_path)

    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "image/png",
            "data": img.read_bytes()
        }
    ]
    return image_parts

def gemini_output(image_path, system_prompt, user_prompt):
    image_info = image_format(image_path)
    input_prompt = [system_prompt, image_info[0], user_prompt]
    response = model.generate_content(input_prompt)
    return response.text

def process_images(folder_path):
    responses = []
    system_prompt = """
                   You are a specialist in understanding dutch language and extract all text from student writing into text.
                   Input images in the form of paper hand written assignments or test will be provided to you,
                   and your task is to respond to questions based on the content of the input image.
                   """
    user_prompt = "Convert hand written data into text based format with appropriate words for the data in image "
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file_name)
                output = gemini_output(image_path, system_prompt, user_prompt)
                responses.append({"image_path": image_path, "response": output})
    return responses

if __name__ == '__main__':
    app.run(debug=True)
