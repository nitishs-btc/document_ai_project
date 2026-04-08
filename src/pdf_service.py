import os
from pdf2image import convert_from_path

class PDFService:

    def __init__(self, image_folder):
        self.image_folder = image_folder
        os.makedirs(self.image_folder, exist_ok=True)

    def convert_pdfs(self, pdf_paths):
        all_images = {}

        for pdf_path in pdf_paths:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            images = convert_from_path(pdf_path, dpi=300)
            image_paths = []

            for i, img in enumerate(images, start=1):
                filename = f"{pdf_name}_{i}.png"
                path = os.path.join(self.image_folder, filename)

                img.save(path, "PNG")
                image_paths.append(path)

            all_images[pdf_name] = image_paths

        return all_images