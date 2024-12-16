import pydicom
from PIL import Image
import numpy as np


def dicom_to_jpg(dicom_file, jpg_file):
    ds = pydicom.dcmread(dicom_file)

    pixel_array = ds.pixel_array
    if ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.amax(pixel_array) - pixel_array
    pixel_array = pixel_array - np.min(pixel_array)
    pixel_array = pixel_array / np.max(pixel_array)
    pixel_array = (pixel_array * 255).astype(np.uint8)

    image = Image.fromarray(pixel_array)
    image.save(jpg_file)


dicom_to_jpg("00053190460d56c53cc3e57321387478.dicom", "output_file.jpg")
