import filetype
from common.utils.base64 import base64_encoder


def get_b64_image_url(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        b64_image = base64_encoder(binary_data=image_file.read())
        mime_type = filetype.guess(image_path).mime

        return f"data:{mime_type};base64,{b64_image}"
