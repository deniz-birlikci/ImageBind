# ImageBind for JSON Serializable Inputs

This repository introduces an enhancement to the ImageBind model by allowing it to operate on JSON Serializable inputs. This feature streamlines the process for hosting models online, making it considerably more efficient to deploy API endpoints. As digital interactions grow more sophisticated, the ability to convert our modalities into a json-serializable format and communicate with an online API becomes crucial.

## JSON Serializable Conversion Functions

To make this seamless, we've introduced two functions that convert image and audio paths to their corresponding binary formats:

```python
import base64
from PIL import Image
from pydub import AudioSegment
import io

def image_paths_to_base64_strings(image_paths: list[str]) -> list[str]:
    """
    Convert a list of image paths to a list of Base64 encoded strings.

    Parameters:
    - image_paths (list[str]): List of paths to the images.

    Returns:
    list[str]: List containing Base64 encoded strings of images.
    """
    image_base64_strings = []
    for path in image_paths:
        with Image.open(path) as img:
            binary_io = io.BytesIO()
            img.save(binary_io, format="PNG")  # You can change the format if needed
            base64_encoded = base64.b64encode(binary_io.getvalue()).decode('utf-8')
            image_base64_strings.append(base64_encoded)
    return image_base64_strings

def audio_paths_to_base64_strings(audio_paths: list[str]) -> list[str]:
    """
    Convert a list of audio paths to a list of Base64 encoded strings.

    Parameters:
    - audio_paths (list[str]): List of paths to the audio files.

    Returns:
    list[str]: List containing Base64 encoded strings of audio files.
    """

    audio_base64_strings = []
    for path in audio_paths:
        audio = AudioSegment.from_file(path)
        binary_io = io.BytesIO()
        audio.export(binary_io, format="wav")  # You can change the format if needed
        base64_encoded = base64.b64encode(binary_io.getvalue()).decode('utf-8')
        audio_base64_strings.append(base64_encoded)
    return audio_base64_strings
```

## How to run the model
```python
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = "cuda:0"
model = imagebind_model.imagebind_huge_serializable(device=device, pretrained=True, 
                                                    bpe_path="/content/bpe/bpe_simple_vocab_16e6.txt.gz")

text_list = ["coastal outfits for the summer"]
image_paths = ["testing_data/coastal-dress.jpeg"]
audio_paths = ["testing_data/ocean-waves-112906.mp3"]

json_serializable_input = {
    "text" : text_list,
    "vision" : image_paths_to_base64_strings(image_paths),
    "audio" : audio_paths_to_base64_strings(audio_paths)
}

model(json_serializable_input)
```

## Deploying this as an endpoint using Baseten
Check out the repo, here.


# Original Model: ImageBind: One Embedding Space To Bind Them All

**[FAIR, Meta AI](https://ai.facebook.com/research/)** 

Rohit Girdhar*,
Alaaeldin El-Nouby*,
Zhuang Liu,
Mannat Singh,
Kalyan Vasudev Alwala,
Armand Joulin,
Ishan Misra*

```
@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang
and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2023}
}
```
