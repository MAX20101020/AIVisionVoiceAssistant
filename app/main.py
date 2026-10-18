from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from dotenv import load_dotenv
import torch
from ema_lightning import EMA
from playsound import playsound

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
token = os.getenv("TOKEN")

if token:
    os.environ["HF_TOKEN"] = token

photo_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")
photo_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
tts = EMA()


def main():
    while True:
        try:
            menu = int(input(
                "===== AI Vision and Voice Assistant =====\n"
                "1. Вибрати фотографію\n"
                "2. Запустити аудіо файл\n"
                "0. Вихід\n"
            ))
        except ValueError:
            print("Введіть номер пункту.")
            continue

        match menu:
            case 0:
                break
            case 1:
                convert_photo_to_voice()
            case 2:
                play_audio()
        input("Press Enter to continue: ")


def input_photo():
    allowed_suffixes = [".jpg", ".png", ".jpeg"]
    path = Path(input("Enter the path of image: "))

    if not path.exists():
        print("Error: file is not found")
        return None
    if not path.suffix.lower() in allowed_suffixes:
        print(f"Error: suffix of photo not allowed! Allowed: {allowed_suffixes}")
        return None

    return path


def write_description_to_file(path, description):
    descriptions_path = Path(__file__).parent / "descriptions.txt"

    with open(descriptions_path, "a", encoding="utf-8") as file:
        file.write(f"Photo {path}:\n{description}\n\n")


def convert_photo_to_voice():
    path = input_photo()
    img = Image.open(path)
    if img is None:
        return None

    inputs = photo_processor(img, return_tensors="pt").to(device)
    out = photo_model.generate(**inputs, max_new_tokens=50)
    description = photo_processor.decode(out[0], skip_special_tokens=True)

    print(f"Image description:\n{description}")
    write_description_to_file(path, description)

    while True:
        name = input("Enter a name of audio file:")
        if name is not None:
            break

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / f"{name}.wav"
    speech = tts.say(description, path=str(audio_path))

    print(f"Duration: {speech.duration}")
    print(f"Audio saved to: {audio_path}")


def play_audio():
    output_dir = Path(__file__).parent.parent / "output"
    audio_files = list(output_dir.glob("*.wav"))

    if not audio_files:
        print("Files nor found")
        return

    print("===== Audio files =====")

    for i, file in enumerate(audio_files, start=1):
        print(f"{i}. {file.name}")

    try:
        choice = int(input("Оберіть аудіофайл за номером: "))
    except ValueError:
        print("Введіть число.")
        return

    if choice < 1 or choice > len(audio_files):
        print("Error: incorrect number")
        return

    selected_file = audio_files[choice - 1]
    print(f"Playing: {selected_file.name}")
    playsound(str(selected_file))


if __name__ == "__main__":
    main()