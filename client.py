from time import sleep

import requests
from loguru import logger


def predict():
    logger.info("Sending POST requests!")
    files = {
        "file": open("images/ga.png", "rb"),
    }
    response = requests.post(
        "http://api.example.com/predict",
        headers={
            "accept": "application/json",
        },
        files=files,
    )

    # logger.info(f"Response: {response.json()}")


if __name__ == "__main__":
    while True:
        predict()
        sleep(20)
