import requests
import argparse
import base64


def main(args):
    with open(args.path, "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = requests.post("http://localhost:3000/classify", json={"img": image_b64})
    print(response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    main(args)
