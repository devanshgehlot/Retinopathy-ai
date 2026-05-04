import requests
import json

# Path to your retinal image
image_path = "demo_images/test_retina.jpg"  # must be a real .jpg image

print("Sending image to API...")

with open(image_path, "rb") as f:
    response = requests.post(
        "http://localhost:5000/predict",
        files={"image": ("test_retina.jpg", f, "image/jpeg")}
    )

if response.status_code == 200:
    result = response.json()
    print("\n===== PREDICTION RESULT =====")
    print(f"Grade       : {result['grade']} — {result['grade_label']}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"Is Serious  : {result['is_serious']}")
    print(f"Disclaimer  : {result['disclaimer']}")
    print(f"\nAll Probabilities:")
    for grade, prob in result['probabilities'].items():
        labels = {
            "0": "No DR",
            "1": "Mild DR", 
            "2": "Moderate DR",
            "3": "Severe DR",
            "4": "Proliferative DR"
        }
        print(f"  Grade {grade} ({labels[grade]}): {prob}%")
    print(f"\nHeatmap base64 length: {len(result['heatmap_base64'])} chars")
    print("==============================")
else:
    print(f"Error {response.status_code}: {response.json()}")