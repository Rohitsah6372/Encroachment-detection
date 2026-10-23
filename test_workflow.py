
import os
from pathlib import Path
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Read the API key from the environment variable
api_key = os.environ.get("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError(
        "ROBOFLOW_API_KEY is not set. Set it in your PowerShell terminal."
    )

# Connect to Roboflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
).configure(
    InferenceConfiguration(api_key_transport="header")
)

# Folder containing your encroachment photos
image_folder = Path(
    r"C:\Users\rohit\Downloads\Encroached\Encroached\Our Captures"
)

# Get the first five JPG images
image_files = sorted(image_folder.glob("*.jpg"))[:5]

if not image_files:
    raise FileNotFoundError(
        f"No JPG images found in: {image_folder}"
    )

# Test each image
for image_path in image_files:
    print(f"\nTesting: {image_path.name}")

    result = client.run_workflow(
        workspace_name="rohits-workspace-2ajh9",
        workflow_id="footpath-encorachment-lzi0k",
        images={"image": str(image_path.resolve())},
        use_cache=True
    )

    # Extract predictions
    predictions = (
        result[0].get("predictions", {}).get("predictions", [])
        if isinstance(result, list) and result
        else []
    )

    print(f"Detections: {len(predictions)}")
    print(predictions)

print("\nFinished testing images.")
