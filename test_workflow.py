import os

from pathlib import Path

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


# Load environment variables from the .env file beside this script
project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")


api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError(
        "ROBOFLOW_API_KEY is missing. Add it to your .env file."
    )




# # Read the API key from the environment variable
# api_key = os.environ.get("ROBOFLOW_API_KEY")

# if not api_key:
#     raise ValueError(
#         "ROBOFLOW_API_KEY is not set. Set it in your PowerShell terminal."
#     )

# Connect to Roboflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
).configure(
    InferenceConfiguration(api_key_transport="header")
)

# Folder containing your encroachment photos
image_folder = Path(
    r"C:\Users\rohit\Downloads\Encroached\Encroached\New folder"
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





from pathlib import Path
import os
from inference_sdk import InferenceHTTPClient

api_key = os.environ["ROBOFLOW_API_KEY"]

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

image_folder = Path(
    r"C:\Users\rohit\Downloads\Encroached\Encroached\New folder"
)
