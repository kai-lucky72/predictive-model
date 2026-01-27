from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

def upload_model():
    # Use the token provided by the user in README.md (now suggested as env var)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in environment variables.")
        print("Please set HF_TOKEN in your .env file or environment.")
        return

    api = HfApi()
    
    # We'll use a generic repo name, user can change if needed
    repo_id = "snakail/student-performance-model"
    
    try:
        print(f"Creating/Accessing repository: {repo_id}...")
        api.create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        
        # Path to the trained model
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'performance', 'model.pkl')
        
        if os.path.exists(model_path):
            print(f"Uploading {model_path} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="model.pkl",
                repo_id=repo_id,
                token=token
            )
            print("Successfully uploaded to Hugging Face! 🚀")
            print(f"Model available at: https://huggingface.co/{repo_id}")
        else:
            print(f"Error: {model_path} not found. Please train the model first.")
            
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    upload_model()
