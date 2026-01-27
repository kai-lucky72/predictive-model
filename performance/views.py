import joblib
import numpy as np
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
import time
from datetime import datetime
from huggingface_hub import hf_hub_download

# Configuration
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
HF_REPO_ID = os.environ.get("HF_REPO_ID", "snakail/student-performance-model")
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        # 1. Try loading local model
        if os.path.exists(model_path):
            print("Loading model from local disk...")
            try:
                MODEL = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading local model: {e}")
        
        # 2. Try loading from Hugging Face if local fails or is corrupted
        if MODEL is None:
            print(f"Model not found locally. Attempting to download from Hugging Face: {HF_REPO_ID}")
            try:
                downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.pkl")
                MODEL = joblib.load(downloaded_path)
                print("Successfully loaded model from Hugging Face.")
            except Exception as e:
                print(f"Failed to load model from Hugging Face: {e}")
    return MODEL

# Pre-load model at module import to avoid delay on first request
get_model()

@api_view(['POST'])
def predict_performance(request):
    """
    API endpoint to predict student performance index
    """
    start_time = time.time()
    try:
        model = get_model()
        if model is None:
            return Response(
                {'error': 'Model not found. Please train or upload the model.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        data = request.data
        
        # Create a DataFrame with the exact column names used during training
        input_df = pd.DataFrame([{
            'Hours Studied': data['hours_studied'],
            'Previous Scores': data['previous_scores'],
            'Extracurricular Activities': 1 if data['extracurricular'] else 0,
            'Sleep Hours': data['sleep_hours'],
            'Sample Question Papers Practiced': data['sample_papers']
        }])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        end_time = time.time()
        
        response = Response({
            'predicted_performance_index': round(float(prediction), 2),
            'source': 'local' if os.path.exists(model_path) else 'huggingface',
            'processing_time_ms': round((end_time - start_time) * 1000, 2),
            'timestamp': datetime.now().isoformat()
        })
        
        # Aggressive cache-busting headers
        response['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        response['Surrogate-Control'] = 'no-store'
        
        return response
    
    except KeyError as e:
        return Response(
            {'error': f'Missing required field: {str(e)}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return Response(
            {'error': f'Prediction failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
