import joblib
import numpy as np
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os

# Load model (will be created after training)
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

@api_view(['POST'])
def predict_performance(request):
    """
    API endpoint to predict student performance index
    """
    try:
        if not os.path.exists(model_path):
            return Response(
                {'error': 'Model not found. Please train the model first.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        model = joblib.load(model_path)
        data = request.data
        
        # Create a DataFrame with the exact column names used during training
        # to fix the "feature names" warning and ensure correct alignment.
        input_df = pd.DataFrame([{
            'Hours Studied': data['hours_studied'],
            'Previous Scores': data['previous_scores'],
            'Extracurricular Activities': 1 if data['extracurricular'] else 0,
            'Sleep Hours': data['sleep_hours'],
            'Sample Question Papers Practiced': data['sample_papers']
        }])
        
        # Make prediction using the DataFrame
        prediction = model.predict(input_df)[0]
        
        return Response({
            'predicted_performance_index': round(float(prediction), 2)
        })
    
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
