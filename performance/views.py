import joblib
import numpy as np
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
    Expected JSON: {
        "hours_studied": int,
        "previous_scores": int,
        "extracurricular": bool,
        "sleep_hours": int,
        "sample_papers": int
    }
    """
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            return Response(
                {'error': 'Model not found. Please train the model first.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Load the model
        model = joblib.load(model_path)
        
        # Extract data from request
        data = request.data
        
        # Prepare features in the correct order
        features = np.array([[
            data['hours_studied'],
            data['previous_scores'],
            1 if data['extracurricular'] else 0,
            data['sleep_hours'],
            data['sample_papers'],
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
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
