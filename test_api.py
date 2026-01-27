import requests
import json

def test_prediction():
    url = "http://127.0.0.1:8000/api/predict/"
    payload = {
        "hours_studied": 1,
        "previous_scores": 78,
        "extracurricular": True,
        "sleep_hours": 2,
        "sample_papers": 7
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to connect: {str(e)}")

if __name__ == "__main__":
    test_prediction()
