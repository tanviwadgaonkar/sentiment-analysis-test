import json
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from groq import Groq
import time

app = Flask(__name__)

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=api_key)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check for file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Load the file into a DataFrame
    if file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        return jsonify({'error': 'Invalid file format. Please upload XLSX or CSV.'}), 400

    # Check for the 'Review' column
    if 'Review' not in df.columns:
        return jsonify({'error': 'No Review column found in the file'}), 400

    # Prepare for sentiment analysis
    reviews = df['Review'].tolist()
    total_reviews = len(reviews)
    positive_score = 0
    negative_score = 0
    neutral_score = 0

    for review in reviews:
        retries = 5  # Number of retries for API calls
        backoff = 1  # Initial backoff time in seconds

        for attempt in range(retries):
            try:
                # Call the Groq API for sentiment analysis
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": f"Analyze the sentiment of the following review and return the scores as a JSON object: {review}"}],
                    model="llama3-groq-8b-8192-tool-use-preview"
                )
                
                # Log the full response for debugging
                sentiment_data = chat_completion.choices[0].message.content
                print("Full response from Groq API:", sentiment_data)  # Log the full response

                # Attempt to extract JSON
                json_part = extract_json(sentiment_data)

                if json_part is None:
                    return jsonify({'error': 'Invalid JSON format received from the API', 'raw_response': sentiment_data}), 500
                
                # Load JSON and update scores
                sentiment_scores = json.loads(json_part)
                scores = sentiment_scores.get('scores', {})
                positive_score += scores.get('positive', 0)
                negative_score += scores.get('negative', 0)
                neutral_score += scores.get('neutral', 0)
                break  # Exit the retry loop after a successful request

            except Exception as e:
                # Check if the error is due to rate limiting
                if 'Rate limit reached' in str(e):  
                    if attempt < retries - 1:  # If not the last attempt
                        wait_time = backoff * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit reached, sleeping for {wait_time} seconds before retrying...")
                        time.sleep(wait_time)  # Sleep before retrying
                    else:
                        print(f"Error processing review: {review}. Error: {str(e)}")
                        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
                else:
                    print(f"Error processing review: {review}. Error: {str(e)}")
                    return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

    return jsonify({
        'total_reviews': total_reviews,
        'positive': positive_score,
        'negative': negative_score,
        'neutral': neutral_score
    }), 200

def extract_json(sentiment_data):
    """Extracts JSON from the Groq API response."""
    try:
        # If the response is wrapped in some additional text
        if "```json" in sentiment_data:
            json_part = sentiment_data.split('```json', 1)[-1]  # Get part after the '```json'
            json_part = json_part.split('```', 1)[0].strip()   # Get the JSON part before the closing '```'
        else:
            json_part = sentiment_data.strip()
        
        # Debugging: log the part that should be JSON
        print("Extracted JSON part:", json_part)

        # Check if json_part is a valid JSON string
        json.loads(json_part)  # Validate the JSON
        return json_part
    except json.JSONDecodeError as decode_err:
        print("Error decoding JSON:", str(decode_err))
        return None
    except ValueError as ve:
        print("ValueError:", str(ve))
        return None

if __name__ == '__main__':
    app.run(debug=True)
