# Smart Grid Optimization

## Setup

### Step 1: Clone the Repository
```bash
git clone github.com/SaiKarthikC-137/Smart-Grid-Optimization
cd "Smart Grid Optimization"
```

### Step 2: Environment Setup
Create a virtual environment and install the required dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Step 3: Create .env File
Create a `.env` file in the root directory and populate it with the necessary API keys:
```
GROQ_API_KEY=<your_groq_api_key_here>
JINA_API_KEY=<your_jina_api_key_here>
HF_TOKEN=<you_huggingface_token_here>
HF_HOME=<your_chosen_directory_for_hf>
```

### Step 4: Build the Model
Run the provided notebook to build the .keras model using your TensorFlow version:
jupyter notebook build_model.ipynb
After executing the notebook, ensure that the model is saved as a `.keras` file in the designated directory.

### Step 5: First Run
Note that the first run will take a significant amount of time due to document embedding generation. These embeddings will be stored and reused in subsequent runs to improve performance.

## Running the Application
Use Streamlit to run the application. This will start a web interface for interacting with the model.
```bash
streamlit run app.py
```

## Models
The supporting LLM models used in this project are:
- Llama3-70b-8192
- Mixtral-8x7b-32768
- Gemma-7b-it

## Contributors
List the people who have contributed to this project.

## License
Include a section for the license if applicable.
