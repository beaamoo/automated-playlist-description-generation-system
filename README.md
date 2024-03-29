# Automated Playlist Description Generation (APDG) system

### File Structure

APDG/
│
├── src/                        # Source files for the APDG system
│   ├── data/                   # Data handling scripts
│   │   ├── __init__.py         # Makes Python treat the directories as containing packages
│   │   └── spotify_api.py      # Script to fetch data from the Spotify API
│   │
│   ├── model/                  # Transformer model and utilities
│   │   ├── __init__.py
│   │   ├── transformer.py      # Implementation of the transformer model
│   │   └── utils.py            # Utility functions for the model, e.g., tokenization
│   │
│   └── main.py                 # Main script to run the APDG system
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── model_exploration.ipynb # Explore model architecture, training, etc.
│   └── api_data_fetching.ipynb # Notebook for exploring Spotify API data fetching
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_spotify_api.py     # Tests for Spotify API data fetching
│   └── test_transformer.py     # Tests for transformer model functionality
│
├── requirements.txt            # Project dependencies
├── setup.py                    # Setup script for the APDG package
└── README.md                   # Project overview and setup instructions
