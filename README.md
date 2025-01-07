# Medical Data RAG System

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/langchain-0.1.0-orange)
![Pinecone](https://img.shields.io/badge/pinecone-latest-yellow)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-2024--05--01--preview-blue)

A Retrieval-Augmented Generation (RAG) system for medical data (patient data) using LangChain, Pinecone, and Azure OpenAI.

## Table of Contents
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Package Details](#package-details)
- [Architecture](#architecture)

## Technology Stack

- **Python**: Core programming language
- **LangChain**: Framework for building LLM applications
- **Pinecone**: Vector database for storing embeddings
- **Azure OpenAI**: LLM and embeddings provider

## Prerequisites

- Python 3.9 or higher
- Azure OpenAI account with API access
- Pinecone account with API key
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kambojananya/rag_using_medical_csv_data
cd rag_using_medical_csv_data
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Unix/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file with the following variables:
```env
AZURE_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_EMBEDDINGS_ENDPOINT=your_embeddings_endpoint
AZURE_EMBEDDINGS_API_KEY=your_embeddings_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Usage

1. Prepare your product inventory CSV file
2. Run the main script:
```bash
python main.py
```

## Package Details

### Core Dependencies

- **langchain** (v0.1.0+)
  - Purpose: Orchestrates the RAG pipeline
  - Key components: Document loaders, text splitters, embeddings integration

- **pinecone-client** (latest)
  - Purpose: Vector database operations
  - Features: Serverless deployment, vector storage, similarity search

- **openai** (latest)
  - Purpose: Interface with Azure OpenAI
  - Used for: Text embeddings, LLM queries

- **python-dotenv** (latest)
  - Purpose: Environment variable management
  - Loads configuration from .env file

### Optional Dependencies

- **pandas** (latest)
  - Purpose: Data manipulation
  - Used for: CSV processing, data transformation

## Architecture

```mermaid
graph LR
    A[CSV Data] --> B[Document Loader - CSVLoader]
    B --> C[Transformer]
    C --> D[Text Splitter]
    D --> E[Embeddings]
    E --> F[Pinecone Vector Store]
    G[Query] --> H[Embeddings]
    H --> F
    F --> I[RAG Chain]
    I --> J[Response]
```

### Data Flow

1. **Data Ingestion**: CSV loader processes product inventory
2. **Transformation**: Document transformer cleanses and enriches data
3. **Chunking**: Text splitter creates optimal chunks
4. **Vectorization**: Azure OpenAI creates embeddings
5. **Storage**: Vectors stored in Pinecone
6. **Retrieval**: Similar vectors retrieved for queries
7. **Generation**: LLM generates responses using retrieved context

## Development

### Adding New Products

1. Update the CSV file with new products
2. Run the script to update vectors:
```bash
python update_inventory.py
```

### Modifying Transformers

Edit the transformer functions in `main.py`:
- `clean_text()`
- `enrich_content()`
- `format_numerical_data()`

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch
3. Submit pull request

## Support

For issues and questions, please open a GitHub issue.
