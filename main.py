# main.py
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import pandas as pd
from typing import List, Dict, Any
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI
azure_endpoint = os.getenv('AZURE_ENDPOINT')
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_embeddings_endpoint = os.getenv('AZURE_EMBEDDINGS_ENDPOINT')
azure_embeddings_api_key = os.getenv('AZURE_EMBEDDINGS_API_KEY')

def clean_text(text: str) -> str:
    cleaned = ' '.join(text.split())
    cleaned = cleaned.replace('electronics', 'Electronics')
    cleaned = cleaned.replace('appliances', 'Appliances')
    return cleaned

def enrich_content(content: str, metadata: Dict[str, Any]) -> str:
    if 'category' in metadata:
        content = f"Category: {metadata['category']} - {content}"
    return content

def format_numerical_data(content: str) -> str:
    parts = content.split()
    formatted_parts = []
    for part in parts:
        if part.replace('.', '').isdigit():
            if '.' in part:
                formatted_parts.append(f"${part}")
            else:
                formatted_parts.append(f"{part} units")
        else:
            formatted_parts.append(part)
    return ' '.join(formatted_parts)

def transform_documents(documents: List[Document]) -> List[Document]:
    transformed_docs = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        enriched_content = enrich_content(cleaned_content, doc.metadata)
        formatted_content = format_numerical_data(enriched_content)
        
        transformed_doc = Document(
            page_content=formatted_content,
            metadata={
                **doc.metadata,
                'processed': True,
                'content_length': len(formatted_content)
            }
        )
        transformed_docs.append(transformed_doc)
    
    return transformed_docs


def initialize_pinecone():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
    cloud = os.getenv('PINECONE_CLOUD', 'aws')
    region = os.getenv('PINECONE_REGION', 'us-east-1')
    
    spec = ServerlessSpec(cloud=cloud, region=region)
    pinecone = Pinecone(api_key=pinecone_api_key)
    
    return pinecone, pinecone_index_name, spec

def create_or_get_index(pinecone, index_name, dimension, spec):
    if index_name in pinecone.list_indexes().names():
        print(f"Index {index_name} already exists")
    else:
        print(f"Creating index {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            spec=spec
        )
    return pinecone.Index(index_name)

def prepare_vectors(chunks, document_embeddings):
    vectors = [
        {"id": f"chunk-{i}", "values": embedding,
            "metadata": {"text": chunk.page_content}}
        for i, (chunk, embedding) in enumerate(zip(chunks, document_embeddings))
    ]
    return vectors

def initialize_vectorstore(splits, embeddings):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Upsert documents
    vectorstore = PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=index_name
    )

    print(f"Initialized vectorstore with {len(splits)} documents")
    
    return vectorstore

def load_and_process_data(embeddings):
    # Load CSV
    loader = CSVLoader(
        file_path='product_inventory.csv',
        source_column='product_id'
    )
    documents = loader.load()   
   
    transformed_docs = transform_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(transformed_docs)
    print(f"Loaded and processed data", len(chunks))

    document_embeddings = [embeddings.embed_query(chunk.page_content) for chunk in chunks]
    return chunks, document_embeddings

def setup_qa_chain(vectorstore, llm_model):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def main():
    # Create dummy data
    create_dummy_csv()
    
    # Initialize models
    llm_model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-05-01-preview",
        deployment_name="gpt-35-turbo",
        model_name="gpt-35-turbo"
    )
    
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=azure_embeddings_endpoint,
        api_key=azure_embeddings_api_key,
        api_version="2024-05-01-preview",
        model='text-embedding-3-small'
    )
    
    # Process data
    # Initialize Pinecone
    pinecone, index_name, spec = initialize_pinecone()
    
    # Load and process data
    chunks, document_embeddings = load_and_process_data(embeddings)
    
    # Create or get index
    index = create_or_get_index(
        pinecone,
        index_name,
        dimension=len(document_embeddings[0]),
        spec=spec
    )
    
    # Prepare and upsert vectors
    vectors = prepare_vectors(chunks, document_embeddings)
    index.upsert(vectors=vectors)
    
    print("Index stats:", index.describe_index_stats())


    
    # Upsert documents
    vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
    
    # Initialize QA chain
    qa_chain = setup_qa_chain(vectorstore, llm_model)
    
    # Example queries
    queries = [
        "Which patients have high blood pressure?",
        "Who has diabetes according to the records?",
        "Identify patients with pre-diabetes based on glucose readings.",
        "What is the BP reading for a patient diagnosed with heart disease?",
        "What is the glucose reading for a patient with diabetes?",
        "What is the recommended treatment for hypertension?",
        "How should diabetes be managed according to the records?",
        "What are the notes for a patient diagnosed with asthma?",
        "Which patient is undergoing rehabilitation therapy for a stroke?",
        "What is the prescribed treatment for chronic fatigue syndrome?
    ]
    
    # Run queries
    for query in queries:
        print(f"\nQuery: {query}")
        response = qa_chain({"query": query})
        print("Answer:", response["result"])
        #print("Sources:", [doc.metadata['source'] for doc in response["source_documents"]])

if __name__ == "__main__":
    main()
