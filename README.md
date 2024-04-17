# Chainlit_RAG
https://www.loom.com/share/5f9e5cd770094c05acec48a150200e89?sid=d40ace7d-95ad-4a46-985a-5022f2d6e388

Due to unavailability of free credits I used the following alternatives
<table>
    <tr><th>ALternative Chosen</th><th>Orignal requirement</th></tr>
    <tr><td>Gemini-Pro model</td><td>GPT4</td></tr>
    <tr><td>BGE-embeddings</td><td>Openai-embeddings</td></tr>
    <tr><td>Pinecone vectorstore</td><td>Typesense</td></tr>
</table>

## Code Structure
The project has the following structure:

- `dataingestion.py`: This is the script related to data ingestion.
- `app.py`: This is the main application file.

### dataingestion/

This script is for data ingestion. It use BGE-Embeddings to create a vector representation of the pdfs present in data folder and store them on pinecone vectorstores.

### app.py

This is the main application file. It initializes the application and contains the main routing logic.
It uses LCEL to create the conversation retrieval chain with memory and uses chainlit to implement the UI.