# A RAG System Build for Manuals

Are you looking for a RAG system that can **help you read and search through manuals and documents** with both text and images? Here it is! This project is a RAG system contains document indexing, retrieving, and complex question answering. It first index and store the document in FAISS vector store with fixed sized chunk and later retrieve based on query. It not only has the capability to answer simple and direct questions like "what is an axial scan?", it also has the ability to compare different concept,correct assumption, analyze statement and answering much more complex questions. You can attach your defined document and run the run_ragFAISS.bat script. Here are the things needed for setup.

## Setup:
1. Environment setup - please follow the requirement.txt for any pacakge settings
2. Create a log file called deepseek_api_auth.log to contain the deepseek api key on the first line. (only need to contain the key from the first character, nothing else)
3. Create a log file called HF_auth.log to contain the hugging face token on the first line.

## Usage:
### First time user
After setup is complete, direclty run .\run_ragFAISS.bat
- it may take a bit to build the context
### After the first time
After the first time running, a copy of the faiss index will be stored in the project locally. If you changed a document or any parameters, please remove the index copy by running:
```
Remove-Item -Recurse -Force faiss_index
```
Then run the .\run_ragFAISS.bat directly to rebuild the index and continue to search.

## Roadmap:
- [x] first step is to accelerate the index building and searching process (check)
- [x] second step is to enhance the reasoning capabilities 
- [ ] Add image capabilities
