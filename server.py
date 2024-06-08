import os
from flask import Flask, request
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch
from operator import itemgetter
from langchain.schema.runnable import RunnableMap
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES
import time
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
port = int(os.environ.get('PORT', 3000))


load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")


@app.route('/indexdocument', methods = ['POST'])
def indexdocument():
    print("In indexdocument")
    fileName = request.headers['Content-Disposition'].split('=')[1]
    print(fileName)
    FileStorage(request.stream).save(os.path.join("./", fileName))
    # Check if the file exists
    if os.path.exists(os.path.join("./", fileName)):
        # The file exists, you can use it now
            vector_store = None
            aoai_embeddings = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-ada-002",
                openai_api_version="2023-05-15",  # e.g., "2023-12-01-preview"
            )
            # Initiate Azure AI Document Intelligence to load the document. You can either specify file_path or url_path to load the document.
            loader = AzureAIDocumentIntelligenceLoader(file_path="./" + fileName, api_key = doc_intelligence_key, api_endpoint = doc_intelligence_endpoint, api_model="prebuilt-layout")
            #loader = AzureAIDocumentIntelligenceLoader(url_path=document_url, api_key = doc_intelligence_key, api_endpoint = doc_intelligence_endpoint, api_model="prebuilt-layout")
            print(doc_intelligence_endpoint)
            docs = loader.load()

            # Split the document into chunks base on markdown headers.
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

            docs_string = docs[0].page_content
            #print(docs_string)
            splits = text_splitter.split_text(docs_string)

            print("Length of splits: " + str(len(splits)))


            vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
            vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")


            index_name: str = get_random_name(separator="-", style="lowercase")
            vector_store: AzureSearch = AzureSearch(
                azure_search_endpoint=vector_store_address,
                azure_search_key=vector_store_password,
                index_name=index_name,
                embedding_function=aoai_embeddings.embed_query,
                semantic_configuration_name="test"
            )

            vector_store.add_documents(documents=splits)
            #time.sleep(0.5)
            return index_name
    else:
        # The file does not exist, handle the error
        print("File does not exist")


@app.route('/chatwithdocument')
def chatwithdocument():
    question=request.args.get("question")
    index_name=request.args.get("index_name")
    vector_store = None
    retriever = None
    aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",  # e.g., "2023-12-01-preview"
    )

    vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
    vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    index_name: str = index_name
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=aoai_embeddings.embed_query,
        semantic_configuration_name="test"
    )
    retriever = vector_store.as_retriever(search_type="semantic_hybrid")
    
    # Use a prompt for RAG that is checked into the LangChain prompt hub (https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=989ad331-949f-4bac-9694-660074a208a7)
    prompt = hub.pull("rlm/rag-prompt")
    #print(prompt)
    llm = AzureChatOpenAI(
        openai_api_version="2024-02-01",  # e.g., "2023-12-01-preview"
        azure_deployment="gpt-35-turbo",
        temperature=0,
    )

    def format_docs(docs):
        #print(docs[:3])
        return "\n\n".join(doc.page_content for doc in docs[:3])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Ask a question about the document

    result = rag_chain.invoke(question)

    return result
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)