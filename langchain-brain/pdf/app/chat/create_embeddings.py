import os.path

import pypdf.errors
from langchain.document_loaders import PyPDFLoader #allows us to load data from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter # allows us to specify the chunks of text to extract
from app.chat.vector_stores.pinecone import vector_store # vector store that was created connecting langchain to the index in pinecone.io




def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    """ Generate and store embeddings for the given pdf
      :param pdf_id: The unique identifier for the PDF.
      :param pdf_path: The file path to the PDF.
      """
    # Check if the PDF file path exists
    if not os.path.exists(pdf_path):
        print(f"Error: The file {pdf_path} does not exist")
        return

    """ 2. Divide the extracted text into manageable chunks."""
    # create a text splitter instance to be used within the pdf load process
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 500,
        chunk_overlap= 100  #overlap is helpful for context in case there's a mid-sentence chunk
    )

    """1. Extract text from the specified PDF."""
    #load the PDF File using argument for the path of the PDF (Created at time of upload) and pdf_id (unique id)
    loader=PyPDFLoader(pdf_path)
    # try to split the pdf; else show errors
    try:
        """ 3. Generate an embedding for each chunk."""
        #use text splitter and loader together to split the pdf into chunks, saved as docs array
        docs=loader.load_and_split(text_splitter)
        # print the text_splitter success summary and docs
        print(f"Loaded {len(docs)} chunks from the PDF.")

        """4. loop over every document to update metadata to include info about pdf source name; remove 'source' that
         previously represented where the PDF was stored in the hard drive; add 'pdf_id' property to represent the pdf"""
        for doc in docs:
            doc.metadata={
                # metadeta currently stored in pinecone.io for reference
                "page": doc.metadata["page"],
                "text": doc.page_content,
                "pdf_id": pdf_id
            }

        """5. Persist the generated embeddings."""
        vector_store.add_documents(docs)
    except pypdf.errors.PdfReadError as e:
        print(f"Failed to read PDF: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

