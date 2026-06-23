import asyncio

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=r"PolicyVB",
    embedding_function=embeddings
)


def Retriever(query, printer= False, k=5, typie='mmr')->str:
    '''
    this is retriever function which is used to retrieve the content based on the query from the vector db, it's take 3 parameters in which on parameter is dyanamic which takes query, and another remaining two are static which takes parameters like top k and type of searching algorithm.
    '''
    retriever = vector_db.as_retriever(search_type=typie, search_kwargs={"k": k, "fetch_k": 50})
    result = retriever.invoke(query)

    if printer:
        for i in result:
            print("Source: ", i.metadata['source'].split('\\')[-1].split('.')[0])
            print()
            print("Content: ",i.page_content)
            print()
            print("_"* 80)
    metadata = [i.metadata['source'].split('\\')[-1].split('.')[0] for i in result]

    context = "\n\n".join(
        [
            f"Source: {doc.metadata['source'].split('\\')[-1].split('.')[0]}\nContent: {doc.page_content}"
            for doc in result
        ]
    )

    return context

