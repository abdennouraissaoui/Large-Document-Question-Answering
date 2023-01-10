import streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai

openai.api_key = streamlit.secrets['openai_api_key']

QUERY_EMBEDDINGS_MODEL = f"text-search-curie-query-001"
SEPARATOR = "\n* "
MAX_SECTION_LEN = 1000
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 1200,
    "model": 'text-davinci-003',
}


def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section['token_count']
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section['paragraph'].replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    header = """Answer the question as truthfully as possible using the provided context on the company Netflix, and if the answer is not contained within the context below or if it's unclear, say "Not sure" or "Answer is not found."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
        show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )

    return response["choices"][0]["text"].strip(" \n")

st.title('Answering Questions from Netflix 2022 FORM-10K')
st.write(f"Link to the document: [Netflix 2022 FORM 10K](https://s22.q4cdn.com/959853165/files/doc_financials/2021/q4/da27d24b-9358-4b5c-a424-6da061d91836.pdf)")
st.write(f"Link to the code: [Jupyter Notebook](https://github.com/abdennouraissaoui/Large-Document-Question-Answering/blob/main/NFLX_10K_Question_Answering.ipynb)")

df = pd.read_csv("nflx_10K.csv")
document_embeddings = pickle.load(open('nflx_10k_embeddings.pkl', "rb"))

question = st.text_input("Insert question/query here:", placeholder="Ex: How many paid memberships does Netflix have, list for me the risks associated with Netflix, etc")
st.text("")
if question != "":
    with st.spinner(text="Searching for answer..."):
        answer = answer_query_with_context(question, df, document_embeddings)
        st.markdown(answer)
