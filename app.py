# The Website reads the Document's given to it/ we fetch the data from the website and then convert it to text chunks.
# After converting it to text chunks it is converted to text embeddings and stored in vector database.
# Later this vector database is stored to pinecone and is successfully pushed.
# We basically take the data, break it into chunks and then convert it into vector embeddings and then store it to the vector store


import streamlit as st
from utils import *
import constants


# Creating Session Variables
if "HuggingFace_API_Key" not in st.session_state:
    st.session_state["HuggingFace_API_Key"] = ""

if "Pinecone_API_Key" not in st.session_state:
    st.session_state["Pinecone_API_Key"] = ""


st.title("NitroGPT Navigator 🎯")


# Sidebar Functionality


# Capturing the API Keys in the Sidebar
st.sidebar.title("API Keys, A Top Secret!!")
st.session_state["HuggingFace_API_Key"] = st.sidebar.text_input(
    "Enter The Hugging Face API Key", type="password"
)
st.session_state["Pinecone_API_Key"] = st.sidebar.text_input(
    "Enter The Pinecone API Key", type="password"
)


load_button = st.sidebar.button("Load The Data")


# When the button is Clicked, then Push the Data to Pinecone

if load_button:
    # Proceed Only if API Keys have been provided
    if (
        st.session_state["HuggingFace_API_Key"] != ""
        and st.session_state["Pinecone_API_Key"] != ""
    ):
        # Fetch Data From The Site
        site_data = get_website_data(constants.WEBSITE_URL)
        st.write("Data Pulling has been done...")

        # Splitting The Data Into Chunks
        chunks_data = split_data(site_data)
        st.write("Spliting Of Data has been done...")

        # Creating Embeddings Instance
        embeddings = create_embeddings()
        st.write("Embeddings Instance has been created...")

        # Pushing Data To Pinecone
        push_to_pinecone(
            st.session_state["Pinecone_API_Key"],
            constants.PINECONE_ENVIRONMENT,
            constants.PINECONE_INDEX,
            embeddings,
            chunks_data,
        )
        st.write("Pushing of Data To Pinecone has been done...")

        st.sidebar.success("Data Pushed To Pinecone Successfully!")

    else:
        st.sidebar.error("Please Provide The API Keys!!!")


############################################################################################


# Capturing The User Input and working on that

prompt = st.text_input("Heyoo! I'm NitroGPT Navigator. How May I Help You? 😎")

# Number Of Words To Return In Response
word_count = st.slider("What's The Size Of Response You Want?")

submit = st.button("Search")

if submit:
    # Proceed only if API Keys are given
    if (
        st.session_state["HuggingFace_API_Key"] != ""
        and st.session_state["Pinecone_API_Key"] != ""
    ):
        # Creating Embeddings Instance
        embeddings = create_embeddings()
        st.write("Embeddings Instance has been created...")

        # Pull Index Data From Pinecone
        index = pull_from_pinecone(
            st.session_state["Pinecone_API_Key"],
            constants.PINECONE_ENVIRONMENT,
            constants.PINECONE_INDEX,
            embeddings,
        )
        st.write("Pinecone Index Retrieval Done...")

        # Fetch Relevant Documents From Pinecone Index
        relevant_docs = get_similar_docs(index, prompt, word_count)
        st.write(relevant_docs)

    else:
        st.sidebar.error("Ooops!! Please Provide The API Keys...")
