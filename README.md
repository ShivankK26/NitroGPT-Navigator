# NitroGPT Navigator: GPT-3.5 Based AI ChatBot

NitroGPT Navigator is an advanced chatbot that leverages cutting-edge technologies, including OpenAI's GPT-3.5, LangChain, LLMs and Pinecone Vector Database. It's designed to provide accurate and context-aware answers to any questions related to the content on routerprotocol.com.

## Features

- **Powered by GPT-3.5:** NitroGPT Navigator utilizes the power of OpenAI's latest language model to generate human-like responses.
- **LangChain Integration:** LangChain enhances linguistic capabilities, enabling a more nuanced understanding of user queries.
- **Vector Embeddings:** Using a Large Language Model from HuggingFace, all-MiniLM-L6-V2, to convert Text to Numeric form for generating Vector Embeddings.
- **Pinecone Vector Database:** Leveraging Pinecone allows for efficient vector similarity searches, improving the accuracy of information retrieval.

## How It Works

NitroGPT Navigator operates by processing user queries through a multi-layered approach:

1. **User Input:** Users submit questions related to routerprotocol.com.
2. **LangChain Processing:** LangChain parses and understands the user's query, extracting key information.
3. **Vector Search with Pinecone:** Pinecone performs vector searches to identify relevant content.
4. **GPT-3.5 Response Generation:** OpenAI's GPT-3.5 generates context-aware responses based on the extracted information.

## Installation

To run NitroGPT Navigator locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/routerbot.git
   cd NitroGPT Navigator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - Obtain API keys for OpenAI, Hugging Face, and Pinecone.
   - Add these keys to the corresponding configuration files.

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

Once the application is running, users can interact with RouterBot by asking questions related to routerprotocol.com. The chatbot will provide informative and context-aware responses.

Example:
```bash
What is Router Nitro?
```

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.
