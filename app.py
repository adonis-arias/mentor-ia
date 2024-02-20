import streamlit as st
from PIL import Image
from utils import *
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.vectorstores import Pinecone as pc_vectorstorage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain.chains import RetrievalQA
import os



openai_api_key = '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'
pinecone_api_key = 'd1ed4513-7f4e-4bfd-912a-0feb8da27263'

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

st.set_page_config(page_title = "Chatbot usando Langchain, OpenAI y Streamlit", page_icon = "https://python.langchain.com/img/favicon.ico")

# Inicializar la memoria de la conversación
if 'llm_chain' not in st.session_state.keys():
    st.session_state.llm_chain = ConversationChain(
        llm=ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0
        ),
        verbose=True,
        memory=ConversationBufferMemory()
    )


with st.sidebar:

    st.title("Usando la API de OpenAI con Streamlit y Langchain")

    image = Image.open('logos.png')
    st.image(image, caption = 'OpenAI, Langchain y Streamlit')

    st.markdown(
        """
        Integrando OpenAI con Streamlit y Langchain. 
    """
    )


def clear_chat_history():
    st.session_state.messages = [{"role" : "assistant", "content": msg_chatbot}]

st.sidebar.button('Limpiar historial de chat', on_click = clear_chat_history)

msg_chatbot = """
¡Bienvenido! Soy tu chatbot mentor, aquí para ayudarte a potenciar tus habilidades de liderazgo.

¿Tienes preguntas sobre cómo mejorar en áreas clave? ¡Estoy aquí para ayudarte! Puedes consultarme sobre:

    - Estrategias de productividad
    - Consejos financieros prácticos
    - Cómo desarrollar disciplina efectiva
    - Estrategias para hacer crecer tu negocio
    - Mejorar tus habilidades de comunicación y relaciones humanas
    - Autoayuda y desarrollo personal
    - Explorar la espiritualidad para un liderazgo más consciente

Mi conocimiento se basa en los mejores libros y expertos en cada campo. ¡No dudes en preguntar!
"""


#Si no existe la variable messages, se crea la variable y se muestra por defecto el mensaje de bienvenida al chatbot.
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content" : msg_chatbot}]
    
    

if "context" not in st.session_state.keys():
    st.session_state.context = []
    

# Muestra todos los mensajes de la conversación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Obtener contexto de la memoria
context_memory = st.session_state.context

# Agregar mensajes anteriores al contexto
if context_memory:
    previous_messages = [item["input"] for item in context_memory]
    previous_messages.extend([item["output"] for item in context_memory])
    previous_conversation = " ".join(previous_messages)
else:
    previous_conversation = ""

# Create a Function to generate
def generate_openai_pinecone_response(prompt_input):
    
    model = "gpt-3.5-turbo"

    llm = ChatOpenAI(
        openai_api_key = openai_api_key,
        model_name = model,
        temperature = 0
    )

    template = """Responda a la pregunta basada en el siguiente contexto, responde como si fueras un gran mentor mediante consejos  y tips del contexto que puedan ayudar e inspirar.
    Si no puedes responder a la pregunta, usa la siguiente respuesta "¡Hola! Gracias por tu pregunta. Sin embargo, parece que está fuera del alcance de este chat o del enfoque del mentor. Nos centramos en ayudar a mejorar habilidades de liderazgo, como productividad, finanzas, disciplina, negocios, relaciones humanas, autoayuda y espiritualidad. Si tienes alguna pregunta relacionada con estos temas, estaré encantado de ayudarte. ¡Gracias por tu comprensión!"

    Contexto: 
    {context}.
    Pregunta: {question}
    Respuesta utilizando también emoticones: 
    """
    
    prompt = PromptTemplate(
        input_variables = ["context", "question"],
        template = template
    )

    chain_type_kwargs = {"prompt": prompt, "verbose": True}

    embeddings = OpenAIEmbeddings()
    
    # Connect with Pinecone

    index_name= 'idbookmentor'

    pc = Pinecone(api_key=pinecone_api_key)
    
    text_field = "text"
    # switch back to normal index for langchain
    index = pc.Index(index_name)
    vectorstore = pc_vectorstorage(
        index, embeddings.embed_query, text_field
    )

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = vectorstore.as_retriever(),
        verbose = True,
        chain_type_kwargs = chain_type_kwargs
    )

    output = qa.run(prompt_input)
    print(qa)

    return output


prompt = st.chat_input("Ingresa tu pregunta")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Agregar mensajes actuales a la conversación anterior
current_input = st.session_state.messages[-1]["content"]
full_conversation = previous_conversation + " " + current_input

# Generar una nueva respuesta si el último mensaje no es de un assistant, sino de un user, entonces entra al bloque de código
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Esperando respuesta, dame unos segundos."):

            response = generate_openai_pinecone_response(prompt_input=full_conversation)
            print(response)
            placeholder = st.empty()
            full_response = ''
            
            for item in response:
                full_response += item
                placeholder.markdown(full_response)

            placeholder.markdown(full_response)


    message = {"role" : "assistant", "content" : full_response}
    st.session_state.context.append({"input": prompt,"output" : full_response}) 
    context = st.session_state.context
    st.session_state.messages.append(message) #Agrega elemento a la caché de mensajes de chat.

