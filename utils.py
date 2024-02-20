from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI


def init_conversation(openai_api_key):
    convesation =ConversationChain(
            llm=ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name="gpt-3.5-turbo",
                temperature=0
            ),
            verbose=True,
            memory=ConversationBufferMemory()
        )
    
    return convesation

def predic_memory_conversation(cconvesation, prompt):

    response = cconvesation.predict(input=prompt)

    return response


def clear_chat_history(st, msg_chatbot):
    st.session_state.messages = [{"role" : "assistant", "content": msg_chatbot}]
    st.session_state.conversation = ""



## Se envía el prompt de usuario al modelo de GPT-3.5-Turbo para que devuelva una respuesta
def get_response_openai(prompt, openai_api_key):
    
    model = "gpt-3.5-turbo"

    llm = ChatOpenAI(
        openai_api_key = openai_api_key,
        model_name = model,
        temperature = 0
    )

    conversation = ConversationChain(
        llm = llm, 
        verbose = True, 
        memory = ConversationBufferMemory()
        )   

    print(conversation)


    return conversation.predict(input=prompt)

