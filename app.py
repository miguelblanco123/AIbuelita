import streamlit as st
import json
import os
import csv
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import random


# Configure Streamlit page
st.set_page_config(page_title="AIbuelita: Salsa Recipes", page_icon="🌶️", layout="wide")

# Define Mexican villa-inspired color palette
COLORS = {
    "terra_cotta": "#E07A5F",
    "marigold": "#F2CC8F",
    "turquoise": "#81B29A",
    "indigo": "#3D405B",
    "cream": "#F4F1DE",
    "white": "#FFFFFF",
    "black": "#000000"
}

st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS['cream']};
        color: {COLORS['indigo']};
    }}
    .stButton>button {{
        background-color: {COLORS['terra_cotta']};
        color: {COLORS['white']};
        font-weight: bold;
        border: 2px solid {COLORS['marigold']};
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['marigold']};
        color: {COLORS['indigo']};
        border-color: {COLORS['terra_cotta']};
    }}
    .st-bb {{
        background-color: {COLORS['turquoise']};
    }}
    .st-at {{
        background-color: {COLORS['marigold']};
    }}
    h1, h2, h3 {{
        color: {COLORS['indigo']};
    }}
    .stTextInput>div>div>input {{
        background-color: {COLORS['white']};
        color: {COLORS['indigo']};
        border: 2px solid {COLORS['turquoise']};
        border-radius: 10px;
    }}
    .stSelectbox>div>div>div {{
        background-color: {COLORS['white']};
        color: {COLORS['indigo']};
        border: 2px solid {COLORS['turquoise']};
        border-radius: 10px;
    }}
    .sidebar .sidebar-content {{
        background-color: {COLORS['marigold']};
    }}
</style>
""", unsafe_allow_html=True)

# Load avatar images
user_avatar = Image.open("images/User.png")
bot_avatar = Image.open("images/AIbuelita.png")

# Configure ChatOpenAI client
chat = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# Load knowledge and prepare vectorizer
@st.cache_resource
def load_knowledge():
    with open('data/salsas.json', 'r', encoding='utf-8') as f:
        conocimiento = json.load(f)
    vectorizer = TfidfVectorizer()
    salsa_texts = [f"{salsa['name']} {' '.join(salsa['flavor_notes'])} {' '.join(salsa['pairs_well_with'])}" for salsa in conocimiento['salsa_recipes']]
    X = vectorizer.fit_transform(salsa_texts)
    return conocimiento, vectorizer, X

# New function to get all salsa names
def get_all_salsa_names(conocimiento):
    return [salsa['name'] for salsa in conocimiento['salsa_recipes']]

def process_query(query, conocimiento, vectorizer, X):
    all_salsa_names = get_all_salsa_names(conocimiento)
    salsa_names_str = ", ".join(all_salsa_names)

    system_prompt = f"""You are AIbuelita, an AI assistant specialized in Mexican salsas. Your task is to respond to the user's query about salsas and always recommend a salsa, even for general questions. Follow these guidelines:

1. For specific dish queries, recommend a suitable salsa and explain why it's a good match.
2. For general questions, provide relevant information and still recommend a salsa that relates to the query.
3. If asked to list salsas, use ONLY the following list of salsas: {salsa_names_str}. Do not invent or add any salsas not in this list.
4. For unusual requests (like the "weirdest" salsa), use your knowledge creatively to recommend an interesting salsa from the provided list.
5. Always maintain a warm, grandmotherly tone in your responses.
6. Conclude your response by mentioning why the recommended salsa is a good choice.
7. If the user specifically asks for all salsa names or a complete list of salsas, provide the full list of salsa names without any additional commentary.
8. If the asked salsa is practically the same name as one of the salsas in the list, recommend the one in the list. Do not say that the salsa is not in the list.

Respond in a concise yet informative manner, always including a salsa recommendation from the provided list, unless specifically asked for the full list of salsas."""

    user_prompt = f"User query: {query}\n\nPlease respond to this query about salsas and provide a salsa recommendation from the list provided, or the full list if requested."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = chat(messages)
    return response.content

def encontrar_salsa_similar(query, conocimiento, vectorizer, X):
    query_vector = vectorizer.transform([query])
    similitudes = cosine_similarity(query_vector, X)
    indice_mas_similar = similitudes.argmax()
    return conocimiento['salsa_recipes'][indice_mas_similar]

def guardar_feedback(query, salsa_recomendada, util):
    filename = 'feedback_salsas.csv'
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['fecha', 'query', 'salsa_recomendada', 'util']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query': query,
            'salsa_recomendada': salsa_recomendada['name'],
            'util': util
        })

def main():
    # Load knowledge
    conocimiento, vectorizer, X = load_knowledge()

    # Sidebar
    st.sidebar.image("images/AIbuelita.png", use_column_width=True)
    st.sidebar.title("AIbuelita")
    st.sidebar.markdown("""
    ¡Hola, mi niño! Soy AIbuelita, tu experta en salsas mexicanas. 
    Estoy aquí para ayudarte a encontrar la salsa perfecta para tus platillos favoritos 
    o responder cualquier pregunta que tengas sobre nuestras deliciosas salsas. 
    ¡Pregúntame lo que quieras!
    
    ¡Que aproveche!
    """)

    # Main content
    st.title("🌶️ La Cocina de AIbuelita 🌮")
    st.markdown("---")

    st.header("¿Qué quieres saber sobre las salsas, mi niño?")
    query = st.text_input("Pregúntame sobre salsas, platillos, o lo que quieras saber:", placeholder="Ej. Recomiéndame una salsa para tacos al pastor, o ¿Cuáles son todas las salsas que conoces?")

    if st.button("¡Cuéntame, Abuelita!"):
        if query:
            with st.spinner('Ay mi niño, déjame pensar en la mejor salsita para ti...'):
                respuesta = process_query(query, conocimiento, vectorizer, X)
                salsa_recomendada = encontrar_salsa_similar(query, conocimiento, vectorizer, X)
            
            st.success("¡Listo! Aquí tienes mi respuesta, corazón:")
            st.markdown(f"{respuesta}")
            
            with st.expander("Ver receta de la salsa recomendada"):
                st.subheader(f"Receta: {salsa_recomendada['name']}")
                st.write("**Ingredientes:**")
                for ingredient in salsa_recomendada['ingredients']:
                    st.markdown(f"- {ingredient}")
                st.write("**Instrucciones:**")
                for i, instruction in enumerate(salsa_recomendada['instructions'], 1):
                    st.markdown(f"{i}. {instruction}")

if __name__ == "__main__":
    main()