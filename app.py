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
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AIbuelita: Salsa Recipes", page_icon="🌶️", layout="wide")

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

user_avatar = Image.open("images/User.png")
bot_avatar = Image.open("images/AIbuelita.png")

chat = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key = os.getenv("OPENAI_API_KEY"),
)

@st.cache_resource
def load_knowledge():
    with open('data/salsas.json', 'r', encoding='utf-8') as f:
        conocimiento = json.load(f)
    vectorizer = TfidfVectorizer()
    salsa_texts = [f"{salsa['name']} {' '.join(salsa['flavor_notes'])} {' '.join(salsa['pairs_well_with'])}" for salsa in conocimiento['salsa_recipes']]
    X = vectorizer.fit_transform(salsa_texts)
    return conocimiento, vectorizer, X

def get_all_salsa_names(conocimiento):
    return [salsa['name'] for salsa in conocimiento['salsa_recipes']]

def process_query(query, conocimiento, vectorizer, X):
    all_salsa_names = get_all_salsa_names(conocimiento)
    salsa_names_str = ", ".join(all_salsa_names)

    # Find the most similar salsa
    salsa_recomendada = encontrar_salsa_similar(query, conocimiento, vectorizer, X)

    system_prompt = f"""You are AIbuelita, an AI assistant specialized in Mexican salsas. Your task is to respond to the user's query about salsas and always recommend the following salsa: {salsa_recomendada['name']}. Follow these guidelines:

1. For all queries, recommend {salsa_recomendada['name']} and explain why it's a good match.
2. Use the exact ingredients and instructions provided for {salsa_recomendada['name']}.
3. For general questions, provide relevant information and still relate it to {salsa_recomendada['name']}.
4. Always maintain a warm, grandmotherly tone in your responses.
5. Conclude your response by mentioning why {salsa_recomendada['name']} is a good choice.
6. If the user specifically asks for all salsa names or a complete list of salsas, provide the full list of salsa names without any additional commentary.

Respond in a concise yet informative manner, always in Spanish, always including information about {salsa_recomendada['name']}, unless specifically asked for the full list of salsas."""

    user_prompt = f"User query: {query}\n\nPlease respond to this query about salsas and provide information about {salsa_recomendada['name']}, or the full list if requested."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = chat(messages)
    return response.content, salsa_recomendada

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
                respuesta, salsa_recomendada = process_query(query, conocimiento, vectorizer, X)
            
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