import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime

# Optional imports (OpenAI) — used only if API key is available
try:
    import openai
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

# Hugging Face fallback
try:
    from transformers import pipeline, Conversation
    HF_INSTALLED = True
except Exception:
    HF_INSTALLED = False

# ----------------------
# Config
# ----------------------
MEMORY_FILE = Path("memories.json")
HISTORY_KEY = "chat_history"
MEMORIES_KEY = "memories"

st.set_page_config(page_title="Chatbot Conversacional con Memoria", layout="centered")
st.title("🤖 Chatbot conversacional con memoria")
st.write("Una app de ejemplo con memoria a corto/largo plazo. Puedes usar OpenAI si configuras la clave, o caer al modelo de Hugging Face si no.")

# ----------------------
# Helpers: load/save memories
# ----------------------

def load_memories():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_memories(memories):
    MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------
# Initialize session state
# ----------------------
if HISTORY_KEY not in st.session_state:
    st.session_state[HISTORY_KEY] = []  # list of {role: 'user'|'bot', 'text': ...}

if MEMORIES_KEY not in st.session_state:
    st.session_state[MEMORIES_KEY] = load_memories()

# ----------------------
# Model loading
# ----------------------
@st.cache_resource
def init_hf_bot():
    # Conversational pipeline (fallback)
    if not HF_INSTALLED:
        return None
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

hf_bot = None
use_openai = False

# Check for OpenAI key in env or Streamlit secrets
openai_key = None
if "OPENAI_API_KEY" in os.environ:
    openai_key = os.environ["OPENAI_API_KEY"]
elif st.secrets.get("OPENAI_API_KEY") if hasattr(st, 'secrets') else None:
    openai_key = st.secrets.get("OPENAI_API_KEY")

if openai_key and OPENAI_INSTALLED:
    use_openai = True
    openai.api_key = openai_key
else:
    # Fallback to HF
    if HF_INSTALLED:
        hf_bot = init_hf_bot()

# ----------------------
# Simple memory retriever
# ----------------------

def get_relevant_memories(text, memories, top_k=3):
    """Retrieve memories that share tokens with the input text (simple heuristic).
    Returns up to top_k memories ordered by match count."""
    twords = set([w.lower() for w in text.split() if len(w) > 2])
    scored = []
    for m in memories:
        mwords = set([w.lower() for w in m['text'].split() if len(w) > 2])
        score = len(twords & mwords)
        scored.append((score, m))
    scored = [m for m in scored if m[0] > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]

# ----------------------
# UI: sidebar for memory management
# ----------------------
with st.sidebar:
    st.header("Memoria Larga")
    st.write("Gestiona recuerdos que el bot usará como contexto adicional.")

    # Show existing memories
    if st.session_state[MEMORIES_KEY]:
        for i, mem in enumerate(st.session_state[MEMORIES_KEY]):
            st.write(f"**{i+1}.** {mem['text']}")
            cols = st.columns([1, 1, 3])
            if cols[0].button("Eliminar", key=f"del_{i}"):
                st.session_state[MEMORIES_KEY].pop(i)
                save_memories(st.session_state[MEMORIES_KEY])
                st.rerun()
            if cols[1].button("Editar", key=f"edit_{i}"):
                new = st.text_input("Editar memoria", value=mem['text'], key=f"edit_input_{i}")
                if st.button("Guardar", key=f"save_mem_{i}"):
                    st.session_state[MEMORIES_KEY][i]['text'] = new
                    save_memories(st.session_state[MEMORIES_KEY])
                    st.rerun()

    else:
        st.write("(Sin recuerdos guardados)")

    st.markdown("---")
    new_mem = st.text_area("Añadir memoria (hecho importante sobre el usuario):", "Ej: Le gusta el fútbol y trabaja en educación.")
    if st.button("Guardar memoria"):
        if new_mem.strip():
            mem_obj = {"text": new_mem.strip(), "created_at": datetime.utcnow().isoformat()}
            st.session_state[MEMORIES_KEY].append(mem_obj)
            save_memories(st.session_state[MEMORIES_KEY])
            st.success("Memoria guardada")
            st.rerun()


    st.markdown("---")
    st.write("Configuración:")
    st.write(f"Usando OpenAI: {use_openai}")
    if not use_openai and not hf_bot:
        st.warning("No hay modelo disponible. Instala 'openai' (opcional) o 'transformers' y 'torch'.")

# ----------------------
# Main chat UI
# ----------------------

col1, col2 = st.columns([4,1])
with col1:
    for msg in st.session_state[HISTORY_KEY]:
        if msg['role'] == 'user':
            st.markdown(f"**Tú:** {msg['text']}")
        else:
            st.markdown(f"**Bot:** {msg['text']}")

with col2:
    st.write("")

user_input = st.text_input("Escribe un mensaje:", key="input_box")
save_as_memory = st.checkbox("¿Guardar este mensaje como recuerdo?", value=False)

if st.button("Enviar") and user_input.strip():
    # Append user message
    st.session_state[HISTORY_KEY].append({'role': 'user', 'text': user_input})

    # Optionally save as memory
    if save_as_memory:
        mem_obj = {"text": user_input.strip(), "created_at": datetime.utcnow().isoformat()}
        st.session_state[MEMORIES_KEY].append(mem_obj)
        save_memories(st.session_state[MEMORIES_KEY])

    # Retrieve relevant memories
    relevant = get_relevant_memories(user_input, st.session_state[MEMORIES_KEY], top_k=3)

    # Prepare context
    memory_texts = [m['text'] for m in relevant]
    memory_context = "\n".join([f"- {t}" for t in memory_texts])
    if memory_context:
        memory_header = "Recuerdos relevantes:\n" + memory_context + "\n---\n"
    else:
        memory_header = ""

    # Build prompt / call model
    if use_openai:
        # Build messages: system + conversation
        messages = []
        system_msg = "Eres un asistente conversacional útil y amable."
        if memory_header:
            system_msg += "\nContexto adicional: " + memory_context
        messages.append({"role": "system", "content": system_msg})

        # Add history (user + assistant)
        for m in st.session_state[HISTORY_KEY]:
            role = "user" if m['role']== 'user' else 'assistant'
            messages.append({"role": role, "content": m['text']})

        # Call OpenAI ChatCompletion
        resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        bot_text = resp['choices'][0]['message']['content'].strip()

    else:
        # Hugging Face conversational fallback
        if not hf_bot:
            bot_text = "Error: No hay modelo disponible. Instala 'transformers' y 'torch' o configura OPENAI_API_KEY."
        else:
            conv = Conversation(user_input)
            try:
                hf_bot(conv)
                # The pipeline stores responses in conv.generated_responses
                bot_text = conv.generated_responses[-1] if conv.generated_responses else ""
            except Exception as e:
                bot_text = f"Error durante la generación: {e}"

    # Append bot reply
    st.session_state[HISTORY_KEY].append({'role': 'bot', 'text': bot_text})
    st.rerun()


# ----------------------
# Footer: show memories being used
# ----------------------
st.markdown("---")
st.subheader("Memoria activa")
if st.session_state[MEMORIES_KEY]:
    st.write(f"Tienes {len(st.session_state[MEMORIES_KEY])} recuerdos guardados.")
    st.write("Si quieres que el bot recuerde menos, elimina recuerdos en la barra lateral.")
else:
    st.write("No hay recuerdos guardados aún.")

st.caption("Archivo de memoria local: memories.json")
