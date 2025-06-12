import os
import json
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from fpdf import FPDF
from gtts import gTTS
import gradio as gr

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from langchain_community.chat_models import ChatOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#from huggingface_hub import login


# ✅ Load environment variables first
def load_env_var(var_name):
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"❌ Required environment variable '{var_name}' is missing.")
    return value

# Usage
hf_token = load_env_var("HF_TOKEN")
openai_key = load_env_var("OPENAI_API_KEY")


# ✅ Login to Hugging Face with token
#hf_token = os.getenv("HF_TOKEN")
#if not hf_token:
    #raise ValueError("❌ HF_TOKEN not found in environment")
#else:
    #login(token=hf_token)

# ✅ Load OpenAI key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found")

# ✅ Initialize LLMs
llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo", openai_api_key=openai_key)
teen_memory = ConversationBufferMemory()
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_youtube_links():
    path = os.path.join(os.path.dirname(__file__), "kids_youtube_links.csv")
    if os.path.exists(path):
        return pd.read_csv(path, encoding="latin1")  # use encoding to avoid special char errors
    return pd.DataFrame()

# Load once
df_links = load_youtube_links()

def get_youtube_link(age, theme):
    age = age.strip()
    theme = theme.strip().lower()

    match = df_links[
        (df_links["age_group"].str.strip() == age) &
        (df_links["theme"].str.strip().str.lower() == theme)
    ]
    if not match.empty:
        url = match.iloc[0]["links"]
        desc = match.iloc[0]["description"]
        return f'<b>{desc}</b><br><a href="{url}" target="_blank">🎥 Watch on YouTube</a>'
    return "No video available for this selection."

# Preload genre-based RAG
file_map = {}
def load_genre_indexes():
    for genre in ["mystery", "comedy", "drama", "romance", "fantasy"]:
        path = os.path.join("story_knowledge/teens", f"{genre}.txt")
        if os.path.exists(path):
            retriever = create_retriever(path)
            file_map[genre] = retriever

def create_retriever(path):
    docs = SimpleDirectoryReader(input_files=[path]).load_data()
    return VectorStoreIndex.from_documents(docs).as_query_engine()
    
def read_file_safe(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "❌ File not found."
    except Exception as e:
        return f"❌ Error reading file: {e}"


# TTS fallback
def text_to_speech(text, style="Default"):
    try:
        if not text.strip():
            raise ValueError("Text is empty. Cannot convert to speech.")
        tts = gTTS(text, lang="en")
        audio_path = "story_audio.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"❌ gTTS failed: {e}")
        return None


# Save as PDF
def save_as_pdf(text, name):
    try:
        file_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(file_name)
        return file_name
    except Exception as e:
        print(f"❌ Failed to save PDF: {e}")
        return None

# Save story log
def save_story_to_file(character, story_text, age_group="kids", tone="", theme=""):
    story_data = {
        "character": character,
        "story": story_text,
        "age_group": age_group,
        "tone": tone,
        "theme": theme,
        "timestamp": datetime.now().isoformat()
    }
    with open("story_logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(story_data) + "\n")

# ---------------- Kids Prompt (ChatPromptTemplate) ----------------
kids_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a creative assistant that writes fun, educational, and age-appropriate short stories for children. Use simple language and vivid imagery suitable for ages {age}."
    ),
    HumanMessagePromptTemplate.from_template(
        "Write a {length}-word story for a child aged {age}. "
        "Theme: {theme}. Tone: {tone}. Main character: {name}. "
        "Make it engaging, simple, and creative."
    )
])


def generate_kids(age, theme, tone, name, length):
    messages = kids_prompt.format_messages(age=age, theme=theme, tone=tone, name=name, length=length)
    story = llm(messages).content
    save_story_to_file(name, story, age_group="kids", tone=tone, theme=theme)
    video_html = get_youtube_link(age, theme)  # from your CSV file
    return story, video_html
    
def generate_kids_audio(story):
    return text_to_speech(story)

def generate_kids_pdf(story, name):
    return save_as_pdf(story, name)

# ---------------- Teen Prompt & Sequel (ChatPromptTemplate) ----------------
teen_story_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert teen fiction writer creating captivating and imaginative short stories for teenagers. Include realistic emotions, challenges, and age-appropriate vocabulary."
    ),
    HumanMessagePromptTemplate.from_template(
        "Write a {length}-word creative gentre realated teen story. The main character should be named {name}. "
        "Set the tone to {tone}. Use the following inspiration: {inspiration}."
    )
])


sequel_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a creative assistant generating sequels for teen stories."
    ),
    HumanMessagePromptTemplate.from_template(
        "Write a sequel to the following story:\n\n{last_story}\n\n"
        "Make it {length} words long. Maintain the tone: {tone}. Main character: {name}. Be creative and interesting"
    )
])


def highlight_sections(text):
    for section in ["HOOKS", "CHARACTERS", "SCENES", "TROPES", "SETTINGS"]:
        text = re.sub(fr"\[{section}\]", f"**[{section}]**", text)
    return text

def update_preview(file):
    if file and hasattr(file, "name"):
        try:
            file_text = read_file_safe(file.name)
            return highlight_sections(file_text)
        except Exception as e:
            return f"❌ Error reading file: {e}"
    return ""

def load_default_template():
    file_text = read_file_safe("default_template.txt")
    return highlight_sections(file_text)

def toggle_genre_edit(use_edited):
    return gr.update(interactive=not use_edited)

def load_story_logs():
    logs = []
    if os.path.exists("story_logs.json"):
        with open("story_logs.json", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    return logs[::-1]  # Show newest first

def display_story(index):
    logs = load_story_logs()
    if index < len(logs):
        return logs[index]["story"]
    return "❌ Invalid selection"

# Call once at startup
load_genre_indexes()

# Track last story for sequel
last_story_text = {"story": ""}
def generate_teen(genre, tone, name, length, uploaded_file=None, edited_text="", use_edited=False, make_sequel=False, voice_style="Default"):
    try:
        uploaded_preview = ""
        retriever = None

        if use_edited and edited_text.strip():
            with open("temp_edited.txt", "w", encoding="utf-8") as f:
                f.write(edited_text)
            docs = SimpleDirectoryReader(input_files=["temp_edited.txt"]).load_data()
            retriever = VectorStoreIndex.from_documents(docs).as_query_engine()
            uploaded_preview = highlight_sections(edited_text)

        elif uploaded_file and hasattr(uploaded_file, "name") and not use_edited:
            docs = SimpleDirectoryReader(input_files=[uploaded_file.name]).load_data()
            retriever = VectorStoreIndex.from_documents(docs).as_query_engine()
            with open(uploaded_file.name, "r", encoding="utf-8") as f:
                uploaded_preview = highlight_sections(f.read())

        elif genre.lower() in file_map:
            retriever = file_map[genre.lower()]
            uploaded_preview = f"**Using preloaded genre template: {genre}**"
        else:
            uploaded_preview = "**Using default inspiration.**"

        # Safely extract inspiration
        if retriever:
            result = retriever.query(f"Inspire a {tone} {genre} story")
            inspiration = result.response if hasattr(result, "response") else str(result)
        else:
            inspiration = "A brave teen facing a challenge."

        prompt_len = length or "150"

        # Check if character name is present
        if not name.strip():
            return "❌ Please enter a main character name.", None, None, uploaded_preview

        if make_sequel:
            if not last_story_text["story"]:
                return "❌ No previous story found for sequel generation.", None, None, uploaded_preview

            prompt = f"Write a sequel to the following story:\n\n{last_story_text['story']}\n\nContinue the story in around {prompt_len} words. Start the continuation with '--- Continued ---'"
            sequel_story = llm.predict(prompt).strip()
            full_story = f"{last_story_text['story'].strip()}\n\n--- Continued ---\n\n{sequel_story}"
            save_story_to_file(name, full_story, age_group="teen", tone=tone, theme=genre)
            return full_story, uploaded_preview

        messages = teen_story_prompt.format_messages(
            tone=tone,
            name=name.strip(),
            inspiration=inspiration.strip(),
            length=prompt_len
        )
        story = llm(messages).content.strip()
        last_story_text["story"] = story
        save_story_to_file(name, story, age_group="teen", tone=tone, theme=genre)
        return story, uploaded_preview

    except Exception as e:
        return f"❌ Error: {e}", ""

def generate_teen_audio(story):
    return text_to_speech(story)

def generate_teen_pdf(story, name):
    return save_as_pdf(story, name)





# ---------------- Gradio UI ----------------
# ---------------- Gradio UI ----------------


with gr.Blocks(title="📚 Children's Story Assistant") as app:
    gr.HTML("""
    <style>
        body {
            background-color: #e8f5e9;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            background-color: #c8e6c9;
            color: #2e7d32;
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 1px 1px 8px rgba(0, 0, 0, 0.1);
        }
        label {
            display: inline-block;
            background-color: #a5d6a7;
            color: #1b5e20;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 15px;
            margin-bottom: 4px;
            box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
        }
        select, input[type=radio] {
            border: 1px solid #a5d6a7;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
            background-color: #ffffff;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 10px;
            transition: 0.3s ease-in-out;
            appearance: none;
            background-image: url("data:image/svg+xml;utf8,<svg fill='%232e7d32' height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'><path d='M7 10l5 5 5-5z'/></svg>");
            background-repeat: no-repeat;
            background-position: right 10px center;
            background-size: 20px;
            padding-right: 35px;
            cursor: pointer;
        }
        select:hover {
            border-color: #66bb6a;
            box-shadow: 0 0 8px rgba(76, 175, 80, 0.3);
        }
        textarea, input[type=text] {
            border-radius: 8px;
            border: 1px solid #a5d6a7;
            padding: 10px;
            font-size: 15px;
            background-color: #ffffff;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        }
        button, .gr-button {
            background-color: #4CAF50 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            transition: background-color 0.3s;
            border: none;
        }
        button:hover, .gr-button:hover {
            background-color: #388e3c !important;
        }
        .gr-box {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 12px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
        #banner1-img img {
            max-width: 60%;
            height: auto;
            border-radius: 10px;
            display: block;
            margin: 0 auto;
        }
        input[type=radio],
        input[type=checkbox] {
            accent-color: #4CAF50;
            width: 18px;
            height: 18px;
            margin-right: 8px;
            vertical-align: middle;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        input[type=radio]:hover,
        input[type=checkbox]:hover {
            transform: scale(1.1);
        }
    </style>
    """)

            
   

    with gr.Column():
        gr.Image(value="banner1.jpg", show_label=False, show_download_button=False, container=False, elem_id="banner1-img")

    # ✅ Kids UI
    with gr.Tab("Kids"):
        gr.HTML("""
        <div style='background-color: #f8bbd0; color: black; padding: 12px; border-radius: 8px; font-weight: bold;'>
        ✨ Welcome to the Kids Story Generator!
        </div>
        """)
        age = gr.Dropdown(["1 to 3", "4 to 6", "7 to 9", "10 to 12"], label="Age")
        theme = gr.Dropdown(["Animals", "Adventure", "Magic", "Bedtime"], label="Theme")
        tone = gr.Dropdown(["Happy", "Funny", "Gentle", "Exciting"], label="Tone")
        name = gr.Textbox(label="Character Name")
        length = gr.Radio(["50", "100", "150"], label="Length (words)")
        output = gr.Textbox(label="Story", lines=10)
        video_display = gr.HTML(label="Watch Related Video")
        story_btn = gr.Button("📜 Generate Story")
        audio = gr.Audio(label="Listen", autoplay=True)
        audio_btn = gr.Button("🔊 Generate Voice")
        pdf = gr.File(label="Download PDF")
        pdf_btn = gr.Button("📄 Generate PDF")
        clr = gr.Button("Clear")

        story_btn.click(generate_kids, inputs=[age, theme, tone, name, length], outputs=[output, video_display])

        audio_btn.click(generate_kids_audio, inputs=[output], outputs=[audio])
        pdf_btn.click(generate_kids_pdf, inputs=[output, name], outputs=[pdf])
        clr.click(lambda: ("", None, None), None, [output, audio, pdf])

    # ✅ Teens UI
    with gr.Tab("Teens"):
        gr.HTML("""
        <div style='background-color: #f8bbd0; color: black; padding: 12px; border-radius: 8px; font-weight: bold;'>
        🎒 Welcome to the Teen Story Generator!
        </div>
        """)
        genre2 = gr.Dropdown(["", "Mystery", "Comedy", "Drama", "Romance", "Fantasy"], label="Genre")
        tone2 = gr.Dropdown(["Exciting", "Spooky", "Serious", "Romantic"], label="Tone")
        name2 = gr.Textbox(label="Main Character")
        length2 = gr.Radio(["75", "125", "150"], label="Length (words)")
        uploaded_file = gr.File(label="Upload Template", file_types=[".txt"])
        preview_text = gr.Textbox(label="📄 Template Preview", lines=8, interactive=True)
        use_edited = gr.Checkbox(label="✍️ Use Edited Template", value=False)
        sequel_checkbox = gr.Checkbox(label="🔁 Generate Sequel")
        out2 = gr.Textbox(label="Story", lines=10)
        generate_btn = gr.Button("📜 Generate Story")
        audio2 = gr.Audio(label="Listen", autoplay=True)
        audio2_btn = gr.Button("🔊 Generate Voice")
        pdf2 = gr.File(label="Download PDF")
        pdf2_btn = gr.Button("📄 Generate PDF")
        clr2 = gr.Button("Clear")

        generate_btn.click(generate_teen, inputs=[genre2, tone2, name2, length2, uploaded_file, preview_text, use_edited, sequel_checkbox], outputs=[out2, preview_text])
        audio2_btn.click(generate_teen_audio, inputs=[out2], outputs=[audio2])
        pdf2_btn.click(generate_teen_pdf, inputs=[out2, name2], outputs=[pdf2])
        clr2.click(lambda: ("", None, None, ""), None, [out2, audio2, pdf2, preview_text])
        uploaded_file.change(fn=update_preview, inputs=uploaded_file, outputs=preview_text)
        use_edited.change(fn=toggle_genre_edit, inputs=use_edited, outputs=genre2)
        gr.Button("📄 Load Default Template").click(fn=load_default_template, outputs=preview_text)
        
    with gr.Tab("📚 My Stories"):
        gr.Markdown("### Your Saved Stories")

        story_selector = gr.Dropdown(choices=[], label="Select Story")
        editable_box = gr.Textbox(label="✏️ Edit Story", lines=10)
        tone_input = gr.Dropdown(["Happy", "Exciting", "Serious", "Funny", "Gentle", "Spooky"], label="New Tone")
        length_input = gr.Radio(["50", "75", "100", "125", "150"], label="New Length (words)")
    
        regen_btn = gr.Button("🔄 Regenerate")
        delete_btn = gr.Button("🗑️ Delete Selected Story")
        refresh_btn = gr.Button("🔄 Load Stories")

    # --- Helper functions ---
        def get_story_titles():
            logs = load_story_logs()
            return [f"{i+1}. {log['character']} ({log['age_group']}, {log['tone']}, {log['theme']})" for i, log in enumerate(logs)]

        def refresh_dropdown():
            return gr.update(choices=get_story_titles())

        def show_story_by_title(selected_title):
            logs = load_story_logs()
            titles = get_story_titles()
            if selected_title in titles:
                index = titles.index(selected_title)
                return logs[index]["story"]
            return "❌ Invalid selection"

        def delete_story(selected_title):
            logs = load_story_logs()
            titles = get_story_titles()
            if selected_title in titles:
                index = titles.index(selected_title)
                del logs[index]
                # Overwrite the file with remaining logs
                with open("story_logs.json", "w", encoding="utf-8") as f:
                    for log in logs:
                        f.write(json.dumps(log) + "\n")
                return "", gr.update(choices=get_story_titles())  # Clear story + update dropdown
            return "❌ Could not delete story", gr.update()

        def regenerate_story(original_text, new_tone, new_length):
            if not original_text.strip():
                 return "❌ Please select and edit a story first."
            character_match = re.search(r'"character":\s*"([^"]+)"', original_text)
            character = character_match.group(1) if character_match else "Alex"
            prompt = f"Regenerate the following story with a tone of '{new_tone}' and around {new_length} words:\n\n{original_text}"
            return llm.predict(prompt)

            # --- UI Logic ---
        refresh_btn.click(fn=refresh_dropdown, outputs=story_selector)
        story_selector.change(fn=show_story_by_title, inputs=story_selector, outputs=editable_box)
        delete_btn.click(fn=delete_story, inputs=story_selector, outputs=[editable_box, story_selector])
        regen_btn.click(fn=regenerate_story, inputs=[editable_box, tone_input, length_input], outputs=editable_box)

app.launch()
