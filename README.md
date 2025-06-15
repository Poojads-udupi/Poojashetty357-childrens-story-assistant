# Children's Story Assistant – TinyTales-AI

An AI-powered web app to generate culturally safe, creative, and age-appropriate stories for kids and teens in the UAE. Built using Gradio, OpenAI, Hugging Face, and LangChain.

---

##  Features

### Kids Tab
- Age-based storytelling (1–3, 4–6, 7–9, 10–12)
- Themes: Animals, Adventure, Magic, Bedtime
- Tone customization (Happy, Funny, Gentle, Exciting)
- Character-based storytelling
- Audio playback using gTTS
- PDF story downloads
- YouTube video suggestions based on age & theme
- Content moderation using OpenAI Moderation API

### Teens Tab
- Genre-based story generation (Mystery, Comedy, Drama, Romance, Fantasy)
- Upload custom templates or use default
- Preview & edit templates before generation
- Sequel generation from previous stories
- RAG-powered inspiration using LlamaIndex
- Audio & PDF export
- Moderation guardrails for safe content

### 📚 My Stories
- Saved story logs in JSON
- View, edit, regenerate or delete stories
- Refresh dropdown for updated view

---

## 🔧 Tech Stack

- **Frontend**: Gradio
- **Backend**: Python, LangChain, LlamaIndex
- **LLM**: OpenAI GPT-3.5-turbo
- **Embedding**: Hugging Face Sentence Transformers
- **Voice**: gTTS
- **PDF**: fpdf
- **Content Safety**: OpenAI Moderation API

---

## 🛡️ Safety & Cultural Guardrails

- Custom prompt engineering to respect Islamic/Emirati values
- OpenAI Moderation API to flag unsafe content (violence, sexual, dark themes)
- Age filters and tone control
- Safe default inspiration fallback

---

##  How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
```

### 3. Prepare Data
- Ensure these files exist:
  - `default_template.txt`
  - `kids_youtube_links.csv`
  - Genre templates inside `story_knowledge/teens/`:
    - mystery.txt, comedy.txt, etc.

4. Launch App
```bash
python app.py
```

---

Folder Structure

```
├── app.py
├── story_logs.json
├── default_template.txt
├── kids_youtube_links.csv
├── story_knowledge/
│   └── teens/
│       ├── mystery.txt
│       ├── comedy.txt
│       └── ...
```

---

 Author

Pooja Shetty  
AI Resident | Decoding Data Science  
🔗 GitHub: [Poojads-udupi](https://github.com/Poojads-udupi)

---

 License

MIT License


