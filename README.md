Children's Story Assistant-TinyTales AI

Overview

Children's Story Assistant is an AI-powered Gradio application that generates personalized stories for kids and teenagers. Users can choose age groups, tones, themes, and even upload or edit templates for teen stories. The app also offers PDF and audio story exports, story regeneration, and YouTube video suggestions for kids.

---

Features

 Kids Tab

* Age-based story generation (1–12 years)
* Themes: Animals, Adventure, Magic, Bedtime
* Tones: Happy, Funny, Gentle, Exciting
* Text-to-speech audio using gTTS
* Story PDF download (via FPDF)
* YouTube video recommendations from a local CSV

 Teens Tab

* Genre-based story generation using RAG (VectorStoreIndex)
* Genres: Mystery, Comedy, Drama, Romance, Fantasy
* Upload or edit story inspiration templates
* Regenerate story sequels
* Audio narration and downloadable PDF
  
My Stories Tab

* View saved stories
* Edit and regenerate stories with different tones/lengths
* Delete selected stories

---

Tech Stack

* **Frontend:** Gradio
* **Backend:** Python
* **AI Model:** OpenAI GPT-3.5-Turbo (via LangChain)
* **Embeddings:** Hugging Face Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Indexing:** LlamaIndex
* **Text-to-Speech:** Google TTS (gTTS)
* **PDF Export:** FPDF
* **Env Management:** dotenv

---

Setup Instructions

 1. Clone the repository

```bash
git clone <your-repo-url>
cd childrens-story-assistant
```

 2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Setup Environment Variables

Create a `.env` file with the following:

```
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

 4. Required Files & Folders

* `kids_youtube_links.csv` — Stores YouTube links by age and theme
* `story_knowledge/teens/` — Contains genre-based `.txt` files for teen stories
* `default_template.txt` — Used when no uploaded/edited teen template is provided
* `story_logs.json` — Automatically generated to store story history
* `banner1.jpg` — Image displayed in the app

 5. Run the App

```bash
python app.py
```

---
 File Structure

```
.
├── app.py
├── .env
├── requirements.txt
├── kids_youtube_links.csv
├── story_knowledge/
│   └── teens/
│       ├── mystery.txt
│       ├── comedy.txt
│       └── ...
├── default_template.txt
├── story_logs.json
└── banner1.jpg
```

---

Contribution

Feel free to fork this repository and suggest new features like:

* Multilingual support
* Illustration generation
* User login/authentication
* Cloud-based story saving

---

Credits

* Developed as part of the **Decoding Data Science AI Application Challenge**
* Guided by **Mohammad Arshad**

---

## License

This project is licensed under the MIT License.

---

## Contact

For queries or suggestions, reach out via GitHub Issues or email: \[poojashetty357@gmail.com]



