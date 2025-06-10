Project Report: Children’s Story Assistant – AI-Powered StoryBot
Project Overview
Children’s Story Assistant is an AI-driven storytelling application built using Gradio, OpenAI's GPT-3.5 model, and Retrieval-Augmented Generation (RAG). It aims to generate creative, age-appropriate stories for kids and teens, with personalized elements like character names, tone, theme, and genre. The app also offers features such as voice narration (gTTS), PDF export, template uploads, and genre-based retrieval to enrich content quality.
________________________________________
Dataset: Custom Curated Story Templates
Instead of a large-scale public dataset, this project uses domain-specific, manually curated text files that serve as knowledge bases for genre-inspired story generation for teens.
•	Directory: story_knowledge/teens/
•	Files: mystery.txt, fantasy.txt, romance.txt, comedy.txt, drama.txt
•	Structure: Text files contain rich narrative elements such as:
o	[HOOKS]
o	[CHARACTERS]
o	[SCENES]
o	[TROPES]
o	[SETTINGS]
These were manually compiled from online story writing resources and AI-assisted creative writing. They form the basis for RAG-style inspiration prompts using LlamaIndex’s SimpleDirectoryReader.
________________________________________ Preprocessing Steps
Since the dataset is unstructured .txt content, minimal preprocessing was applied:
1.	File Parsing:
o	Used SimpleDirectoryReader to read and chunk documents.
o	Stored each genre as a VectorStoreIndex for quick retrieval.
2.	Section Highlighting:
o	Used regex to identify [HOOKS], [CHARACTERS], etc.
o	Applied visual emphasis (e.g., **[HOOKS]**) in Gradio preview using re.sub().
3.	Prompt Engineering:
o	Extracted genre themes to enrich prompt templates for the LLM.
o	Allowed both file upload and live template editing before submission.
4.	Storage:
o	User-generated stories saved in story_logs.json with metadata (tone, age_group, theme, timestamp).
________________________________________
Challenges Encountered
Challenge	Solution
Lack of a structured public dataset	Created lightweight custom genre-specific knowledge base in .txt format
 Integrating RAG with editable templates	Built dual-input logic (upload + inline edit), dynamically created retriever
Real-time feedback from story	Used Gradio Textbox, Audio, and File components for smooth UX
 Performance on repeated queries	Preloaded retrievers in file_map using load_genre_indexes() at startup
Story continuity (for sequels)	Tracked last story with last_story_text to generate coherent follow-ups
________________________________________
✅ Outcome
The project resulted in a modular and user-friendly application with:
•	Dual tabs for Kids and Teens
•	Optional sequel generation
•	Story saving, playback, and PDF download
The app serves as both an educational tool and a creative playground, demonstrating the power of LLMs + RAG + Gradio in a storytelling context.


