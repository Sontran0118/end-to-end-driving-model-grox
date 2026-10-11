I'm a beginner on my first real project. Work in small steps, one at a time.
- After each step give me a 5-line plain-English summary so I can explain the code.
- Run new code on a tiny example and show me the output before moving on.
- Don't guess column names, file paths, or devkit functions. Read the installed
  physical_ai_av source and the NVIDIA wiki first. If you can't find it, say so.
- Never write my Hugging Face token into any file or ask me to paste it in chat.
- Write small tests for any math (transforms, projections).
- Save images for me to look at rather than reading pixels yourself.
STORAGE RULES:
- data/ plus the Hugging Face cache must stay under 120 GiB. Check free disk space
  before and after every download.
- Never download without an explicit file list. Never download the whole dataset
  repo or a whole sensor folder.
- Download each file once, to one location. No duplicate copies (HF cache vs local).
- Don't extract whole zip chunks; read only needed files from inside the zip.
- Never delete anything without asking me first.
- Before any download, state the total size and wait for my "OK".
