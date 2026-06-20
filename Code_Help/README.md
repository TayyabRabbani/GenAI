# DSA Code Reviewer & Progress Tracker 
## Live Demo

[Try the application](https://dsa-progress-tracker-fo3m.onrender.com/)

A **personal DSA progress tracker** built with **Django**, **LangChain**, and a
**Hugging Face** chat model (`Qwen/Qwen2.5-Coder-32B-Instruct`).

Paste a problem (name + LeetCode-style statement), your solution, and a difficulty.
The LLM **judges whether your solution is correct** and reviews it. If it's solved,
you pick the **topic** in a popup and it's saved to a **dashboard** that tracks
every problem you've solved, grouped by topic.

---

## How it works

1. Enter **problem name**, **statement**, **solution**, and **difficulty** on the home page.
2. The browser POSTs to `/review/`, which runs the LangChain chain in `reviewer/llm.py`.
3. The model returns `solved`, `score`, `complexity`, `summary`, `issues`,
   `suggestions`, and a suggested `topic`.
4. You see a **Solved / Not Solved** verdict and the full review.
5. **If solved**, a popup with a **dropdown of the 11 topics** lets you confirm the
   topic; it's saved via `/save/`.
6. **/dashboard/** lists every solved problem grouped by topic (collapsible cards),
   with a topic dropdown filter and difficulty stats.

Correctness is **LLM-judged** (no code execution). Re-solving the same problem
updates the existing entry (identity = statement hash), never duplicates.

**Topics:** Arrays · Binary Search · Linked List · Recursion · Dynamic Programming ·
Stack and Queue · Sliding Window · Greedy · Trees · Graphs · Miscellaneous

---

## Tech stack

- Python 3.11+, Django 5
- LangChain (`langchain`, `langchain-core`, `langchain-classic`, `langchain-huggingface`)
- Hugging Face Inference Providers — `Qwen/Qwen2.5-Coder-32B-Instruct`
- PostgreSQL (via `DATABASE_URL`), Gunicorn in production
- Vanilla HTML/CSS/JS front end

---

## Local setup

```bash
git clone <your-repo-url>
cd Rate_code

python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt

copy .env.example .env                              # then edit .env (see below)
```

Fill in `.env` (see [`.env.example`](.env.example)). Generate a `SECRET_KEY` with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
```

Create the database and run the app:

```bash
python manage.py migrate
python manage.py runserver           # http://127.0.0.1:8000
```

Optional admin login at `/admin/`:

```bash
python manage.py createsuperuser
```

### Environment variables

| Variable | Purpose | Local | Render |
|---|---|---|---|
| `SECRET_KEY` | Django secret | random string | random string |
| `DEBUG` | debug mode | `True` | `False` |
| `ALLOWED_HOSTS` | allowed domains (space-separated) | `127.0.0.1 localhost` | `your-app.onrender.com` |
| `DATABASE_URL` | Postgres connection | local DB URL | Render DB URL + `?sslmode=require` |
| `HUGGINGFACEHUB_API_TOKEN` | HF token (Inference Providers) | your token | **omit to disable reviewing** |

---

## Deploying to Render

1. Push this repo to GitHub (`.env` is gitignored — never commit it).
2. On Render: **New → Web Service**, connect the repo, runtime **Python 3**.
3. **Build command:** `pip install -r requirements.txt && python manage.py migrate`
4. **Start command:** `gunicorn dsa_tracker.wsgi:application`
5. Create a **Postgres** instance (Render's, or Neon free) and copy its connection URL.
6. Add the **environment variables** from the table above. Set `DEBUG=False`,
   `ALLOWED_HOSTS` to your Render hostname, and `DATABASE_URL` to the Postgres URL
   with `?sslmode=require`.
7. Deploy. The build runs migrations automatically.

> **Bot/credit protection:** leave `HUGGINGFACEHUB_API_TOKEN` **unset** on Render.
> The reviewer endpoint then returns "unavailable" so no one can spend your HF
> credits, while the dashboard stays fully viewable — ideal for a public showcase.

> **Note:** with `DEBUG=False` the app pages work normally (their CSS is inline),
> but `/admin` will be unstyled unless you add WhiteNoise. Not needed for the dashboard.

---

## Moving your data to the Render database

Your solved problems live in whichever Postgres `DATABASE_URL` points at. To copy
local data up to Render:

```bash
# 1. With .env pointing at your LOCAL Postgres, export your problems:
python manage.py dumpdata reviewer.SolvedProblem --indent 2 -o data.json

# 2. Commit data.json, deploy, then in the Render Shell:
python manage.py loaddata data.json
```

Tip: keep your **local** `.env` pointing at your **local** Postgres for day-to-day
work, and set the Render `DATABASE_URL` only in Render's dashboard — otherwise your
local server edits the production database.

---

## Project layout

```
Rate_code/
├── manage.py
├── dsa_tracker/            # project: settings, urls, wsgi
├── reviewer/               # app: models, views, urls, llm.py, topics.py, templates/
├── .env.example            # template for .env (copy to .env)
├── requirements.txt
└── README.md
```

---

## License

Educational / portfolio use.
